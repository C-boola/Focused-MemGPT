import datetime
import uuid
import inspect
import json
from pathlib import Path
import traceback
from typing import List, Tuple, Optional, cast, Union
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from memgpt.metadata import MetadataStore
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.data_types import AgentState, Message, LLMConfig, EmbeddingConfig, Passage, Preset
from memgpt.models import chat_completion_response
from memgpt.interface import AgentInterface
from memgpt.persistence_manager import LocalStateManager
from memgpt.system import get_login_event, package_function_response, package_summarize_message, get_initial_boot_messages
from memgpt.memory import CoreMemory as InContextMemory, summarize_messages, ArchivalMemory, RecallMemory
from memgpt.llm_api_tools import create, is_context_overflow_error
from memgpt.utils import (
    get_utc_time,
    create_random_username,
    get_tool_call_id,
    get_local_time,
    parse_json,
    united_diff,
    printd,
    count_tokens,
    get_schema_diff,
    validate_function_response,
    verify_first_message_correctness,
    create_uuid_from_string,
    is_utc_datetime,
)
from memgpt.constants import (
    FIRST_MESSAGE_ATTEMPTS,
    JSON_LOADS_STRICT,
    MESSAGE_SUMMARY_WARNING_FRAC,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
    LLM_MAX_TOKENS,
    CLI_WARNING_PREFIX,
    JSON_ENSURE_ASCII,
)
from .errors import LLMError
from .functions.functions import USER_FUNCTIONS_DIR, load_all_function_sets
from memgpt.embeddings import create_embedding, calculate_centroid, calculate_cosine_distances


def link_functions(function_schemas: list):
    """Link function definitions to list of function schemas"""

    # need to dynamically link the functions
    # the saved agent.functions will just have the schemas, but we need to
    # go through the functions library and pull the respective python functions

    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    # agent.functions is a list of schemas (OpenAI kwarg functions style, see: https://platform.openai.com/docs/api-reference/chat/create)
    # [{'name': ..., 'description': ...}, {...}]
    available_functions = load_all_function_sets()
    linked_function_set = {}
    for f_schema in function_schemas:
        # Attempt to find the function in the existing function library
        f_name = f_schema.get("name")
        if f_name is None:
            raise ValueError(f"While loading agent.state.functions encountered a bad function schema object with no name:\n{f_schema}")
        linked_function = available_functions.get(f_name)
        if linked_function is None:
            raise ValueError(
                f"Function '{f_name}' was specified in agent.state.functions, but is not in function library:\n{available_functions.keys()}"
            )
        # Once we find a matching function, make sure the schema is identical
        if json.dumps(f_schema, ensure_ascii=JSON_ENSURE_ASCII) != json.dumps(
            linked_function["json_schema"], ensure_ascii=JSON_ENSURE_ASCII
        ):
            # error_message = (
            #     f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different."
            #     + f"\n>>>agent.state.functions\n{json.dumps(f_schema, indent=2, ensure_ascii=JSON_ENSURE_ASCII)}"
            #     + f"\n>>>function library\n{json.dumps(linked_function['json_schema'], indent=2, ensure_ascii=JSON_ENSURE_ASCII)}"
            # )
            schema_diff = get_schema_diff(f_schema, linked_function["json_schema"])
            error_message = (
                f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different.\n"
                + "".join(schema_diff)
            )

            # NOTE to handle old configs, instead of erroring here let's just warn
            # raise ValueError(error_message)
            printd(error_message)
        linked_function_set[f_name] = linked_function
    return linked_function_set


def initialize_memory(ai_notes: Union[str, None], human_notes: Union[str, None]):
    if ai_notes is None:
        raise ValueError(ai_notes)
    if human_notes is None:
        raise ValueError(human_notes)
    memory = InContextMemory(human_char_limit=CORE_MEMORY_HUMAN_CHAR_LIMIT, persona_char_limit=CORE_MEMORY_PERSONA_CHAR_LIMIT)
    memory.edit_persona(ai_notes)
    memory.edit_human(human_notes)
    return memory


def construct_system_with_memory(
    system: str,
    memory: InContextMemory,
    memory_edit_timestamp: str,
    archival_memory: Optional[ArchivalMemory] = None,
    recall_memory: Optional[RecallMemory] = None,
    include_char_count: bool = True,
):
    full_system_message = "\n".join(
        [
            system,
            "\n",
            f"### Memory [last modified: {memory_edit_timestamp.strip()}]",
            f"{len(recall_memory) if recall_memory else 0} previous messages between you and the user are stored in recall memory (use functions to access them)",
            f"{len(archival_memory) if archival_memory else 0} total memories you created are stored in archival memory (use functions to access them)",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
            f'<persona characters="{len(memory.persona)}/{memory.persona_char_limit}">' if include_char_count else "<persona>",
            memory.persona,
            "</persona>",
            f'<human characters="{len(memory.human)}/{memory.human_char_limit}">' if include_char_count else "<human>",
            memory.human,
            "</human>",
        ]
    )
    return full_system_message


def initialize_message_sequence(
    model: str,
    system: str,
    memory: InContextMemory,
    archival_memory: Optional[ArchivalMemory] = None,
    recall_memory: Optional[RecallMemory] = None,
    memory_edit_timestamp: Optional[str] = None,
    include_initial_boot_message: bool = True,
) -> List[dict]:
    if memory_edit_timestamp is None:
        memory_edit_timestamp = get_local_time()

    full_system_message = construct_system_with_memory(
        system, memory, memory_edit_timestamp, archival_memory=archival_memory, recall_memory=recall_memory
    )
    first_user_message = get_login_event()  # event letting MemGPT know the user just logged in

    if include_initial_boot_message:
        if model is not None and "gpt-3.5" in model:
            initial_boot_messages = get_initial_boot_messages("startup_with_send_message_gpt35")
        else:
            initial_boot_messages = get_initial_boot_messages("startup_with_send_message")
        messages = (
            [
                {"role": "system", "content": full_system_message},
            ]
            + initial_boot_messages
            + [
                {"role": "user", "content": first_user_message},
            ]
        )

    else:
        messages = [
            {"role": "system", "content": full_system_message},
            {"role": "user", "content": first_user_message},
        ]

    return messages


class Agent(object):
    def __init__(
        self,
        interface: AgentInterface,
        # agents can be created from providing agent_state
        agent_state: Optional[AgentState] = None,
        # or from providing a preset (requires preset + extra fields)
        preset: Optional[Preset] = None,
        created_by: Optional[uuid.UUID] = None,
        name: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        mem_mode: Optional[str] = "hybrid",  # new memory mode parameter
        beta: Optional[float] = 0.5,  # new beta parameter for hybrid mode (0-1)
        cluster_summaries: Optional[bool] = False,  # new clustering parameter
        # extras
        messages_total: Optional[int] = None,  # TODO remove?
        first_message_verify_mono: bool = True,  # TODO move to config?
    ):
        # Validate beta parameter
        if beta is not None and (beta < 0.0 or beta > 1.0):
            raise ValueError(f"Beta parameter must be between 0.0 and 1.0, got {beta}")
        
        # An agent can be created from a Preset object
        if preset is not None:
            assert agent_state is None, "Can create an agent from a Preset or AgentState (but both were provided)"
            assert created_by is not None, "Must provide created_by field when creating an Agent from a Preset"
            assert llm_config is not None, "Must provide llm_config field when creating an Agent from a Preset"
            assert embedding_config is not None, "Must provide embedding_config field when creating an Agent from a Preset"

            # if agent_state is also provided, override any preset values
            init_agent_state = AgentState(
                name=name if name else create_random_username(),
                user_id=created_by,
                persona=preset.persona,
                human=preset.human,
                llm_config=llm_config,
                embedding_config=embedding_config,
                preset=preset.name,  # TODO link via preset.id instead of name?
                state={
                    "persona": preset.persona,
                    "human": preset.human,
                    "system": preset.system,
                    "functions": preset.functions_schema,
                    "messages": None,
                    "mem_mode": mem_mode if mem_mode is not None else "focus", # Store mem_mode from preset creation
                    "beta": beta if beta is not None else 0.5, # Store beta from preset creation
                    "cluster_summaries": cluster_summaries if cluster_summaries is not None else False, # Store cluster_summaries from preset creation
                },
            )
            self.mem_mode = mem_mode if mem_mode is not None else "focus"
            self.beta = beta if beta is not None else 0.5
            self.cluster_summaries = cluster_summaries if cluster_summaries is not None else False
            self.prompt_type = "memgpt_default"  # Default prompt type for preset-based agents

        # An agent can also be created directly from AgentState
        elif agent_state is not None:
            assert preset is None, "Can create an agent from a Preset or AgentState (but both were provided)"
            assert agent_state.state is not None and agent_state.state != {}, "AgentState.state cannot be empty"

            # Assume the agent_state passed in is formatted correctly
            init_agent_state = agent_state
            self.mem_mode = agent_state.state.get("mem_mode", "fifo") # Load mem_mode from state
            self.beta = agent_state.state.get("beta", 0.5) # Load beta from state
            self.cluster_summaries = agent_state.state.get("cluster_summaries", False) # Load cluster_summaries from state
            self.prompt_type = agent_state.state.get("prompt_type", "memgpt_default") # Load prompt_type from state

        else:
            raise ValueError("Both Preset and AgentState were null (must provide one or the other)")

        # Hold a copy of the state that was used to init the agent
        self.agent_state = init_agent_state

        # gpt-4, gpt-3.5-turbo, ...
        self.model = self.agent_state.llm_config.model
        print(f"Agent '{self.agent_state.name}' is using LLM model: {self.model}")

        # Store the system instructions (used to rebuild memory)
        if "system" not in self.agent_state.state:
            raise ValueError(f"'system' not found in provided AgentState")
        self.system = self.agent_state.state["system"]

        if "functions" not in self.agent_state.state:
            raise ValueError(f"'functions' not found in provided AgentState")
        # Store the functions schemas (this is passed as an argument to ChatCompletion)
        self.functions = self.agent_state.state["functions"]  # these are the schema
        # Link the actual python functions corresponding to the schemas
        self.functions_python = {k: v["python_function"] for k, v in link_functions(function_schemas=self.functions).items()}
        assert all([callable(f) for k, f in self.functions_python.items()]), self.functions_python

        # Initialize the memory object
        if "persona" not in self.agent_state.state:
            raise ValueError(f"'persona' not found in provided AgentState")
        if "human" not in self.agent_state.state:
            raise ValueError(f"'human' not found in provided AgentState")
        self.memory = initialize_memory(ai_notes=self.agent_state.state["persona"], human_notes=self.agent_state.state["human"])

        # Interface must implement:
        # - internal_monologue
        # - assistant_message
        # - function_message
        # ...
        # Different interfaces can handle events differently
        # e.g., print in CLI vs send a discord message with a discord bot
        self.interface = interface

        # Create the persistence manager object based on the AgentState info
        # TODO
        self.persistence_manager = LocalStateManager(agent_state=self.agent_state)

        # State needed for heartbeat pausing
        self.pause_heartbeats_start = None
        self.pause_heartbeats_minutes = 0

        self.first_message_verify_mono = first_message_verify_mono

        # Controls if the convo memory pressure warning is triggered
        # When an alert is sent in the message queue, set this to True (to avoid repeat alerts)
        # When the summarizer is run, set this back to False (to reset)
        self.agent_alerted_about_memory_pressure = False

        self._messages: List[Message] = []

        # Once the memory object is initialized, use it to "bake" the system message
        if "messages" in self.agent_state.state and self.agent_state.state["messages"] is not None:
            # print(f"Agent.__init__ :: loading, state={agent_state.state['messages']}")
            if not isinstance(self.agent_state.state["messages"], list):
                raise ValueError(f"'messages' in AgentState was bad type: {type(self.agent_state.state['messages'])}")
            assert all([isinstance(msg, str) for msg in self.agent_state.state["messages"]])

            # Convert to IDs, and pull from the database
            raw_messages = [
                self.persistence_manager.recall_memory.storage.get(id=uuid.UUID(msg_id)) for msg_id in self.agent_state.state["messages"]
            ]
            assert all([isinstance(msg, Message) for msg in raw_messages]), (raw_messages, self.agent_state.state["messages"])
            self._messages.extend([cast(Message, msg) for msg in raw_messages if msg is not None])

            for m in self._messages:
                # assert is_utc_datetime(m.created_at), f"created_at on message for agent {self.agent_state.name} isn't UTC:\n{vars(m)}"
                # TODO eventually do casting via an edit_message function
                if not is_utc_datetime(m.created_at):
                    printd(f"Warning - created_at on message for agent {self.agent_state.name} isn't UTC (text='{m.text}')")
                    m.created_at = m.created_at.replace(tzinfo=datetime.timezone.utc)

        else:
            # print(f"Agent.__init__ :: creating, state={agent_state.state['messages']}")
            init_messages = initialize_message_sequence(
                self.model,
                self.system,
                self.memory,
            )
            init_messages_objs = []
            for msg in init_messages:
                init_messages_objs.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id, user_id=self.agent_state.user_id, model=self.model, openai_message_dict=msg
                    )
                )
            assert all([isinstance(msg, Message) for msg in init_messages_objs]), (init_messages_objs, init_messages)
            self.messages_total = 0
            self._append_to_messages(added_messages=[cast(Message, msg) for msg in init_messages_objs if msg is not None])

            for m in self._messages:
                assert is_utc_datetime(m.created_at), f"created_at on message for agent {self.agent_state.name} isn't UTC:\n{vars(m)}"
                # TODO eventually do casting via an edit_message function
                if not is_utc_datetime(m.created_at):
                    printd(f"Warning - created_at on message for agent {self.agent_state.name} isn't UTC (text='{m.text}')")
                    m.created_at = m.created_at.replace(tzinfo=datetime.timezone.utc)

        # Keep track of the total number of messages throughout all time
        self.messages_total = messages_total if messages_total is not None else (len(self._messages) - 1)  # (-system)
        # self.messages_total_init = self.messages_total
        self.messages_total_init = len(self._messages) - 1
        printd(f"Agent initialized, self.messages_total={self.messages_total}")

        # Create the agent in the DB
        # self.save()
        self.update_state()

    @property
    def messages(self) -> List[dict]:
        """Getter method that converts the internal Message list into OpenAI-style dicts"""
        return [msg.to_openai_dict() for msg in self._messages]

    @messages.setter
    def messages(self, value):
        raise Exception("Modifying message list directly not allowed")

    def _trim_messages(self, num):
        """Trim messages from the front, not including the system message"""
        self.persistence_manager.trim_messages(num)

        new_messages = [self._messages[0]] + self._messages[num:]
        self._messages = new_messages

    def _prepend_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.prepend to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.prepend_to_messages(added_messages)

        new_messages = [self._messages[0]] + added_messages + self._messages[1:]  # prepend (no system)
        self._messages = new_messages
        self.messages_total += len(added_messages)  # still should increment the message counter (summaries are additions too)

    def _append_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.append to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.append_to_messages(added_messages)

        # strip extra metadata if it exists
        # for msg in added_messages:
        # msg.pop("api_response", None)
        # msg.pop("api_args", None)
        new_messages = self._messages + added_messages  # append

        self._messages = new_messages
        self.messages_total += len(added_messages)

    def append_to_messages(self, added_messages: List[dict]):
        """An external-facing message append, where dict-like messages are first converted to Message objects"""
        added_messages_objs = [
            Message.dict_to_message(
                agent_id=self.agent_state.id,
                user_id=self.agent_state.user_id,
                model=self.model,
                openai_message_dict=msg,
            )
            for msg in added_messages
        ]
        self._append_to_messages(added_messages_objs)

    def _swap_system_message(self, new_system_message: Message):
        assert isinstance(new_system_message, Message)
        assert new_system_message.role == "system", new_system_message
        assert self._messages[0].role == "system", self._messages

        self.persistence_manager.swap_system_message(new_system_message)

        new_messages = [new_system_message] + self._messages[1:]  # swap index 0 (system)
        self._messages = new_messages

    def _get_ai_reply(
        self,
        message_sequence: List[dict],
        function_call: str = "auto",
        first_message: bool = False,  # hint
    ) -> chat_completion_response.ChatCompletionResponse:
        """Get response from LLM API"""
        try:
            response = create(
                agent_state=self.agent_state,
                messages=message_sequence,
                functions=self.functions,
                functions_python=self.functions_python,
                function_call=function_call,
                # hint
                first_message=first_message,
            )
            # special case for 'length'
            if response.choices[0].finish_reason == "length":
                raise Exception("Finish reason was length (maximum context length)")

            # catches for soft errors
            if response.choices[0].finish_reason not in ["stop", "function_call", "tool_calls"]:
                raise Exception(f"API call finish with bad finish reason: {response}")

            # unpack with response.choices[0].message.content
            return response
        except Exception as e:
            raise e

    def _handle_ai_response(
        self, response_message: chat_completion_response.Message, override_tool_call_id: bool = True
    ) -> Tuple[List[Message], bool, bool]:
        """Handles parsing and function execution"""

        messages = []  # append these to the history when done

        # Step 2: check if LLM wanted to call a function
        if response_message.function_call or (response_message.tool_calls is not None and len(response_message.tool_calls) > 0):
            if response_message.function_call:
                raise DeprecationWarning(response_message)
            if response_message.tool_calls is not None and len(response_message.tool_calls) > 1:
                # raise NotImplementedError(f">1 tool call not supported")
                # TODO eventually support sequential tool calling
                printd(f">1 tool call not supported, using index=0 only\n{response_message.tool_calls}")
                response_message.tool_calls = [response_message.tool_calls[0]]
            assert response_message.tool_calls is not None and len(response_message.tool_calls) > 0

            # generate UUID for tool call
            if override_tool_call_id or response_message.function_call:
                tool_call_id = get_tool_call_id()  # needs to be a string for JSON
                response_message.tool_calls[0].id = tool_call_id
            else:
                tool_call_id = response_message.tool_calls[0].id
                assert tool_call_id is not None  # should be defined

            # only necessary to add the tool_cal_id to a function call (antipattern)
            # response_message_dict = response_message.model_dump()
            # response_message_dict["tool_call_id"] = tool_call_id

            # role: assistant (requesting tool call, set tool call ID)
            messages.append(
                # NOTE: we're recreating the message here
                # TODO should probably just overwrite the fields?
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            printd(f"Function call message: {messages[-1]}")

            # The content if then internal monologue, not chat
            self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors

            # Failure case 1: function name is wrong
            function_call = (
                response_message.function_call if response_message.function_call is not None else response_message.tool_calls[0].function
            )
            function_name = function_call.name
            printd(f"Request to call function {function_name} with tool_call_id: {tool_call_id}")
            try:
                function_to_call = self.functions_python[function_name]
            except KeyError as e:
                error_msg = f"No function named {function_name}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                return messages, False, True  # force a heartbeat to allow agent to handle error

            # Failure case 2: function name is OK, but function args are bad JSON
            try:
                raw_function_args = function_call.arguments
                function_args = parse_json(raw_function_args)
            except Exception as e:
                error_msg = f"Error parsing JSON for function '{function_name}' arguments: {function_call.arguments}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                return messages, False, True  # force a heartbeat to allow agent to handle error

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            heartbeat_request = function_args.pop("request_heartbeat", None)
            if not (isinstance(heartbeat_request, bool) or heartbeat_request is None):
                printd(
                    f"{CLI_WARNING_PREFIX}'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )
                heartbeat_request = False

            # Failure case 3: function failed during execution
            # NOTE: the msg_obj associated with the "Running " message is the prior assistant message, not the function/tool role message
            #       this is because the function/tool role message is only created once the function/tool has executed/returned
            self.interface.function_message(f"Running {function_name}({function_args})", msg_obj=messages[-1])
            try:
                spec = inspect.getfullargspec(function_to_call).annotations

                for name, arg in function_args.items():
                    if isinstance(function_args[name], dict):
                        function_args[name] = spec[name](**function_args[name])

                function_args["self"] = self  # need to attach self to arg since it's dynamically linked

                function_response = function_to_call(**function_args)
                if function_name in ["conversation_search", "conversation_search_date", "archival_memory_search"]:
                    # with certain functions we rely on the paging mechanism to handle overflow
                    truncate = False
                else:
                    # but by default, we add a truncation safeguard to prevent bad functions from
                    # overflow the agent context window
                    truncate = True
                function_response_string = validate_function_response(function_response, truncate=truncate)
                function_args.pop("self", None)
                function_response = package_function_response(True, function_response_string)
                function_failed = False
            except Exception as e:
                function_args.pop("self", None)
                # error_msg = f"Error calling function {function_name} with args {function_args}: {str(e)}"
                # Less detailed - don't provide full args, idea is that it should be in recent context so no need (just adds noise)
                error_msg = f"Error calling function {function_name}: {str(e)}"
                error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                printd(error_msg_user)
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Ran {function_name}({function_args})", msg_obj=messages[-1])
                self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                return messages, False, True  # force a heartbeat to allow agent to handle error

            # If no failures happened along the way: ...
            # Step 4: send the info on the function call and function response to GPT
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                        "tool_call_id": tool_call_id,
                    },
                )
            )  # extend conversation with function response
            self.interface.function_message(f"Ran {function_name}({function_args})", msg_obj=messages[-1])
            self.interface.function_message(f"Success: {function_response_string}", msg_obj=messages[-1])

        else:
            # Standard non-function reply
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])
            heartbeat_request = False
            function_failed = False

        return messages, heartbeat_request, function_failed

    def step(
        self,
        user_message: Union[Message, str],  # NOTE: should be json.dump(dict)
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,  # if True, return dicts, if False, return Message objects
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
    ) -> Tuple[List[Union[dict, Message]], bool, bool, bool]:
        """Top-level event message handler for the MemGPT agent"""

        def strip_name_field_from_user_message(user_message_text: str) -> Tuple[str, Optional[str]]:
            """If 'name' exists in the JSON string, remove it and return the cleaned text + name value"""
            try:
                user_message_json = dict(json.loads(user_message_text, strict=JSON_LOADS_STRICT))
                # Special handling for AutoGen messages with 'name' field
                # Treat 'name' as a special field
                # If it exists in the input message, elevate it to the 'message' level
                name = user_message_json.pop("name", None)
                clean_message = json.dumps(user_message_json, ensure_ascii=JSON_ENSURE_ASCII)

            except Exception as e:
                print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed with: {e}")

            return clean_message, name

        def validate_json(user_message_text: str, raise_on_error: bool) -> str:
            try:
                user_message_json = dict(json.loads(user_message_text, strict=JSON_LOADS_STRICT))
                user_message_json_val = json.dumps(user_message_json, ensure_ascii=JSON_ENSURE_ASCII)
                return user_message_json_val
            except Exception as e:
                print(f"{CLI_WARNING_PREFIX}couldn't parse user input message as JSON: {e}")
                if raise_on_error:
                    raise e

        try:
            # Step 0: add user message
            if user_message is not None:
                if isinstance(user_message, Message):
                    # Validate JSON via save/load
                    user_message_text = validate_json(user_message.text, False)
                    cleaned_user_message_text, name = strip_name_field_from_user_message(user_message_text)

                    if name is not None:
                        # Update Message object
                        user_message.text = cleaned_user_message_text
                        user_message.name = name

                    # Recreate timestamp
                    if recreate_message_timestamp:
                        user_message.created_at = get_utc_time()

                elif isinstance(user_message, str):
                    # Validate JSON via save/load
                    user_message = validate_json(user_message, False)
                    cleaned_user_message_text, name = strip_name_field_from_user_message(user_message)

                    # If user_message['name'] is not None, it will be handled properly by dict_to_message
                    # So no need to run strip_name_field_from_user_message

                    # Create the associated Message object (in the database)
                    user_message = Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={"role": "user", "content": cleaned_user_message_text, "name": name},
                    )

                else:
                    raise ValueError(f"Bad type for user_message: {type(user_message)}")

                self.interface.user_message(user_message.text, msg_obj=user_message)

                input_message_sequence = self.messages + [user_message.to_openai_dict()]
            # Alternatively, the requestor can send an empty user message
            else:
                input_message_sequence = self.messages

            if len(input_message_sequence) > 1 and input_message_sequence[-1]["role"] != "user":
                printd(f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue")

            # Step 1: send the conversation and available functions to GPT
            if not skip_verify and (first_message or self.messages_total == self.messages_total_init):
                printd(f"This is the first message. Running extra verifier on AI response.")
                counter = 0
                while True:
                    response = self._get_ai_reply(
                        message_sequence=input_message_sequence,
                        first_message=True,  # passed through to the prompt formatter
                    )
                    if verify_first_message_correctness(response, require_monologue=self.first_message_verify_mono):
                        break

                    counter += 1
                    if counter > first_message_retry_limit:
                        raise Exception(f"Hit first message retry limit ({first_message_retry_limit})")

            else:
                response = self._get_ai_reply(
                    message_sequence=input_message_sequence,
                )

            # Step 2: check if LLM wanted to call a function
            # (if yes) Step 3: call the function
            # (if yes) Step 4: send the info on the function call and function response to LLM
            response_message = response.choices[0].message
            response_message.model_copy()  # TODO why are we copying here?
            all_response_messages, heartbeat_request, function_failed = self._handle_ai_response(response_message)

            # Add the extra metadata to the assistant response
            # (e.g. enough metadata to enable recreating the API call)
            # assert "api_response" not in all_response_messages[0]
            # all_response_messages[0]["api_response"] = response_message_copy
            # assert "api_args" not in all_response_messages[0]
            # all_response_messages[0]["api_args"] = {
            #     "model": self.model,
            #     "messages": input_message_sequence,
            #     "functions": self.functions,
            # }

            # Step 4: extend the message history
            if user_message is not None:
                if isinstance(user_message, Message):
                    all_new_messages = [user_message] + all_response_messages
                else:
                    raise ValueError(type(user_message))
            else:
                all_new_messages = all_response_messages

            # Check the memory pressure and potentially issue a memory pressure warning
            current_total_tokens = response.usage.total_tokens
            active_memory_warning = False
            # We can't do summarize logic properly if context_window is undefined
            if self.agent_state.llm_config.context_window is None:
                # Fallback if for some reason context_window is missing, just set to the default
                print(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
                print(f"{self.agent_state}")
                self.agent_state.llm_config.context_window = (
                    LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
                )
            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window):
                printd(
                    f"{CLI_WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )
                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this
            else:
                printd(
                    f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )

            self._append_to_messages(all_new_messages)
            messages_to_return = [msg.to_openai_dict() for msg in all_new_messages] if return_dicts else all_new_messages
            return messages_to_return, heartbeat_request, function_failed, active_memory_warning, response.usage.completion_tokens

        except Exception as e:
            printd(f"step() failed\nuser_message = {user_message}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                # A separate API call to run a summarizer
                if self.mem_mode == "fifo":
                    print("=" * 70)
                    print("  ||  CONTEXT OVERFLOW - FIFO MODE ACTIVATED - PRE-SUMMARY DIAGNOSTICS  ||")
                    print("=" * 70)
                    
                    original_user_message_text = user_message.text if isinstance(user_message, Message) else str(user_message)
                    print(f"FIFO Pre-Summary: Original user_message causing overflow (text snippet): {original_user_message_text[:200]}...")
                    print(f"FIFO Pre-Summary: Current self._messages count: {len(self._messages)}")
                    current_messages_tokens = 0
                    for i, msg_obj in enumerate(self._messages):
                        role = msg_obj.role
                        content_snippet = msg_obj.text[:100] if msg_obj.text else "[No text content]"
                        msg_tokens = count_tokens(str(msg_obj.to_openai_dict()))
                        current_messages_tokens += msg_tokens
                        print(f"  [{i}] Role: {role}, ID: {msg_obj.id}, Tokens: {msg_tokens}, Content: '{content_snippet}...'")
                    print(f"FIFO Pre-Summary: Total tokens in self._messages: {current_messages_tokens}")
                    
                    # Estimate tokens if current user_message was added
                    user_message_tokens_for_diag = count_tokens(str(user_message.to_openai_dict() if isinstance(user_message, Message) else {"role": "user", "content": str(user_message)}))
                    print(f"FIFO Pre-Summary: Tokens in current user_message: {user_message_tokens_for_diag}")
                    print(f"FIFO Pre-Summary: Estimated total tokens IF user_message was appended (before summary): {current_messages_tokens + user_message_tokens_for_diag}")
                    
                    print(f"Context overflow in FIFO mode, calling summarize_messages_inplace().")
                    self.summarize_messages_inplace()
                    
                    print("-" * 70)
                    print("  ||  FIFO MODE - POST-SUMMARY DIAGNOSTICS (BEFORE RETRY)  ||")
                    print("-" * 70)
                    
                    user_message_for_retry_text = user_message.text if isinstance(user_message, Message) else str(user_message)
                    print(f"FIFO Post-Summary: User_message for retry (text snippet): {user_message_for_retry_text[:200]}...")
                    print(f"FIFO Post-Summary: New self._messages count: {len(self._messages)}")
                    new_messages_tokens = 0
                    for i, msg_obj in enumerate(self._messages):
                        role = msg_obj.role
                        content_snippet = msg_obj.text[:100] if msg_obj.text else "[No text content]"
                        msg_tokens = count_tokens(str(msg_obj.to_openai_dict()))
                        new_messages_tokens += msg_tokens
                        print(f"  [{i}] Role: {role}, ID: {msg_obj.id}, Tokens: {msg_tokens}, Content: '{content_snippet}...'")
                    print(f"FIFO Post-Summary: Total tokens in new self._messages: {new_messages_tokens}")
                    
                    user_message_tokens_for_retry_diag = count_tokens(str(user_message.to_openai_dict() if isinstance(user_message, Message) else {"role": "user", "content": str(user_message)}))
                    # This is the user_message that will be appended in the retry call
                    print(f"FIFO Post-Summary: Tokens in user_message for retry: {user_message_tokens_for_retry_diag}")
                    estimated_tokens_for_retry_api_call = new_messages_tokens + user_message_tokens_for_retry_diag
                    print(f"FIFO Post-Summary: Estimated total tokens for API call in retry: {estimated_tokens_for_retry_api_call}")
                    print(f"FIFO Post-Summary: LLM context window: {self.agent_state.llm_config.context_window}")
                    
                    # Try step again
                    return self.step(user_message, first_message=first_message, return_dicts=return_dicts, recreate_message_timestamp=recreate_message_timestamp)
                elif self.mem_mode == "focus":
                    print("=" * 70)
                    print("  ||  CONTEXT OVERFLOW - FOCUS MODE ACTIVATED - PRE-SUMMARY DIAGNOSTICS  ||")
                    print("=" * 70)
                    
                    original_user_message_text = user_message.text if isinstance(user_message, Message) else str(user_message)
                    print(f"Focus Pre-Summary: Original user_message causing overflow (text snippet): {original_user_message_text[:200]}...")
                    print(f"Focus Pre-Summary: Current self._messages count: {len(self._messages)}")
                    current_messages_tokens = 0
                    for i, msg_obj in enumerate(self._messages):
                        role = msg_obj.role
                        content_snippet = msg_obj.text[:100] if msg_obj.text else "[No text content]"
                        msg_tokens = count_tokens(str(msg_obj.to_openai_dict()))
                        current_messages_tokens += msg_tokens
                        print(f"  [{i}] Role: {role}, ID: {msg_obj.id}, Tokens: {msg_tokens}, Content: '{content_snippet}...'")
                    print(f"Focus Pre-Summary: Total tokens in self._messages: {current_messages_tokens}")
                    
                    # Estimate tokens if current user_message was added
                    user_message_tokens_for_diag = count_tokens(str(user_message.to_openai_dict() if isinstance(user_message, Message) else {"role": "user", "content": str(user_message)}))
                    print(f"Focus Pre-Summary: Tokens in current user_message: {user_message_tokens_for_diag}")
                    print(f"Focus Pre-Summary: Estimated total tokens IF user_message was appended (before summary): {current_messages_tokens + user_message_tokens_for_diag}")
                    
                    print(f"Context overflow in 'focus' mode. Attempting to generate message pair embeddings.")
                    try:
                        message_pair_embeddings_with_ids = self._create_robust_message_pair_embeddings(
                            message_sequence=self._messages, 
                            embedding_config=self.agent_state.embedding_config
                        )
                        print(f"Generated {len(message_pair_embeddings_with_ids)} message pair embeddings for focus mode.")

                        if not message_pair_embeddings_with_ids:
                            print("No message pair embeddings generated. Defaulting to FIFO summarization or error.")
                            raise e 

                        # Extract just the embedding vectors for centroid and distance calculation
                        actual_embedding_vectors = [pair[2] for pair in message_pair_embeddings_with_ids]

                        # Calculate centroid
                        centroid_vec = calculate_centroid(actual_embedding_vectors)
                        print('Centroid vector calculated successfully.')

                        if centroid_vec is None:
                            print("Centroid calculation failed (no vectors or other issue). Defaulting to FIFO or error.")
                            raise e
                        
                        # Convert list of list to list of np.array for distance calculation
                        np_embedding_vectors = [np.array(vec) for vec in actual_embedding_vectors]
                        distances = calculate_cosine_distances(centroid_vec, np_embedding_vectors)
                        
                        # Combine message IDs with their distances
                        messages_with_distances = []
                        for i, pair_data in enumerate(message_pair_embeddings_with_ids):
                            user_msg_id = pair_data[0]
                            assistant_msg_id = pair_data[1]
                            messages_with_distances.append({
                                "user_msg_id": user_msg_id,
                                "assistant_msg_id": assistant_msg_id,
                                "distance": distances[i]
                            })
                        
                        # Sort by distance, descending (furthest first)
                        sorted_messages_by_distance = sorted(messages_with_distances, key=lambda x: x["distance"], reverse=True)
                        
                        print(f"Message pairs sorted by distance from centroid (furthest first):")
                        for item in sorted_messages_by_distance:
                            print(f"  User_ID: {item['user_msg_id']}, Asst_ID: {item['assistant_msg_id']}, Distance: {item['distance']:.4f}")

                        # Implement summarization based on these sorted messages.
                        try:
                            self.summarize_messages_focus_inplace(sorted_messages_by_distance)
                            
                            print("-" * 70)
                            print("  ||  FOCUS MODE - POST-SUMMARY DIAGNOSTICS (BEFORE RETRY)  ||")
                            print("-" * 70)
                            
                            user_message_for_retry_text = user_message.text if isinstance(user_message, Message) else str(user_message)
                            print(f"Focus Post-Summary: User_message for retry (text snippet): {user_message_for_retry_text[:200]}...")
                            print(f"Focus Post-Summary: New self._messages count: {len(self._messages)}")
                            new_messages_tokens = 0
                            for i, msg_obj in enumerate(self._messages):
                                role = msg_obj.role
                                content_snippet = msg_obj.text[:100] if msg_obj.text else "[No text content]"
                                msg_tokens = count_tokens(str(msg_obj.to_openai_dict()))
                                new_messages_tokens += msg_tokens
                                print(f"  [{i}] Role: {role}, ID: {msg_obj.id}, Tokens: {msg_tokens}, Content: '{content_snippet}...'")
                            print(f"Focus Post-Summary: Total tokens in new self._messages: {new_messages_tokens}")
                            
                            user_message_tokens_for_retry_diag = count_tokens(str(user_message.to_openai_dict() if isinstance(user_message, Message) else {"role": "user", "content": str(user_message)}))
                            print(f"Focus Post-Summary: Tokens in user_message for retry: {user_message_tokens_for_retry_diag}")
                            estimated_tokens_for_retry_api_call = new_messages_tokens + user_message_tokens_for_retry_diag
                            print(f"Focus Post-Summary: Estimated total tokens for API call in retry: {estimated_tokens_for_retry_api_call}")
                            print(f"Focus Post-Summary: LLM context window: {self.agent_state.llm_config.context_window}")
                            
                            # Try step again
                            print("Focus mode: Summarization complete, retrying step.")
                            return self.step(user_message, first_message=first_message, return_dicts=return_dicts, recreate_message_timestamp=recreate_message_timestamp)
                        except LLMError as focus_sum_e: # Catch specific error from our focus summarizer
                            print(f"{CLI_WARNING_PREFIX}Focus mode summarization failed: {focus_sum_e}. Re-raising original overflow error to halt.")
                            raise e # Re-raise the original overflow error (e, not focus_sum_e, to keep original trace if needed)
                        except Exception as general_sum_e: # Catch any other unexpected error during summarization
                            print(f"{CLI_WARNING_PREFIX}Unexpected error during focus mode summarization: {general_sum_e}. Re-raising original overflow error to halt.")
                            raise e # Re-raise the original overflow error
                        
                    except Exception as emb_e:
                        print(f"Error during focus mode context overflow handling (embedding/distance phase): {emb_e}")
                        # If embedding generation or processing itself fails, re-raise the original overflow error.
                        raise e # Re-raise the original overflow error
                elif self.mem_mode == "hybrid":
                    print("=" * 70)
                    print("  ||  CONTEXT OVERFLOW - HYBRID MODE ACTIVATED - PRE-SUMMARY DIAGNOSTICS  ||")
                    print("=" * 70)
                    
                    original_user_message_text = user_message.text if isinstance(user_message, Message) else str(user_message)
                    print(f"Hybrid Pre-Summary: Original user_message causing overflow (text snippet): {original_user_message_text[:200]}...")
                    print(f"Hybrid Pre-Summary: Current self._messages count: {len(self._messages)}")
                    print(f"Hybrid Pre-Summary: Beta parameter: {self.beta}")
                    current_messages_tokens = 0
                    for i, msg_obj in enumerate(self._messages):
                        role = msg_obj.role
                        content_snippet = msg_obj.text[:100] if msg_obj.text else "[No text content]"
                        msg_tokens = count_tokens(str(msg_obj.to_openai_dict()))
                        current_messages_tokens += msg_tokens
                        print(f"  [{i}] Role: {role}, ID: {msg_obj.id}, Tokens: {msg_tokens}, Content: '{content_snippet}...'")
                    print(f"Hybrid Pre-Summary: Total tokens in self._messages: {current_messages_tokens}")
                    
                    # Estimate tokens if current user_message was added
                    user_message_tokens_for_diag = count_tokens(str(user_message.to_openai_dict() if isinstance(user_message, Message) else {"role": "user", "content": str(user_message)}))
                    print(f"Hybrid Pre-Summary: Tokens in current user_message: {user_message_tokens_for_diag}")
                    print(f"Hybrid Pre-Summary: Estimated total tokens IF user_message was appended (before summary): {current_messages_tokens + user_message_tokens_for_diag}")
                    
                    print(f"Context overflow in 'hybrid' mode. Attempting hybrid summarization with beta={self.beta}.")
                    try:
                        self.summarize_messages_hybrid_inplace()
                        
                        print("-" * 70)
                        print("  ||  HYBRID MODE - POST-SUMMARY DIAGNOSTICS (BEFORE RETRY)  ||")
                        print("-" * 70)
                        
                        user_message_for_retry_text = user_message.text if isinstance(user_message, Message) else str(user_message)
                        print(f"Hybrid Post-Summary: User_message for retry (text snippet): {user_message_for_retry_text[:200]}...")
                        print(f"Hybrid Post-Summary: New self._messages count: {len(self._messages)}")
                        new_messages_tokens = 0
                        for i, msg_obj in enumerate(self._messages):
                            role = msg_obj.role
                            content_snippet = msg_obj.text[:100] if msg_obj.text else "[No text content]"
                            msg_tokens = count_tokens(str(msg_obj.to_openai_dict()))
                            new_messages_tokens += msg_tokens
                            print(f"  [{i}] Role: {role}, ID: {msg_obj.id}, Tokens: {msg_tokens}, Content: '{content_snippet}...'")
                        print(f"Hybrid Post-Summary: Total tokens in new self._messages: {new_messages_tokens}")
                        
                        user_message_tokens_for_retry_diag = count_tokens(str(user_message.to_openai_dict() if isinstance(user_message, Message) else {"role": "user", "content": str(user_message)}))
                        print(f"Hybrid Post-Summary: Tokens in user_message for retry: {user_message_tokens_for_retry_diag}")
                        estimated_tokens_for_retry_api_call = new_messages_tokens + user_message_tokens_for_retry_diag
                        print(f"Hybrid Post-Summary: Estimated total tokens for API call in retry: {estimated_tokens_for_retry_api_call}")
                        print(f"Hybrid Post-Summary: LLM context window: {self.agent_state.llm_config.context_window}")
                        
                        # Try step again
                        print("Hybrid mode: Summarization complete, retrying step.")
                        return self.step(user_message, first_message=first_message, return_dicts=return_dicts, recreate_message_timestamp=recreate_message_timestamp)
                    except LLMError as hybrid_sum_e: # Catch specific error from our hybrid summarizer
                        print(f"{CLI_WARNING_PREFIX}Hybrid mode summarization failed: {hybrid_sum_e}. Re-raising original overflow error to halt.")
                        raise e # Re-raise the original overflow error (e, not hybrid_sum_e, to keep original trace if needed)
                    except Exception as general_sum_e: # Catch any other unexpected error during summarization
                        print(f"{CLI_WARNING_PREFIX}Unexpected error during hybrid mode summarization: {general_sum_e}. Re-raising original overflow error to halt.")
                        raise e # Re-raise the original overflow error
                else:
                    # Default to FIFO if mode is unrecognized
                    print(f"Context overflow in unrecognized mem_mode '{self.mem_mode}', defaulting to FIFO summarization.")
                    self.summarize_messages_inplace()
                    # Try step again
                    return self.step(user_message, first_message=first_message, return_dicts=return_dicts, recreate_message_timestamp=recreate_message_timestamp)
            else:
                printd(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e

    def summarize_messages_inplace(self, cutoff=None, preserve_last_N_messages=True, disallow_tool_as_first=True):
        assert self.messages[0]["role"] == "system", f"self.messages[0] should be system (instead got {self.messages[0]})"

        # Start at index 1 (past the system message),
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # Do not allow truncation of the last N messages, since these are needed for in-context examples of function calling
        token_counts = [count_tokens(str(msg)) for msg in self.messages]
        message_buffer_token_count = sum(token_counts[1:])  # no system message
        desired_token_count_to_summarize = int(message_buffer_token_count * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC)
        candidate_messages_to_summarize = self.messages[1:]
        token_counts = token_counts[1:]

        if preserve_last_N_messages:
            candidate_messages_to_summarize = candidate_messages_to_summarize[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]
            token_counts = token_counts[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]

        # if disallow_tool_as_first:
        #     # We have to make sure that a "tool" call is not sitting at the front (after system message),
        #     # otherwise we'll get an error from OpenAI (if using the OpenAI API)
        #     while len(candidate_messages_to_summarize) > 0:
        #         if candidate_messages_to_summarize[0]["role"] in ["tool", "function"]:
        #             candidate_messages_to_summarize.pop(0)
        #         else:
        #             break

        printd(f"MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC={MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC}")
        printd(f"MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}")
        printd(f"token_counts={token_counts}")
        printd(f"message_buffer_token_count={message_buffer_token_count}")
        printd(f"desired_token_count_to_summarize={desired_token_count_to_summarize}")
        printd(f"len(candidate_messages_to_summarize)={len(candidate_messages_to_summarize)}")

        # If at this point there's nothing to summarize, throw an error
        if len(candidate_messages_to_summarize) == 0:
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(self.messages)}, preserve_N={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}]"
            )

        # Walk down the message buffer (front-to-back) until we hit the target token count
        tokens_so_far = 0
        cutoff = 0
        for i, msg in enumerate(candidate_messages_to_summarize):
            cutoff = i
            tokens_so_far += token_counts[i]
            if tokens_so_far > desired_token_count_to_summarize:
                break
        # Account for system message
        cutoff += 1

        # Try to make an assistant message come after the cutoff
        try:
            printd(f"Selected cutoff {cutoff} was a 'user', shifting one...")
            if self.messages[cutoff]["role"] == "user":
                new_cutoff = cutoff + 1
                if self.messages[new_cutoff]["role"] == "user":
                    printd(f"Shifted cutoff {new_cutoff} is still a 'user', ignoring...")
                cutoff = new_cutoff
        except IndexError:
            pass

        # Make sure the cutoff isn't on a 'tool' or 'function'
        if disallow_tool_as_first:
            while self.messages[cutoff]["role"] in ["tool", "function"] and cutoff < len(self.messages):
                printd(f"Selected cutoff {cutoff} was a 'tool', shifting one...")
                cutoff += 1

        message_sequence_to_summarize = self.messages[1:cutoff]  # do NOT get rid of the system message
        if len(message_sequence_to_summarize) <= 1:
            # This prevents a potential infinite loop of summarizing the same message over and over
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(message_sequence_to_summarize)} <= 1]"
            )
        else:
            printd(f"Attempting to summarize {len(message_sequence_to_summarize)} messages [1:{cutoff}] of {len(self.messages)}")

        # We can't do summarize logic properly if context_window is undefined
        if self.agent_state.llm_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            print(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
            print(f"{self.agent_state}")
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
            )
        summary = summarize_messages(agent_state=self.agent_state, message_sequence_to_summarize=message_sequence_to_summarize)
        printd(f"Got summary: {summary}")

        # Metadata that's useful for the agent to see
        all_time_message_count = self.messages_total
        remaining_message_count = len(self.messages[cutoff:])
        hidden_message_count = all_time_message_count - remaining_message_count
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(summary, summary_message_count, hidden_message_count, all_time_message_count)
        printd(f"Packaged into message: {summary_message}")

        prior_len = len(self.messages)
        self._trim_messages(cutoff)
        packed_summary_message = {"role": "user", "content": summary_message}
        self._prepend_to_messages(
            [
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=packed_summary_message,
                )
            ]
        )

        # reset alert
        self.agent_alerted_about_memory_pressure = False

        printd(f"Ran summarizer, messages length {prior_len} -> {len(self.messages)}")

    def summarize_messages_focus_inplace(self, sorted_messages_by_distance: List[dict]):
        """Summarizes messages based on their distance from the conversation centroid.
        
        Removes message pairs (user and subsequent assistant) that are furthest from 
        the centroid until the context window is sufficiently reduced.
        The text of the removed messages is summarized and prepended to the context.
        """
        print("FOCUS MODE: Summarizing messages based on centroid distance.")
        if not self._messages or len(self._messages) <= 1:  # Must have more than system message
            raise LLMError("Focus summarize error: Not enough messages to process for summarization.")

        # Ensure context window is known
        if self.agent_state.llm_config.context_window is None:
            print(f"{CLI_WARNING_PREFIX}Focus summarize: context_window missing in agent config, setting to default.")
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS.get(self.model, LLM_MAX_TOKENS["DEFAULT"])
            )
        context_window = int(self.agent_state.llm_config.context_window)
        target_token_count = int(context_window * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC)

        # Calculate current total tokens from self._messages
        current_total_tokens = sum(count_tokens(str(msg.to_openai_dict())) for msg in self._messages)
        print(f"Focus summarize: Current tokens: {current_total_tokens}, Target tokens after summarization: {target_token_count}")

        tokens_to_free = current_total_tokens - target_token_count
        if tokens_to_free <= 0:
            print(f"Focus summarize: No tokens need to be freed (current: {current_total_tokens}, target: {target_token_count}). Skipping summarization.")
            self.agent_alerted_about_memory_pressure = False  # Reset alert as we are "handling" it or it's not an issue
            return

        print(f"Focus summarize: Need to free approximately {tokens_to_free} tokens.")

        message_ids_to_remove_set = set()
        # Map message IDs to Message objects for quick lookup
        message_id_to_message_obj = {msg.id: msg for msg in self._messages if msg.id is not None}

        # Preserve last N messages: Get their IDs. System message is not part of this.
        ids_to_preserve = set()
        if MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST > 0:
            # Consider only messages after the system prompt for preservation indexing
            relevant_messages_for_preservation = [m for m in self._messages if m.role != "system"]
            if len(relevant_messages_for_preservation) > MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST:
                for i in range(1, MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST + 1):
                    if relevant_messages_for_preservation[-i].id:
                        ids_to_preserve.add(relevant_messages_for_preservation[-i].id)
        print(f"Focus summarize: Preserving IDs of last {len(ids_to_preserve)} non-system messages: {ids_to_preserve}")

        tokens_freed_so_far = 0
        
        # Store (user_msg_id, assistant_msg_id) tuples for messages selected
        selected_pair_ids_for_summarization = [] 

        for pair_info in sorted_messages_by_distance:  # Furthest first
            user_msg_id = pair_info["user_msg_id"]
            assistant_msg_id = pair_info["assistant_msg_id"]

            # Skip if either message in the pair is in the preserve list
            if user_msg_id in ids_to_preserve or assistant_msg_id in ids_to_preserve:
                print(f"Focus summarize: Skipping pair (User: {user_msg_id}, Asst: {assistant_msg_id}) as it's in preserve list.")
                continue
            
            # Skip if already marked for removal
            if user_msg_id in message_ids_to_remove_set or assistant_msg_id in message_ids_to_remove_set:
                continue

            user_msg_obj = message_id_to_message_obj.get(user_msg_id)
            assistant_msg_obj = message_id_to_message_obj.get(assistant_msg_id)

            if not user_msg_obj or not assistant_msg_obj:
                print(f"{CLI_WARNING_PREFIX}Focus summarize: Could not find message objects for pair (User: {user_msg_id}, Asst: {assistant_msg_id}). Skipping.")
                continue
            
            # Skip if either message is a system message
            if user_msg_obj.role == "system" or assistant_msg_obj.role == "system":
                print(f"{CLI_WARNING_PREFIX}Focus summarize: Attempted to select a system message in pair ({user_msg_id}, {assistant_msg_id}). Skipping.")
                continue

            pair_tokens = count_tokens(str(user_msg_obj.to_openai_dict())) + count_tokens(str(assistant_msg_obj.to_openai_dict()))
            
            message_ids_to_remove_set.add(user_msg_id)
            message_ids_to_remove_set.add(assistant_msg_id)
            selected_pair_ids_for_summarization.append({'user':user_msg_id, 'assistant':assistant_msg_id}) # Store for chronological collection
            tokens_freed_so_far += pair_tokens
            
            # print(f"Focus summarize: Tentatively selected pair (User: {user_msg_id}, Asst: {assistant_msg_id}) for summarization. Tokens in pair: {pair_tokens}. Total tokens potentially freed: {tokens_freed_so_far}")

            if tokens_freed_so_far >= tokens_to_free:
                break
        
        if not message_ids_to_remove_set:
            print(f"{CLI_WARNING_PREFIX}Focus summarize: No messages selected for summarization. This might be due to all distant messages being protected by 'preserve_last_N', or too few messages overall.")
            # If we can't free space, we can't proceed. The agent will likely hit the context error again.
            # Raise an error to indicate failure to reduce context via this method.
            raise LLMError("Focus summarize: Failed to select any messages for summarization to reduce context pressure.")

        # Collect the actual message dicts for the summarizer, in CHRONOLOGICAL order
        actual_messages_for_summarizer_chronological = []
        # Iterate through the original self._messages to maintain order
        for msg_obj in self._messages:
            if msg_obj.id in message_ids_to_remove_set:
                actual_messages_for_summarizer_chronological.append(msg_obj.to_openai_dict())
        
        if not actual_messages_for_summarizer_chronological:
             raise LLMError("Focus summarize: Messages were marked for removal, but could not be collected for the summarizer. Internal logic error.")

        print(f"Focus summarize: Summarizing {len(actual_messages_for_summarizer_chronological)} individual messages from ({len(message_ids_to_remove_set)//2} pairs), in chronological order.")
        
        # Use cluster-based summarization if enabled
        if self.cluster_summaries:
            print("Focus summarize: Cluster-based summarization enabled, generating embeddings for clustering...")
            try:
                # Generate message pair embeddings for clustering
                message_pair_embeddings_with_ids = self._create_robust_message_pair_embeddings(
                    message_sequence=self._messages, 
                    embedding_config=self.agent_state.embedding_config
                )
                if not message_pair_embeddings_with_ids:
                    print("Focus summarize: No message pair embeddings generated. Falling back to regular summarization.")
                    raise Exception("No embeddings generated")
                
                cluster_assignments = self.cluster_messages_for_summarization(message_pair_embeddings_with_ids)
                summary_text = self.summarize_messages_by_clusters(message_ids_to_remove_set, cluster_assignments, tokens_to_free)
                summary_mode = "Focus+Cluster"
            except Exception as cluster_e:
                print(f"Focus summarize: Clustering failed ({cluster_e}), falling back to regular summarization")
                summary_text = summarize_messages(
                    agent_state=self.agent_state,
                    message_sequence_to_summarize=actual_messages_for_summarizer_chronological
                )
                summary_mode = "Focus"
        else:
            summary_text = summarize_messages(
                agent_state=self.agent_state,
                message_sequence_to_summarize=actual_messages_for_summarizer_chronological
            )
            summary_mode = "Focus"
        
        print(f"Focus summarize: Got summary: {summary_text}")

        # Package summary message using the standard utility
        # We need to calculate how many messages are "hidden" by this summary.
        # For package_summarize_message:
        # summary_message_count = len(actual_messages_for_summarizer_chronological)
        # hidden_message_count could be more complex; it refers to all messages not in current window + those being summarized now.
        # Let's use a simplified approach for now, or use the structure from summarize_messages_inplace
        
        num_original_messages_summarized = len(actual_messages_for_summarizer_chronological)
        # This is a rough estimate for 'hidden_message_count' for the purpose of the summary string
        # A more accurate count would involve looking at total persisted messages vs current.
        # For now, let's consider 'hidden' as those we just summarized.
        # The existing `package_summarize_message` is tied to FIFO logic's view of history.
        # We will create a slightly different summary string for focus mode.
        summary_insert_content = f"[{summary_mode} Summary: condensed {num_original_messages_summarized} prior messages that were less central to the recent conversation flow]:\n{summary_text}"

        summary_message_to_prepend = Message.dict_to_message(
            agent_id=self.agent_state.id,
            user_id=self.agent_state.user_id,
            model=self.model, 
            openai_message_dict={"role": "user", "content": summary_insert_content} # Role 'user' is common for injected summaries
        )
        
        # Remove the selected messages from self._messages and persistence manager
        original_message_count = len(self._messages)
        new_messages_list = [msg for msg in self._messages if msg.id not in message_ids_to_remove_set]
        
        if isinstance(self.persistence_manager, LocalStateManager):
            deleted_count_pm = 0
            for msg_id_to_delete in message_ids_to_remove_set:
                try:
                    if self.persistence_manager.recall_memory.storage.get(id=msg_id_to_delete):
                        self.persistence_manager.recall_memory.storage.delete(filters={"id": msg_id_to_delete})
                        deleted_count_pm +=1
                except Exception as del_e:
                    print(f"{CLI_WARNING_PREFIX}Focus summarize: Error deleting message {msg_id_to_delete} from persistence: {del_e}")
            print(f"Focus summarize: Deleted {deleted_count_pm}/{len(message_ids_to_remove_set)} messages from persistence manager.")
        else:
            print(f"{CLI_WARNING_PREFIX}Focus summarize: Persistence manager is not LocalStateManager or compatible; selective deletion from PM might not occur. Message list in AgentState will be updated.")

        self._messages = new_messages_list
        
        # Prepend the summary message (this also handles persistence of the new summary message)
        self._prepend_to_messages([summary_message_to_prepend])
        
        self.agent_alerted_about_memory_pressure = False
        
        final_token_count = sum(count_tokens(str(msg.to_openai_dict())) for msg in self._messages)
        print(f"Focus summarize: Completed. Original messages: {original_message_count}, New messages: {len(self._messages)}. Original tokens: {current_total_tokens}, Approx Freed: {tokens_freed_so_far}, Final tokens: {final_token_count}.")

    def summarize_messages_hybrid_inplace(self):
        """Hybrid summarization that combines FIFO and focus approaches using beta weighting.
        
        Beta = 1.0: Pure focus mode (distance-based)
        Beta = 0.0: Pure FIFO mode (chronological)
        Beta = 0.5: Equal weighting of both approaches
        """
        print("HYBRID MODE: Summarizing messages using hybrid approach (FIFO + Focus).")
        if not self._messages or len(self._messages) <= 1:  # Must have more than system message
            raise LLMError("Hybrid summarize error: Not enough messages to process for summarization.")

        # Ensure context window is known
        if self.agent_state.llm_config.context_window is None:
            print(f"{CLI_WARNING_PREFIX}Hybrid summarize: context_window missing in agent config, setting to default.")
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS.get(self.model, LLM_MAX_TOKENS["DEFAULT"])
            )
        context_window = int(self.agent_state.llm_config.context_window)
        target_token_count = int(context_window * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC)

        # Calculate current total tokens from self._messages
        current_total_tokens = sum(count_tokens(str(msg.to_openai_dict())) for msg in self._messages)
        print(f"Hybrid summarize: Current tokens: {current_total_tokens}, Target tokens after summarization: {target_token_count}")

        tokens_to_free = current_total_tokens - target_token_count
        if tokens_to_free <= 0:
            print(f"Hybrid summarize: No tokens need to be freed (current: {current_total_tokens}, target: {target_token_count}). Skipping summarization.")
            self.agent_alerted_about_memory_pressure = False
            return

        print(f"Hybrid summarize: Need to free approximately {tokens_to_free} tokens. Beta={self.beta}")

        # Step 1: Generate message pair embeddings for focus approach
        print("Hybrid summarize: Generating embeddings for focus component...")
        try:
            message_pair_embeddings_with_ids = self._create_robust_message_pair_embeddings(
                message_sequence=self._messages, 
                embedding_config=self.agent_state.embedding_config
            )
            if not message_pair_embeddings_with_ids:
                print("Hybrid summarize: No message pair embeddings generated. Falling back to pure FIFO mode.")
                self.summarize_messages_inplace()
                return
        except Exception as emb_e:
            print(f"Hybrid summarize: Error generating embeddings: {emb_e}. Falling back to pure FIFO mode.")
            self.summarize_messages_inplace()
            return

        # Extract just the embedding vectors for centroid and distance calculation
        actual_embedding_vectors = [pair[2] for pair in message_pair_embeddings_with_ids]

        # Calculate centroid
        centroid_vec = calculate_centroid(actual_embedding_vectors)
        if centroid_vec is None:
            print("Hybrid summarize: Centroid calculation failed. Falling back to pure FIFO mode.")
            self.summarize_messages_inplace()
            return

        # Convert list of list to list of np.array for distance calculation
        np_embedding_vectors = [np.array(vec) for vec in actual_embedding_vectors]
        distances = calculate_cosine_distances(centroid_vec, np_embedding_vectors)

        # Step 2: Create focus scores (distance-based, for pairs)
        pair_focus_scores = {}
        max_distance = max(distances) if distances else 1.0
        for i, pair_data in enumerate(message_pair_embeddings_with_ids):
            user_msg_id = pair_data[0]
            assistant_msg_id = pair_data[1]
            # Normalize distance (higher distance = higher score for removal)
            normalized_distance = distances[i] / max_distance if max_distance > 0 else 0.0
            pair_focus_scores[(user_msg_id, assistant_msg_id)] = normalized_distance

        # Step 3: Create FIFO scores (chronological order, for pairs)
        pair_fifo_scores = {}
        total_pairs = len(message_pair_embeddings_with_ids)
        
        # Create FIFO scores where older pairs get higher scores (more likely to be removed)
        for i, pair_data in enumerate(message_pair_embeddings_with_ids):
            user_msg_id = pair_data[0]
            assistant_msg_id = pair_data[1]
            # Higher scores for older pairs (earlier in history)
            fifo_score = (total_pairs - i) / total_pairs if total_pairs > 0 else 0.0
            pair_fifo_scores[(user_msg_id, assistant_msg_id)] = max(0.0, fifo_score)

        # Step 4: Combine scores using beta weighting
        pair_hybrid_scores = {}
        
        for pair_key in pair_focus_scores.keys():
            focus_component = pair_focus_scores.get(pair_key, 0.0) * self.beta
            fifo_component = pair_fifo_scores.get(pair_key, 0.0) * (1.0 - self.beta)
            pair_hybrid_scores[pair_key] = focus_component + fifo_component

        print(f"Hybrid summarize: Generated {len(pair_hybrid_scores)} hybrid pair scores (beta={self.beta})")

        # Step 5: Select message pairs for removal based on hybrid scores (PAIR-BASED SELECTION)
        message_id_to_message_obj = {msg.id: msg for msg in self._messages if msg.id is not None}

        # Preserve last N messages (operate on pairs)
        ids_to_preserve = set()
        if MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST > 0:
            relevant_messages_for_preservation = [m for m in self._messages if m.role != "system"]
            if len(relevant_messages_for_preservation) > MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST:
                for i in range(1, MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST + 1):
                    if relevant_messages_for_preservation[-i].id:
                        ids_to_preserve.add(relevant_messages_for_preservation[-i].id)

        # Sort pairs by hybrid score (highest first = most likely to be removed)
        sorted_pairs_by_score = sorted(pair_hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        message_ids_to_remove_set = set()
        tokens_freed_so_far = 0

        # Use PAIR-BASED selection (same as focus mode)
        for (user_msg_id, assistant_msg_id), hybrid_score in sorted_pairs_by_score:
            # Skip if either message in the pair is in the preserve list
            if user_msg_id in ids_to_preserve or assistant_msg_id in ids_to_preserve:
                print(f"Hybrid summarize: Skipping pair (User: {user_msg_id}, Asst: {assistant_msg_id}) as it's in preserve list.")
                continue
            
            # Skip if already marked for removal
            if user_msg_id in message_ids_to_remove_set or assistant_msg_id in message_ids_to_remove_set:
                continue

            user_msg_obj = message_id_to_message_obj.get(user_msg_id)
            assistant_msg_obj = message_id_to_message_obj.get(assistant_msg_id)

            if not user_msg_obj or not assistant_msg_obj:
                print(f"{CLI_WARNING_PREFIX}Hybrid summarize: Could not find message objects for pair (User: {user_msg_id}, Asst: {assistant_msg_id}). Skipping.")
                continue
            
            # Skip if either message is a system message
            if user_msg_obj.role == "system" or assistant_msg_obj.role == "system":
                print(f"{CLI_WARNING_PREFIX}Hybrid summarize: Attempted to select a system message in pair ({user_msg_id}, {assistant_msg_id}). Skipping.")
                continue

            # Calculate tokens for the entire pair
            pair_tokens = count_tokens(str(user_msg_obj.to_openai_dict())) + count_tokens(str(assistant_msg_obj.to_openai_dict()))
            
            # Add BOTH messages from the pair (maintain conversation integrity)
            message_ids_to_remove_set.add(user_msg_id)
            message_ids_to_remove_set.add(assistant_msg_id)
            tokens_freed_so_far += pair_tokens

            # print(f"Hybrid summarize: Selected pair (User: {user_msg_id}, Asst: {assistant_msg_id}) for removal (hybrid_score: {hybrid_score:.4f}, tokens: {pair_tokens})")

            if tokens_freed_so_far >= tokens_to_free:
                break

        if not message_ids_to_remove_set:
            print(f"{CLI_WARNING_PREFIX}Hybrid summarize: No message pairs selected for summarization.")
            raise LLMError("Hybrid summarize: Failed to select any message pairs for summarization to reduce context pressure.")

        # Step 6: Collect messages for summarization in chronological order
        actual_messages_for_summarizer_chronological = []
        for msg_obj in self._messages:
            if msg_obj.id in message_ids_to_remove_set:
                actual_messages_for_summarizer_chronological.append(msg_obj.to_openai_dict())

        if not actual_messages_for_summarizer_chronological:
            raise LLMError("Hybrid summarize: Messages were marked for removal, but could not be collected for the summarizer.")

        # Step 7: Generate summary
        print(f"Hybrid summarize: Summarizing {len(actual_messages_for_summarizer_chronological)} messages from {len(message_ids_to_remove_set)//2} pairs.")
        
        # Use cluster-based summarization if enabled
        if self.cluster_summaries:
            print("Hybrid summarize: Cluster-based summarization enabled, performing clustering...")
            try:
                cluster_assignments = self.cluster_messages_for_summarization(message_pair_embeddings_with_ids)
                summary_text = self.summarize_messages_by_clusters(message_ids_to_remove_set, cluster_assignments, tokens_to_free)
                summary_mode = f"Hybrid+Cluster (β={self.beta})"
            except Exception as cluster_e:
                print(f"Hybrid summarize: Clustering failed ({cluster_e}), falling back to regular summarization")
                summary_text = summarize_messages(
                    agent_state=self.agent_state,
                    message_sequence_to_summarize=actual_messages_for_summarizer_chronological
                )
                summary_mode = f"Hybrid (β={self.beta})"
        else:
            summary_text = summarize_messages(
                agent_state=self.agent_state,
                message_sequence_to_summarize=actual_messages_for_summarizer_chronological
            )
            summary_mode = f"Hybrid (β={self.beta})"
        
        print(f"Hybrid summary: {summary_text}")

        # Step 8: Create summary message
        num_original_messages_summarized = len(actual_messages_for_summarizer_chronological)
        summary_insert_content = f"[{summary_mode} Summary: condensed {num_original_messages_summarized} messages using combined FIFO and focus-based pair selection]:\n{summary_text}"

        summary_message_to_prepend = Message.dict_to_message(
            agent_id=self.agent_state.id,
            user_id=self.agent_state.user_id,
            model=self.model, 
            openai_message_dict={"role": "user", "content": summary_insert_content}
        )

        # Step 9: Remove selected messages and add summary
        original_message_count = len(self._messages)
        new_messages_list = [msg for msg in self._messages if msg.id not in message_ids_to_remove_set]

        # Clean up persistence manager
        if isinstance(self.persistence_manager, LocalStateManager):
            deleted_count_pm = 0
            for msg_id_to_delete in message_ids_to_remove_set:
                try:
                    if self.persistence_manager.recall_memory.storage.get(id=msg_id_to_delete):
                        self.persistence_manager.recall_memory.storage.delete(filters={"id": msg_id_to_delete})
                        deleted_count_pm += 1
                except Exception as del_e:
                    print(f"{CLI_WARNING_PREFIX}Hybrid summarize: Error deleting message {msg_id_to_delete} from persistence: {del_e}")
            print(f"Hybrid summarize: Deleted {deleted_count_pm}/{len(message_ids_to_remove_set)} messages from persistence manager.")

        self._messages = new_messages_list
        self._prepend_to_messages([summary_message_to_prepend])
        self.agent_alerted_about_memory_pressure = False

        final_token_count = sum(count_tokens(str(msg.to_openai_dict())) for msg in self._messages)
        print(f"Hybrid summarize: Completed. Original messages: {original_message_count}, New messages: {len(self._messages)}. Original tokens: {current_total_tokens}, Approx Freed: {tokens_freed_so_far}, Final tokens: {final_token_count}.")

    def cluster_messages_for_summarization(self, message_embeddings_with_ids: List[Tuple], min_clusters: int = 2, max_clusters: int = 6) -> dict:
        """Cluster messages based on their embeddings to group similar topics together.
        
        Args:
            message_embeddings_with_ids: List of tuples (user_msg_id, assistant_msg_id, embedding_vector)
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Dictionary mapping message_id -> cluster_id
        """
        if len(message_embeddings_with_ids) < min_clusters:
            print(f"Clustering: Not enough message pairs ({len(message_embeddings_with_ids)}) for clustering. Using single cluster.")
            # Return all messages in cluster 0
            cluster_assignments = {}
            for user_id, assistant_id, _ in message_embeddings_with_ids:
                cluster_assignments[user_id] = 0
                cluster_assignments[assistant_id] = 0
            return cluster_assignments

        # Extract embeddings
        embeddings = np.array([pair[2] for pair in message_embeddings_with_ids])
        
        # Determine optimal number of clusters using silhouette analysis
        best_score = -1
        best_k = min_clusters
        max_k = min(max_clusters, len(message_embeddings_with_ids))
        
        print(f"Clustering: Testing {min_clusters} to {max_k} clusters for {len(message_embeddings_with_ids)} message pairs...")
        
        for k in range(min_clusters, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                    score = silhouette_score(embeddings, cluster_labels)
                    print(f"Clustering: k={k}, silhouette_score={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                else:
                    print(f"Clustering: k={k} resulted in only 1 cluster, skipping")
                    
            except Exception as e:
                print(f"Clustering: Error with k={k}: {e}")
                continue
        
        # Perform final clustering with best k
        print(f"Clustering: Using k={best_k} clusters (silhouette_score={best_score:.4f})")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_labels = kmeans.fit_predict(embeddings)
        
        # Map message IDs to cluster assignments
        cluster_assignments = {}
        for i, (user_id, assistant_id, _) in enumerate(message_embeddings_with_ids):
            cluster_id = int(final_labels[i])
            cluster_assignments[user_id] = cluster_id
            cluster_assignments[assistant_id] = cluster_id
        
        # Print cluster summary
        cluster_counts = {}
        for cluster_id in cluster_assignments.values():
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        print(f"Clustering: Cluster distribution: {cluster_counts}")
        
        return cluster_assignments

    def summarize_messages_by_clusters(self, message_ids_to_remove_set: set, cluster_assignments: dict, tokens_to_free: int) -> str:
        """Summarize messages grouped by clusters to avoid diluted topic mixing.
        
        Args:
            message_ids_to_remove_set: Set of message IDs to be summarized
            cluster_assignments: Dictionary mapping message_id -> cluster_id
            tokens_to_free: Total number of tokens that need to be freed
            
        Returns:
            Combined summary text from all clusters
        """
        print("Cluster-based summarization: Grouping messages by cluster...")
        
        # Group messages by cluster
        message_id_to_message_obj = {msg.id: msg for msg in self._messages if msg.id is not None}
        clusters = {}
        cluster_tokens = {}
        
        for msg_id in message_ids_to_remove_set:
            if msg_id not in cluster_assignments:
                continue  # Skip messages without cluster assignment
                
            cluster_id = cluster_assignments[msg_id]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
                cluster_tokens[cluster_id] = 0
            
            msg_obj = message_id_to_message_obj.get(msg_id)
            if msg_obj:
                clusters[cluster_id].append(msg_obj)
                cluster_tokens[cluster_id] += count_tokens(str(msg_obj.to_openai_dict()))
        
        if not clusters:
            print("Cluster-based summarization: No clusters found, falling back to regular summarization")
            # Fallback to regular summarization
            actual_messages = []
            for msg_obj in self._messages:
                if msg_obj.id in message_ids_to_remove_set:
                    actual_messages.append(msg_obj.to_openai_dict())
            return summarize_messages(
                agent_state=self.agent_state,
                message_sequence_to_summarize=actual_messages
            )
        
        print(f"Cluster-based summarization: Found {len(clusters)} clusters")
        for cluster_id, msgs in clusters.items():
            print(f"  Cluster {cluster_id}: {len(msgs)} messages, {cluster_tokens[cluster_id]} tokens")
        
        # Generate summary for each cluster
        cluster_summaries = []
        total_tokens_in_clusters = sum(cluster_tokens.values())
        
        for cluster_id in sorted(clusters.keys()):
            cluster_messages = clusters[cluster_id]
            if not cluster_messages:
                continue
                
            # Convert to chronological order
            cluster_messages_sorted = sorted(cluster_messages, key=lambda x: self._messages.index(x))
            cluster_messages_dicts = [msg.to_openai_dict() for msg in cluster_messages_sorted]
            
            print(f"Cluster-based summarization: Summarizing cluster {cluster_id} with {len(cluster_messages_dicts)} messages...")
            
            try:
                cluster_summary = summarize_messages(
                    agent_state=self.agent_state,
                    message_sequence_to_summarize=cluster_messages_dicts
                )
                cluster_token_percentage = (cluster_tokens[cluster_id] / total_tokens_in_clusters) * 100
                cluster_summaries.append(f"Topic Cluster {cluster_id} ({cluster_token_percentage:.1f}% of summarized content): {cluster_summary}")
                
            except Exception as e:
                print(f"Cluster-based summarization: Error summarizing cluster {cluster_id}: {e}")
                # Fallback: create a simple summary
                cluster_summaries.append(f"Topic Cluster {cluster_id}: Contains {len(cluster_messages_dicts)} messages [summarization failed: {str(e)}]")
        
        # Combine all cluster summaries
        if len(cluster_summaries) == 1:
            combined_summary = cluster_summaries[0]
        else:
            combined_summary = "Multi-topic summary:\n" + "\n\n".join(cluster_summaries)
        
        print(f"Cluster-based summarization: Generated combined summary with {len(cluster_summaries)} topic clusters")
        return combined_summary

    def heartbeat_is_paused(self):
        """Check if there's a requested pause on timed heartbeats"""

        # Check if the pause has been initiated
        if self.pause_heartbeats_start is None:
            return False

        # Check if it's been more than pause_heartbeats_minutes since pause_heartbeats_start
        elapsed_time = get_utc_time() - self.pause_heartbeats_start
        return elapsed_time.total_seconds() < self.pause_heartbeats_minutes * 60

    def rebuild_memory(self):
        """Rebuilds the system message with the latest memory object"""
        curr_system_message = self.messages[0]  # this is the system + memory bank, not just the system prompt
        new_system_message = initialize_message_sequence(
            self.model,
            self.system,
            self.memory,
            archival_memory=self.persistence_manager.archival_memory,
            recall_memory=self.persistence_manager.recall_memory,
        )[0]

        diff = united_diff(curr_system_message["content"], new_system_message["content"])
        printd(f"Rebuilding system with new memory...\nDiff:\n{diff}")

        # Swap the system message out
        self._swap_system_message(
            Message.dict_to_message(
                agent_id=self.agent_state.id, user_id=self.agent_state.user_id, model=self.model, openai_message_dict=new_system_message
            )
        )

    def add_function(self, function_name: str) -> str:
        if function_name in self.functions_python.keys():
            msg = f"Function {function_name} already loaded"
            printd(msg)
            return msg

        available_functions = load_all_function_sets()
        if function_name not in available_functions.keys():
            raise ValueError(f"Function {function_name} not found in function library")

        self.functions.append(available_functions[function_name]["json_schema"])
        self.functions_python[function_name] = available_functions[function_name]["python_function"]

        msg = f"Added function {function_name}"
        # self.save()
        self.update_state()
        printd(msg)
        return msg

    def remove_function(self, function_name: str) -> str:
        if function_name not in self.functions_python.keys():
            msg = f"Function {function_name} not loaded, ignoring"
            printd(msg)
            return msg

        # only allow removal of user defined functions
        user_func_path = Path(USER_FUNCTIONS_DIR)
        func_path = Path(inspect.getfile(self.functions_python[function_name]))
        is_subpath = func_path.resolve().parts[: len(user_func_path.resolve().parts)] == user_func_path.resolve().parts

        if not is_subpath:
            raise ValueError(f"Function {function_name} is not user defined and cannot be removed")

        self.functions = [f_schema for f_schema in self.functions if f_schema["name"] != function_name]
        self.functions_python.pop(function_name)

        msg = f"Removed function {function_name}"
        # self.save()
        self.update_state()
        printd(msg)
        return msg

    def update_state(self) -> AgentState:
        updated_state = {
            "persona": self.memory.persona,
            "human": self.memory.human,
            "system": self.system,
            "functions": self.functions,
            "messages": [str(msg.id) for msg in self._messages],
            "mem_mode": self.mem_mode,  # Persist mem_mode
            "beta": self.beta,  # Persist beta
            "cluster_summaries": self.cluster_summaries,  # Persist cluster_summaries
            "prompt_type": self.prompt_type,  # Persist prompt_type
        }

        self.agent_state = AgentState(
            name=self.agent_state.name,
            user_id=self.agent_state.user_id,
            persona=self.agent_state.persona,
            human=self.agent_state.human,
            llm_config=self.agent_state.llm_config,
            embedding_config=self.agent_state.embedding_config,
            preset=self.agent_state.preset,
            id=self.agent_state.id,
            created_at=self.agent_state.created_at,
            state=updated_state,
        )
        return self.agent_state

    def migrate_embedding(self, embedding_config: EmbeddingConfig):
        """Migrate the agent to a new embedding"""
        # TODO: archival memory

        # TODO: recall memory
        raise NotImplementedError()

    def attach_source(self, source_name, source_connector: StorageConnector, ms: MetadataStore):
        """Attach data with name `source_name` to the agent from source_connector."""
        # TODO: eventually, adding a data source should just give access to the retriever the source table, rather than modifying archival memory

        filters = {"user_id": self.agent_state.user_id, "data_source": source_name}
        size = source_connector.size(filters)
        # typer.secho(f"Ingesting {size} passages into {agent.name}", fg=typer.colors.GREEN)
        page_size = 100
        generator = source_connector.get_all_paginated(filters=filters, page_size=page_size)  # yields List[Passage]
        all_passages = []
        for i in tqdm(range(0, size, page_size)):
            passages = next(generator)

            # need to associated passage with agent (for filtering)
            for passage in passages:
                assert isinstance(passage, Passage), f"Generate yielded bad non-Passage type: {type(passage)}"
                passage.agent_id = self.agent_state.id

                # regenerate passage ID (avoid duplicates)
                passage.id = create_uuid_from_string(f"{source_name}_{str(passage.agent_id)}_{passage.text}")

            # insert into agent archival memory
            self.persistence_manager.archival_memory.storage.insert_many(passages)
            all_passages += passages

        assert size == len(all_passages), f"Expected {size} passages, but only got {len(all_passages)}"

        # save destination storage
        self.persistence_manager.archival_memory.storage.save()

        # attach to agent
        source = ms.get_source(source_name=source_name, user_id=self.agent_state.user_id)
        assert source is not None, f"source does not exist for source_name={source_name}, user_id={self.agent_state.user_id}"
        source_id = source.id
        ms.attach_source(agent_id=self.agent_state.id, source_id=source_id, user_id=self.agent_state.user_id)

        total_agent_passages = self.persistence_manager.archival_memory.storage.size()

        printd(
            f"Attached data source {source_name} to agent {self.agent_state.name}, consisting of {len(all_passages)}. Agent now has {total_agent_passages} embeddings in archival memory.",
        )

    def _create_robust_message_pair_embeddings(self, message_sequence, embedding_config):
        """
        Robust embedding function that works with both JSON and plain text messages.
        Creates vector embeddings for message pairs (user message and subsequent assistant message).
        """
        import json
        
        pair_embeddings = []
        if not message_sequence or len(message_sequence) < 2:
            return pair_embeddings

        print(f"Creating robust embeddings for {len(message_sequence)} messages...")
        
        for i in range(len(message_sequence) - 1):
            msg1 = message_sequence[i]
            msg2 = message_sequence[i+1]

            # Look for user-assistant pairs
            if msg1.role == "user" and msg2.role == "assistant":
                try:
                    # Handle both JSON and plain text messages
                    text1 = ""
                    text2 = ""
                    
                    # Try JSON parsing first (for regular MemGPT messages)
                    try:
                        msg1_content = json.loads(msg1.text) if msg1.text else {}
                        if msg1_content.get('type') == 'user_message':
                            text1 = msg1_content.get('message', '') or msg1.text
                        else:
                            text1 = msg1.text if msg1.text else ""
                    except (json.JSONDecodeError, TypeError):
                        # Fall back to plain text (for longmemeval injected messages)
                        text1 = msg1.text if msg1.text else ""
                    
                    # Assistant messages are typically plain text
                    text2 = msg2.text if msg2.text else ""
                    
                    # Skip empty messages or system-like messages
                    if not text1.strip() or not text2.strip():
                        continue
                    
                    # Skip login messages and other system messages
                    if any(keyword in text1.lower() for keyword in ['login', 'bootup', 'system']):
                        continue
                    
                    # Ensure IDs are not None
                    if msg1.id is None or msg2.id is None:
                        print(f"Warning: Skipping message pair due to missing ID(s). User msg ID: {msg1.id}, Assistant msg ID: {msg2.id}")
                        continue

                    # Combine the messages
                    combined_text = text1.strip() + " " + text2.strip()

                    # Skip if combined text is too short
                    if len(combined_text.strip()) < 20:
                        continue

                    try:
                        # Create embedding for the combined text
                        embedding_vector = create_embedding(
                            text=combined_text,
                            embedding_config=embedding_config,
                        )
                        pair_embeddings.append((msg1.id, msg2.id, embedding_vector))
                            
                    except Exception as e:
                        print(f"Error creating embedding for pair (user_msg_id={msg1.id}, assistant_msg_id={msg2.id}): {e}")
                        continue
                        
                except Exception as e:
                    print(f"Error processing message pair at index {i}: {e}")
                    continue

        print(f"Successfully created {len(pair_embeddings)} message pair embeddings")
        return pair_embeddings


def save_agent(agent: Agent, ms: MetadataStore):
    """Save agent to metadata store"""

    agent.update_state()
    agent_state = agent.agent_state

    if ms.get_agent(agent_id=agent_state.id):
        ms.update_agent(agent_state)
    else:
        ms.create_agent(agent_state)
