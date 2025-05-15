import datetime
import uuid
import inspect
import json
from pathlib import Path
import traceback
from typing import List, Tuple, Optional, cast, Union
from tqdm import tqdm
import numpy as np
import os
import datetime

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
        # extras
        messages_total: Optional[int] = None,  # TODO remove?
        first_message_verify_mono: bool = True,  # TODO move to config?
        num_recent_pairs_to_protect: int = 1, # New parameter
        num_initial_pairs_to_protect: int = 1, # New parameter for initial pairs
        allow_llm_archival_memory_insert: bool = True, # New flag to control LLM's archival_memory_insert
        log_step_embeddings: bool = False, # Flag to log all archival embeddings after each step
        embeddings_log_dir: Optional[str] = None # Directory for step embedding logs
    ):
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
                },
            )

        # An agent can also be created directly from AgentState
        elif agent_state is not None:
            assert preset is None, "Can create an agent from a Preset or AgentState (but both were provided)"
            assert agent_state.state is not None and agent_state.state != {}, "AgentState.state cannot be empty"

            # Assume the agent_state passed in is formatted correctly
            init_agent_state = agent_state

        else:
            raise ValueError("Both Preset and AgentState were null (must provide one or the other)")

        # Hold a copy of the state that was used to init the agent
        self.agent_state = init_agent_state

        # gpt-4, gpt-3.5-turbo, ...
        self.model = self.agent_state.llm_config.model

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

        # Initialize centroid history
        self.centroid_history = []
        self.centroid_timestamps = []

        # Ensure numpy is available if it wasn't auto-imported by something else
        assert np, "NumPy import failed"

        self.num_recent_pairs_to_protect = num_recent_pairs_to_protect # Store the new parameter
        self.num_initial_pairs_to_protect = num_initial_pairs_to_protect # Store the new initial pair protection parameter
        self.allow_llm_archival_memory_insert = allow_llm_archival_memory_insert
        # Determine and set up logging parameters
        self.log_step_embeddings = True # Force True for your current testing needs
        
        if embeddings_log_dir is None:
            if self.agent_state and hasattr(self.agent_state, 'id') and self.agent_state.id:
                self.embeddings_log_dir = Path(".") / "agent_step_embedding_logs" / str(self.agent_state.id)
                print(f"AGENT_INIT_DEBUG: embeddings_log_dir not specified, defaulting to local: {self.embeddings_log_dir}")
            else:
                self.embeddings_log_dir = Path(".") / "agent_step_embedding_logs" / "unknown_agent"
                print(f"AGENT_INIT_WARNING: Agent ID not available for default embeddings_log_dir, using 'unknown_agent'. Check agent_state initialization.")
        else:
            self.embeddings_log_dir = Path(embeddings_log_dir)
        
        self.step_count = 0 # For unique naming of step embedding logs

        # Conditionally remove archival_memory_insert if the flag is False
        if not self.allow_llm_archival_memory_insert:
            if "archival_memory_insert" in self.functions_python:
                self.functions = [f for f in self.functions if f.get("name") != "archival_memory_insert"]
                del self.functions_python["archival_memory_insert"]
                printd("LLM direct archival_memory_insert is DISABLED by agent config.")
            else:
                # This case might occur if the function was never loaded or already removed, which is fine.
                printd("LLM direct archival_memory_insert is DISABLED, but was not found in current functions list.")
        else:
            printd("LLM direct archival_memory_insert is ENABLED by agent config.")

    def _log_message_roles(self, message_list: Union[List[dict], List[Message]], context_label: str = "Current"):
        """Helper function to log the sequence of message roles."""
        roles = []
        for msg in message_list:
            if isinstance(msg, dict):
                roles.append(msg.get("role"))
            elif hasattr(msg, 'role'): # For Message objects
                roles.append(msg.role)
            else:
                roles.append("unknown_type")
        printd(f"MESSAGE ROLES ({context_label}, count: {len(roles)}): {roles}")

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

        self._log_message_roles(message_sequence, context_label="Before API Call") # Logging message roles

        # Log context window usage
        try:
            try:
                # Try with the specific model name first
                prompt_tokens = count_tokens(json.dumps(message_sequence), self.model)
            except Exception as e:
                # If it fails (e.g., unknown model), fallback to a default tokenizer
                # print(f"Warning: count_tokens failed for model {self.model} ({e}), falling back to cl100k_base tokenizer.")
                prompt_tokens = count_tokens(json.dumps(message_sequence), "cl100k_base")
                
            model_context_window = LLM_MAX_TOKENS.get(self.model) or LLM_MAX_TOKENS["DEFAULT"]
            print(f"CONTEXT: Using {prompt_tokens}/{model_context_window} tokens for model {self.model}")

            # Detailed token breakdown
            print("DETAILED CONTEXT TOKEN BREAKDOWN:")
            role_token_counts = {} 
            individual_message_details = []
            tokenizer_model_for_parts = self.model
            tokenizer_endpoint_type_for_parts = self.agent_state.llm_config.model_endpoint_type
            tokenizer_failed_for_parts = False

            for i, msg_dict in enumerate(message_sequence):
                role = msg_dict.get("role", "unknown")
                current_msg_parts_tokens = 0
                msg_display_parts = []

                # Helper to count tokens for a part, with fallback
                def count_part_tokens(text_to_count: str) -> int:
                    nonlocal tokenizer_failed_for_parts, tokenizer_model_for_parts, tokenizer_endpoint_type_for_parts
                    try:
                        if tokenizer_failed_for_parts: # If already failed, use fallback for all subsequent parts
                            return count_tokens(text_to_count, "cl100k_base")
                        return count_tokens(text_to_count, tokenizer_model_for_parts, tokenizer_endpoint_type_for_parts)
                    except Exception as count_exc:
                        if not tokenizer_failed_for_parts: # First time failing for parts
                            print(f"Warning: count_tokens for message parts failed for model {tokenizer_model_for_parts} (endpoint type: {tokenizer_endpoint_type_for_parts}). Error: {count_exc}. Falling back to cl100k_base for this and subsequent parts.")
                            tokenizer_failed_for_parts = True
                        # Use cl100k_base as fallback
                        return count_tokens(text_to_count, "cl100k_base")

                # 1. Content
                if "content" in msg_dict and msg_dict["content"] is not None:
                    content_str = str(msg_dict["content"]) # Ensure string
                    tokens = count_part_tokens(content_str)
                    current_msg_parts_tokens += tokens
                    snippet = content_str.replace("\\n", " ")[:60]
                    if len(content_str) > 60: snippet += "..."
                    msg_display_parts.append(f"content: \"{snippet}\" ({tokens} t)")
                elif "content" in msg_dict and msg_dict["content"] is None: # Explicit None content
                     msg_display_parts.append(f"content: null (0 t)")

                # 2. Tool Calls
                if "tool_calls" in msg_dict and msg_dict["tool_calls"]:
                    tool_calls_data = msg_dict["tool_calls"]
                    tool_calls_str_for_tokens = json.dumps(tool_calls_data, ensure_ascii=JSON_ENSURE_ASCII)
                    tokens = count_part_tokens(tool_calls_str_for_tokens)
                    current_msg_parts_tokens += tokens
                    func_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls_data]
                    msg_display_parts.append(f"tool_calls: {func_names} ({tokens} t)")

                # 3. Function Call (older, deprecated)
                if "function_call" in msg_dict and msg_dict["function_call"]:
                    function_call_data = msg_dict["function_call"]
                    function_call_str_for_tokens = json.dumps(function_call_data, ensure_ascii=JSON_ENSURE_ASCII)
                    tokens = count_part_tokens(function_call_str_for_tokens)
                    current_msg_parts_tokens += tokens
                    func_name = function_call_data.get("name", "?")
                    msg_display_parts.append(f"function_call: {func_name} ({tokens} t)")
                
                # 4. Name (for user/assistant/tool messages if present)
                if "name" in msg_dict and msg_dict["name"] is not None:
                    name_str = str(msg_dict["name"])
                    tokens = count_part_tokens(name_str)
                    current_msg_parts_tokens += tokens
                    msg_display_parts.append(f"name: \"{name_str}\" ({tokens} t)")

                role_token_counts[role] = role_token_counts.get(role, 0) + current_msg_parts_tokens
                individual_message_details.append(
                    f"  Msg {i:<3} Role: {role:<10} | Parts Sum: {current_msg_parts_tokens:<5} t | Details: [{'; '.join(msg_display_parts)}]"
                )

            print("  Role-based token sums (from message parts):")
            for role_name, total_role_tokens in sorted(role_token_counts.items()):
                print(f"    {role_name:<10}: {total_role_tokens} t")
            
            details_tokenizer_name = tokenizer_model_for_parts if not tokenizer_failed_for_parts else 'cl100k_base (fallback)'
            # print(f"  Individual message details (parts tokenized using {details_tokenizer_name}):")
            # for detail_line in individual_message_details:
            #     print(detail_line)
            print("-" * 40) # Separator after detailed breakdown

        except Exception as e:
            # Catch any other unexpected errors during logging
            print(f"Warning: Could not calculate and log token count - {e}")


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

            # If the successful function was archival_memory_insert, log embeddings
            if function_name == "archival_memory_insert" and not function_failed: # function_failed should be false here anyway
                self._trigger_archival_embeddings_log(log_reason="llm_archival_insert")

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

    def create_message_pair_embeddings(
        self, 
        user_message: Message, 
        ai_messages: List[Message],
        context_mode: str = "pair"  # Options: "single", "pair", "window"
    ) -> List[str]:
        """Create embeddings for messages based on context mode
        
        Args:
            user_message: The user message
            ai_messages: List of AI response messages
            context_mode: How to create embeddings
                - "single": Create embeddings for each message individually
                - "pair": Create embeddings for consecutive user-AI message pairs
                - "window": Create embeddings with more context (multiple messages)
                
        Returns:
            List of combined text strings ready for embedding
        """
        try:
            combined_texts = []
            
            if context_mode == "single":
                # Create individual message embeddings
                combined_texts.append(f"User: {user_message.text}")
                for ai_msg in ai_messages:
                    if ai_msg.role == "assistant":
                        combined_texts.append(f"AI: {ai_msg.text}")
                
            elif context_mode == "pair":
                # Create embeddings for consecutive user-AI message pairs
                if ai_messages and ai_messages[0].role == "assistant":
                    combined_texts.append(f"User: {user_message.text}\nAI: {ai_messages[0].text}")
                
            elif context_mode == "window":
                # Create embeddings with more context (e.g., last 3 messages)
                messages = [user_message] + ai_messages
                window_size = 3  # Can be made configurable
                
                for i in range(len(messages) - window_size + 1):
                    window = messages[i:i + window_size]
                    combined_text = "\n".join([
                        f"{msg.role.capitalize()}: {msg.text}" 
                        for msg in window
                    ])
                    combined_texts.append(combined_text)
            
            else:
                raise ValueError(f"Invalid context_mode: {context_mode}. Must be one of: 'single', 'pair', 'window'")
            
            return combined_texts
            
        except Exception as e:
            printd(f"Error creating message embeddings: {e}")
            raise e

    def _archive_least_similar_context_pair(self):
        # Calculate minimum number of actual pairs needed (initial protected + recent protected + 1 candidate)
        min_pairs_needed_for_operation = self.num_initial_pairs_to_protect + self.num_recent_pairs_to_protect + 1
        # Corresponding number of messages (pairs * 2) + 1 for system message
        min_messages_for_operation = 1 + (2 * min_pairs_needed_for_operation)

        printd(f"Attempting to archive least similar context pair (protecting first {self.num_initial_pairs_to_protect} pair(s) and last {self.num_recent_pairs_to_protect} pair(s))...")

        if len(self._messages) < min_messages_for_operation:
            printd(f"Not enough messages ({len(self._messages)}, need {min_messages_for_operation} for {min_pairs_needed_for_operation} pairs: "
                   f"{self.num_initial_pairs_to_protect} initial, {self.num_recent_pairs_to_protect} recent, 1 candidate). Skipping centroid archival.")
            return False, None

        all_identified_pairs_data = []
        
        i = 1 
        while i < len(self._messages) - 1: # Need at least current_msg and next_msg
            current_msg = self._messages[i]
            next_msg = self._messages[i+1]

            if current_msg.role == "user" and next_msg.role == "assistant":
                is_protected_special_user_message = False
                
                # Robust check for system alert by parsing JSON
                try:
                    msg_json = json.loads(current_msg.text)
                    if isinstance(msg_json, dict) and msg_json.get("type") == "system_alert":
                        is_protected_special_user_message = True
                        printd(f"PROTECTED_USER_MESSAGE (type: system_alert via JSON parse, index: {i}) and its assistant pair (index {i+1}) will not be considered for archival.")
                except json.JSONDecodeError:
                    # Not a JSON message, or malformed JSON - it's not a system_alert of the expected format.
                    pass # Proceed to other checks

                # Check for summarization message (if not already protected as a system_alert)
                if not is_protected_special_user_message and current_msg.text.startswith("Summary of"):
                    is_protected_special_user_message = True
                    printd(f"PROTECTED_USER_MESSAGE (type: summary, index: {i}) and its assistant pair (index {i+1}) will not be considered for archival.")

                if not is_protected_special_user_message:
                    # This is a regular user-assistant pair, add it as a candidate
                    try:
                        # create_message_pair_embeddings expects Message objects, which current_msg and next_msg are.
                        combined_texts = self.create_message_pair_embeddings(current_msg, [next_msg])
                        if not combined_texts: # Should not happen if inputs are valid
                            i += 1 # Could not embed, advance past current_msg
                            continue 
                        combined_text = "\n".join(combined_texts)
                        embedding = self.persistence_manager.embedding_model.get_text_embedding(combined_text)
                        # Save embedding as numpy array with timestamp in filename
                        if self.log_step_embeddings and embedding is not None:
                            
                            
                            # Create directory if it doesn't exist
                            embeddings_dir = self.embeddings_log_dir or "stored_vector_embeddings"
                            os.makedirs(embeddings_dir, exist_ok=True)
                            
                            # Generate filename with timestamp
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"embedding_pair_{i}_{i+1}_{timestamp}.npy"
                            filepath = os.path.join(embeddings_dir, filename)
                            
                            # Save the embedding as numpy array
                            np.save(filepath, np.array(embedding))
                            
                            # Also save the combined text with the same filename but .json extension
                            json_filename = f"embedding_pair_{i}_{i+1}_{timestamp}.json"
                            json_filepath = os.path.join(embeddings_dir, json_filename)
                            with open(json_filepath, 'w', encoding='utf-8') as f:
                                json.dump({"text": combined_text}, f, ensure_ascii=False, indent=2)
                            
                            printd(f"Saved embedding to {filepath} and text to {json_filepath}")
                        # print(f"DEBUG: Embedding for pair (user_idx {i}, assistant_idx {i+1}): {np.array(embedding)}")

                        all_identified_pairs_data.append({
                            "user_msg": current_msg, "assistant_msg": next_msg,
                            "user_idx": i, "assistant_idx": i + 1,
                            "embedding": np.array(embedding)
                        })
                        i += 2 # Successfully processed this pair, move to the message after next_msg
                    except Exception as e:
                        printd(f"Error embedding a pair for centroid consideration (user_idx {i}, assistant_idx {i+1}): {e}")
                        i += 1 # Error, advance past current_msg to be safe
                else:
                    # This was a protected special user message, skip this pair by advancing past it and its assistant partner
                    i += 2 
            else:
                # Not a user-assistant pair, advance past current_msg
                i += 1
        
        if not all_identified_pairs_data:
            printd("No suitable user-assistant pairs found in context for centroid archival after applying protections. Skipping.")
            return False, None

        # Determine the range of pairs eligible for removal from all_identified_pairs_data
        start_idx_candidates = self.num_initial_pairs_to_protect
        # This is the exclusive end index for the slice.
        end_idx_candidates = len(all_identified_pairs_data) - self.num_recent_pairs_to_protect

        if start_idx_candidates >= end_idx_candidates:
            printd(f"Not enough non-protected pairs to select candidates. "
                   f"Total identified pairs: {len(all_identified_pairs_data)}, "
                   f"Initial pairs to protect: {self.num_initial_pairs_to_protect}, "
                   f"Recent pairs to protect: {self.num_recent_pairs_to_protect}. Skipping centroid archival.")
            candidate_pairs_for_removal_data = []
        else:
            candidate_pairs_for_removal_data = all_identified_pairs_data[start_idx_candidates:end_idx_candidates]
            printd(f"Identified {len(candidate_pairs_for_removal_data)} candidate pairs for removal "
                   f"(from pair index {start_idx_candidates} to {end_idx_candidates - 1} of {len(all_identified_pairs_data)} total identified pairs).")

        # Check if there are any candidates left to process
        if not candidate_pairs_for_removal_data:
            printd(f"No candidate pairs available for centroid archival after applying protections. Skipping.")
            return False, None

        candidate_embeddings_list = [pair_data["embedding"] for pair_data in candidate_pairs_for_removal_data]
        
        if not candidate_embeddings_list:
            printd("No embeddings from candidate pairs to calculate centroid. Skipping.")
            return False, None
            
        centroid = np.mean(candidate_embeddings_list, axis=0)
        # self.store_centroid(centroid) # Potentially store centroid if a removal happens based on it was successful

        distances_and_pairs = []
        for pair_data_item in candidate_pairs_for_removal_data:
            distance = np.linalg.norm(pair_data_item["embedding"] - centroid)
            distances_and_pairs.append({"distance": distance, "data": pair_data_item})
            
        distances_and_pairs.sort(key=lambda x: x["distance"], reverse=True)

        selected_pair_for_archival = None
        for item in distances_and_pairs:
            current_pair_data = item["data"]
            u_idx = current_pair_data["user_idx"]
            a_idx = current_pair_data["assistant_idx"]

            safe_to_remove = True
            if u_idx > 0 and (a_idx + 1) < len(self._messages):
                prev_message_role = self._messages[u_idx - 1].role
                next_message_role = self._messages[a_idx + 1].role
                if prev_message_role == "assistant" and next_message_role == "assistant":
                    safe_to_remove = False
                    printd(f"STRUCTURAL WARNING: Skipping removal of pair (user index {u_idx}, assistant index {a_idx}) to avoid assistant-assistant sequence.")
            
            if safe_to_remove:
                selected_pair_for_archival = current_pair_data
                printd(f"STRUCTURALLY SAFE: Selected pair (user index {u_idx}, assistant index {a_idx}) for archival.")
                break 
                
        if selected_pair_for_archival is None:
            printd("Could not identify a structurally safe candidate pair from centroid for archival. Skipping this step.")
            # return False
            return False, None # Indicate no archival, no content

        removed_user_msg = selected_pair_for_archival["user_msg"]
        removed_assistant_msg = selected_pair_for_archival["assistant_msg"]
        content_to_archive = f"User: {removed_user_msg.text}\\nAssistant: {removed_assistant_msg.text}"

        try:
            self.persistence_manager.archival_memory.insert(content_to_archive)
            # self.store_centroid(centroid) # Store centroid if archival based on it was successful
            printd(f"Successfully triggered archival insertion for: User: {removed_user_msg.text[:50]}... | Assistant: {removed_assistant_msg.text[:50]}...")
        except Exception as e:
            printd(f"CRITICAL Error calling archival_memory.insert for least similar candidate pair: {e}")
            traceback.print_exc()
            return False, None # Indicate no archival, no content

        ids_to_remove_from_context = {removed_user_msg.id, removed_assistant_msg.id}
        original_len = len(self._messages)
        self._messages = [m for m in self._messages if m.id not in ids_to_remove_from_context]
        removed_count = original_len - len(self._messages)

        if removed_count > 0:
            print(f"AUTO-ARCHIVING (structurally safe, furthest from centroid): User message (original context index {selected_pair_for_archival['user_idx']}): {removed_user_msg.text}")
            print(f"AUTO-ARCHIVING (structurally safe, furthest from centroid): Assistant message (original context index {selected_pair_for_archival['assistant_idx']}): {removed_assistant_msg.text}")
            printd(f"Removed {removed_count} messages from active context after auto-archival.")
            self.update_state()
            self._trigger_archival_embeddings_log(log_reason="auto_archival") # Log after successful auto-archival
            return True, content_to_archive 
        else:
            printd("Warning: Archival insertion succeeded, but no messages were removed from active context.")
            # return False
            return False, None # Indicate no archival, no content

    def step(
        self,
        user_message: Union[Message, str],
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,
        recreate_message_timestamp: bool = True,
    ) -> Tuple[List[Union[dict, Message]], bool, bool, bool, Optional[int]]:
        """Top-level event message handler for the MemGPT agent"""

        def strip_name_field_from_user_message(user_message_text: str) -> Tuple[str, Optional[str]]:
            try:
                user_message_json = dict(json.loads(user_message_text, strict=JSON_LOADS_STRICT))
                name = user_message_json.pop("name", None)
                clean_message = json.dumps(user_message_json, ensure_ascii=JSON_ENSURE_ASCII)
            except Exception as e:
                print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed with: {e}")
                # Fallback if JSON parsing fails but we want to proceed
                return user_message_text, None 
            return clean_message, name

        def validate_json(user_message_text: str, raise_on_error: bool) -> str:
            try:
                user_message_json = dict(json.loads(user_message_text, strict=JSON_LOADS_STRICT))
                return json.dumps(user_message_json, ensure_ascii=JSON_ENSURE_ASCII)
            except Exception as e:
                print(f"{CLI_WARNING_PREFIX}couldn't parse user input message as JSON: {e}")
                if raise_on_error:
                    raise e
                return user_message_text # Return original if not raising, allowing non-JSON

        try:
            # Step 0: add user message
            if user_message is not None:
                if isinstance(user_message, Message):
                    # Ensure text is valid JSON string for strip_name_field
                    try:
                        json.loads(user_message.text, strict=JSON_LOADS_STRICT)
                        user_message_text_content = user_message.text
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain string for stripping logic (which might just pass it through)
                        user_message_text_content = user_message.text
                        printd(f"User message text is not JSON, proceeding with raw text for name stripping: {user_message_text_content[:100]}")

                    self._log_message_roles(self._messages, context_label="Step Start (User is Message Obj)") # Logging message roles

                    cleaned_user_message_text, name = strip_name_field_from_user_message(user_message_text_content)

                    if name is not None:
                        user_message.text = cleaned_user_message_text
                        user_message.name = name
                    if recreate_message_timestamp:
                        user_message.created_at = get_utc_time()

                elif isinstance(user_message, str):
                    # user_message_str_content = validate_json(user_message, False) # validate_json expects a string and returns one
                    # cleaned_user_message_text, name = strip_name_field_from_user_message(user_message_str_content)
                    # Allow non-JSON through if validate_json doesn't raise
                    validated_user_str = validate_json(user_message, False) # False means don't raise, returns original on error
                    cleaned_user_message_text, name = strip_name_field_from_user_message(validated_user_str)


                    user_message = Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={"role": "user", "content": cleaned_user_message_text, "name": name},
                    )
                else:
                    raise ValueError(f"Bad type for user_message: {type(user_message)}")
                
                self._log_message_roles(self._messages, context_label="Step Start (After User Message Processed, Before Appending to Input Seq)") # Logging message roles

                self.interface.user_message(user_message.text, msg_obj=user_message)
                input_message_sequence = self.messages + [user_message.to_openai_dict()]
            else:
                self._log_message_roles(self._messages, context_label="Step Start (No User Message)") # Logging message roles
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
                        first_message=True,
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

            # Step 2 & 3: Handle AI response and potential function calls
            response_message = response.choices[0].message
            # response_message_copy = response_message.model_copy() # model_copy() might not exist, use deepcopy or ensure it's handled
            all_response_messages, heartbeat_request, function_failed = self._handle_ai_response(response_message)

            # Step 4: Process messages for context and storage
            all_new_messages_this_turn = []
            if user_message: # If there was an initial user message for this step
                all_new_messages_this_turn.append(user_message)
            all_new_messages_this_turn.extend(all_response_messages)


            # Automatic embedding of the *current* user-assistant pair, if one was formed
            if user_message and all_response_messages and all_response_messages[0].role == "assistant":
                try:
                    current_pair_user_msg = user_message
                    current_pair_assistant_msg = all_response_messages[0]
                    combined_texts = self.create_message_pair_embeddings(current_pair_user_msg, [current_pair_assistant_msg])
                    if combined_texts:
                        combined_text = "\\n".join(combined_texts)
                        embedding = self.persistence_manager.embedding_model.get_text_embedding(combined_text)
                        passage = self.persistence_manager.archival_memory.create_passage(combined_text, embedding) # create_passage is on ArchivalMemory
                        self.persistence_manager.vector_store.insert(passage)
                        printd(f"Automatically embedded and stored current user-assistant turn ({current_pair_user_msg.id}, {current_pair_assistant_msg.id}) to vector store.")
                        self._trigger_archival_embeddings_log(log_reason="turn_embedding") # Log after successful turn embedding
                except Exception as e:
                    printd(f"Error in automatic embedding of current turn: {e}")
            
            # Append all messages from this turn (user + AI responses) to the main context first
            if all_new_messages_this_turn:
                # Ensure all are Message objects before appending
                processed_new_messages = []
                for msg_data in all_new_messages_this_turn:
                    if isinstance(msg_data, Message):
                        processed_new_messages.append(msg_data)
                    else:
                        printd(f"Warning: Encountered non-Message object in all_new_messages_this_turn: {type(msg_data)}")
                if processed_new_messages:
                    self._append_to_messages(processed_new_messages)

            # Memory pressure check (uses response.usage from the LLM call)
            current_total_tokens = response.usage.total_tokens
            active_memory_warning = False # Reset for this step's check
            if self.agent_state.llm_config.context_window is None:
                print(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
                self.agent_state.llm_config.context_window = (
                    LLM_MAX_TOKENS.get(self.model) or LLM_MAX_TOKENS["DEFAULT"]
                )
            
            effective_context_window = int(self.agent_state.llm_config.context_window)

            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * effective_context_window:
                printd(
                    f"{CLI_WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * effective_context_window}"
                )
                active_memory_warning = True # Indicate threshold was crossed for this current step's logic
                # Only set the persistent flag if it wasn't already set, to avoid repeated warnings from the same state
                if not self.agent_alerted_about_memory_pressure:
                    self.agent_alerted_about_memory_pressure = True 
            else:
                printd(
                    f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * effective_context_window}"
                )
                self.agent_alerted_about_memory_pressure = False # Reset persistent flag if below threshold

            # Now, apply the centroid-based archival to the updated context
            # ONLY if the memory warning threshold was just determined to be active from the last API call
            if active_memory_warning: # This flag is set based on current_total_tokens vs threshold
                printd(f"Memory pressure threshold met (or was already high based on last API call), attempting centroid archival.")
                self._log_message_roles(self._messages, context_label="Before Centroid Archival")

                # Log messages since last genuine user input
                last_genuine_user_idx = -1
                for idx in range(len(self._messages) - 1, -1, -1):
                    msg = self._messages[idx]
                    if msg.role == "user":
                        is_system_generated_user_msg = False
                        try:
                            msg_json = json.loads(msg.text)
                            if isinstance(msg_json, dict) and msg_json.get("type") in ["system_alert", "system_event"]:
                                is_system_generated_user_msg = True
                        except json.JSONDecodeError:
                            pass # Not a JSON, so not one of these system types
                        
                        if msg.text.startswith("Summary of"):
                            is_system_generated_user_msg = True
                        
                        if not is_system_generated_user_msg:
                            last_genuine_user_idx = idx
                            break
                
                if last_genuine_user_idx != -1:
                    printd(f"--- Messages since last genuine user input (index {last_genuine_user_idx}) leading to archival attempt ---")
                    for i in range(last_genuine_user_idx, len(self._messages)):
                        msg_to_log = self._messages[i]
                        content_snippet = msg_to_log.text.replace("\n", " ")[:100]
                        if len(msg_to_log.text) > 100:
                            content_snippet += "..."
                        
                        # Log message type and any additional metadata
                        additional_info = ""
                        if msg_to_log.role == "assistant" and msg_to_log.tool_calls:
                            additional_info = f" (Tool Calls: {[tc.function['name'] for tc in msg_to_log.tool_calls]})"
                        elif msg_to_log.role == "tool":
                            additional_info = f" (Tool)"
                        elif msg_to_log.role == "system":
                            additional_info = f" (System)"
                        elif msg_to_log.role == "user":
                            # Try to detect if it's a system-generated message
                            try:
                                msg_json = json.loads(msg_to_log.text)
                                if isinstance(msg_json, dict) and "type" in msg_json:
                                    additional_info = f" (System-generated: {msg_json['type']})"
                            except json.JSONDecodeError:
                                pass

                        print(f"  Idx {i}: Role='{msg_to_log.role}'{additional_info}, Content='{content_snippet}'")
                    print(f"--------------------------------------------------------------------------------")
                else:
                    print("Could not find a previous genuine user message to log sequence from.")

                archival_success = False
                archived_content_summary = None # Initialize
                try:
                    # _archive_least_similar_context_pair now returns (bool, str_or_None)
                    archival_success, archived_content_text = self._archive_least_similar_context_pair()
                    if archival_success and archived_content_text:
                        # Create a brief summary/snippet for the LLM
                        snippet_length = 100 # Characters
                        archived_content_summary = archived_content_text.replace("\n", " ")[:snippet_length]
                        if len(archived_content_text) > snippet_length:
                            archived_content_summary += "..."
                    self._log_message_roles(self._messages, context_label="After Centroid Archival Attempt")
                except Exception as e:
                    printd(f"CRITICAL Error during _archive_least_similar_context_pair: {e}")
                    traceback.print_exc()
                
                if archival_success:
                    # event_details = f"A message pair was automatically archived. Content snippet: '{archived_content_summary}'"
                    event_details = "An irrelevant message pair was automatically archived."
                    system_event_content = {
                        "type": "system_event",
                        "event": "auto_archival_complete",
                        "details": event_details
                    }
                    system_event_message_obj = Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={"role": "user", "content": json.dumps(system_event_content, ensure_ascii=JSON_ENSURE_ASCII)}
                    )
                    self._append_to_messages([system_event_message_obj])
                    self._log_message_roles(self._messages, context_label="After Injecting Archival System Event")

            else:
                printd(f"Memory pressure threshold not met, skipping centroid archival.")

            self._trigger_archival_embeddings_log(log_reason="end_of_step")
            
            # Determine what messages to return to the caller (typically just the AI's output from this turn)
            # `all_response_messages` are the Message objects generated by the AI in this step
            messages_to_return_openai_dict = [msg.to_openai_dict() for msg in all_response_messages] if return_dicts else all_response_messages
            
            completion_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None

            return messages_to_return_openai_dict, heartbeat_request, function_failed, active_memory_warning, completion_tokens

        except Exception as e:
            printd(f"step() failed\\nuser_message = {user_message}\\nerror = {e}")
            traceback.print_exc() # Print full traceback for easier debugging of step failures

            if is_context_overflow_error(e):
                printd("Context overflow error detected. Attempting to summarize messages inplace.")
                self.summarize_messages_inplace()
                # Try step again
                # Need to ensure user_message is in the correct format for retry
                # If user_message was an object, it should still be an object
                return self.step(user_message, first_message=first_message, skip_verify=skip_verify, return_dicts=return_dicts, recreate_message_timestamp=recreate_message_timestamp)
            else:
                printd(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e

        finally:
            # This block will always execute, even if an error occurs in the try block
            # However, if we want to log embeddings only on successful completion of a step,
            # this might not be the right place. The current placement is after most operations.
            pass # Current end-of-step logging is handled before return

    def _trigger_archival_embeddings_log(self, log_reason: str):
        """Logs a snapshot of all current archival memory embeddings."""
        # if not self.log_step_embeddings: # DEBUG: Temporarily bypass this check
        #     return
        self.log_step_embeddings = True # DEBUG: Force true for this method call

        # Increment step_count only for the primary "end_of_step" log to count agent interactions
        if log_reason == "end_of_step":
            self.step_count += 1

        try:
            log_dir_path = None
            if self.embeddings_log_dir:
                log_dir_path = Path(self.embeddings_log_dir)
            else:
                if hasattr(self.persistence_manager, 'agent_dir') and self.persistence_manager.agent_dir:
                    log_dir_path = self.persistence_manager.agent_dir / "step_embeddings_logs"
                else:
                    from memgpt.config import MemGPTConfig # Corrected import location
                    default_logs_path = Path(MemGPTConfig.config_path()).parent / "embeddings_logs" / str(self.agent_state.id)
                    log_dir_path = default_logs_path
                    print(f"Warning: self.persistence_manager.agent_dir not found, using default log path: {log_dir_path}") # This is already print
            
            if log_dir_path:
                # print(f"AGENT_LOG_DEBUG: Determined log_dir_path for '{log_reason}' embeddings: {log_dir_path.resolve()}")
                # print(f"AGENT_LOG_DEBUG: Attempting to create directory (and parents) for '{log_reason}': {log_dir_path}")
                try:
                    log_dir_path.mkdir(parents=True, exist_ok=True)
                    # print(f"AGENT_LOG_DEBUG: Directory check/creation complete for '{log_reason}': {log_dir_path}")
                except Exception as e_mkdir:
                    # print(f"AGENT_LOG_CRITICAL: Failed to create directory {log_dir_path} for '{log_reason}': {e_mkdir}") # Changed from CRITICAL DEBUG
                    raise
                
                timestamp = get_utc_time().strftime("%Y%m%d_%H%M%S_%f")
                # Use current self.step_count, reason, and timestamp for unique filename
                filename = f"step_{self.step_count:05d}_{log_reason}_{timestamp}_embeddings.json"
                filepath = log_dir_path / filename

                passages_data = self.persistence_manager.archival_memory.storage.get_all()
                embeddings_to_log = []
                for passage in passages_data:
                    embedding_list = None
                    if hasattr(passage, 'embedding') and passage.embedding is not None:
                        if hasattr(passage.embedding, 'tolist'):
                            embedding_list = passage.embedding.tolist()
                        elif isinstance(passage.embedding, list):
                            embedding_list = passage.embedding 
                    
                    embeddings_to_log.append({
                        "id": str(passage.id) if hasattr(passage, 'id') else None,
                        "text_snippet": passage.text[:200] + ("..." if len(passage.text) > 200 else "") if hasattr(passage, 'text') else None,
                        "embedding": embedding_list
                    })
                
                with open(filepath, 'w') as f:
                    json.dump(embeddings_to_log, f, indent=2)
                print(f"AGENT_LOG_INFO: Logged {len(embeddings_to_log)} archival embeddings ({log_reason}) to {filepath}")
            else:
                print(f"AGENT_LOG_ERROR: Could not determine a valid directory for logging '{log_reason}' step embeddings.")

        except Exception as e_log:
            print(f"AGENT_LOG_ERROR: Error during '{log_reason}' step_embeddings logging: {e_log}")
            import traceback
            traceback.print_exc()

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

    def store_centroid(self, centroid: np.ndarray):
        """Store a centroid in the history with its timestamp.
        
        Args:
            centroid: The centroid vector to store
        """
        self.centroid_history.append(centroid.tolist())  # Convert to list for JSON serialization
        self.centroid_timestamps.append(get_utc_time().isoformat())

    def get_centroid_history(self) -> List[Tuple[np.ndarray, str]]:
        """Get the history of centroids with their timestamps.
        
        Returns:
            List of tuples containing (centroid, timestamp)
        """
        return [(np.array(centroid), timestamp) for centroid, timestamp in zip(self.centroid_history, self.centroid_timestamps)]

    def get_furthest_embeddings(self, k: int = 5) -> List[Tuple[str, float]]:
        """Calculate the centroid of stored embeddings and return the top k furthest vectors.
        
        Args:
            k: Number of furthest vectors to return
            
        Returns:
            List of tuples containing (text, distance) for the top k furthest vectors
        """
        try:
            # Get all embeddings from the vector store
            all_embeddings = self.persistence_manager.vector_store.get_all_embeddings()
            
            if not all_embeddings:
                return []
                
            # Extract embeddings and texts
            embeddings = [item['embedding'] for item in all_embeddings]
            texts = [item['text'] for item in all_embeddings]
            
            # Calculate centroid (mean of all embeddings)
            centroid = np.mean(embeddings, axis=0)
            
            # Store the centroid in history
            self.store_centroid(centroid)
            
            # Calculate distances from centroid
            distances = []
            for i, embedding in enumerate(embeddings):
                # Calculate Euclidean distance
                distance = np.linalg.norm(embedding - centroid)
                distances.append((texts[i], distance))
            
            # Sort by distance (descending) and get top k
            distances.sort(key=lambda x: x[1], reverse=True)
            return distances[:k]
            
        except Exception as e:
            printd(f"Error calculating furthest embeddings: {e}")
            return []


def save_agent(agent: Agent, ms: MetadataStore):
    """Save agent to metadata store"""

    agent.update_state()
    agent_state = agent.agent_state

    if ms.get_agent(agent_id=agent_state.id):
        ms.update_agent(agent_state)
    else:
        ms.create_agent(agent_state)
