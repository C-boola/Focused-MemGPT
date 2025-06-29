"""
We introduce a new task based on the synthetic Key-Value
retrieval proposed in prior work (Liu et al., 2023a). The
goal of this task is to demonstrate how MemGPT can col-
late information from multiple data sources. In the original
KV task, the authors generated a synthetic dataset of key-
value pairs, where each key and value is a 128-bit UUID
(universally unique identifier). The agent is then given a
key, and asked to return the associated value for the key.
We create a version of the KV task, nested KV retrieval,
where values themselves may be keys, thus requiring the
agent to perform a multi-hop lookup. In our setup, we fix
the total number of UUIDs pairs to 140, corresponding to
roughly 8k tokens (the context length of our GPT-4 base-
line). We vary the total number of nesting levels from 0
(the initial key-value pair's value is not a key) to 4 (ie 4
total KV lookups are required to find the final value), and
sample 30 different ordering configurations including both
the initial key position and nesting key positions.
"""
import math
import json
import argparse
import uuid
import os
from typing import List
from tqdm import tqdm
from collections import OrderedDict
from openai import OpenAI
import openai
from memgpt import create_client
from memgpt.data_types import Message, AgentState, Passage
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.config import MemGPTConfig
from memgpt.metadata import MetadataStore
from memgpt import utils
from memgpt.constants import MAX_PAUSE_HEARTBEATS, RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE, JSON_ENSURE_ASCII

from typing import Optional

from paper_experiments.utils import load_gzipped_file, get_experiment_config
import time


# TODO: update personas
NESTED_PERSONA = "You are MemGPT DOC-QA bot. Your job is to answer questions about documents that are stored in your archival memory. The answer to the users question will ALWAYS be in your archival memory, so remember to keep searching if you can't find the answer. DO NOT STOP SEARCHING UNTIL YOU VERIFY THAT THE VALUE IS NOT A KEY. Do not stop making nested lookups until this condition is met."  # TODO decide on a good persona/human
NESTED_HUMAN = "The user will ask you questions about documents. Answer them to the best of your ability."
DEFAULT_FILE = "paper_experiments/nested_kv_task/data/kv-retrieval-140_keys.jsonl.gz"
AGENT_NAME = "kv_task_agent"


# memgpt currently does not support text search over archival memory, however this experiment uses synthetic data which is out of distribution for the embedding model.
# we temporarily override archival memory search with text search for this experiment
def archival_memory_text_search(self, query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search archival memory using semantic (embedding-based) search.

    Args:
        query (str): String to search for.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = 10
    results = self.persistence_manager.archival_memory.storage.query_text(query, limit=count, offset=page * count)
    total = len(results)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"memory: {d.text}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def load_jsonl_to_list(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_nested_kv_task(config: MemGPTConfig, memgpt_client, kv_dict, user_message):
    utils.DEBUG = True
    print("Starting run_nested_kv_task function")

    # delete agent if exists
    user_id = uuid.UUID(config.anon_clientid)
    agent_name = f"{AGENT_NAME}_{config.default_llm_config.model}"
    print(f"User ID: {user_id}")
    
    # First try to delete agent using the client directly by name
    try:
        print(f"Trying to delete existing agent with name: {agent_name}")
        # Check if agent exists
        print("Listing existing agents...")
        existing_agents = memgpt_client.list_agents()
        print(f"Got existing agents: {type(existing_agents)}")
        
        # Handle different possible formats returned by list_agents
        if hasattr(existing_agents, 'agents'):
            agents_list = existing_agents.agents
            print(f"Found agents list as attribute, length: {len(agents_list)}")
        elif isinstance(existing_agents, dict) and 'agents' in existing_agents:
            agents_list = existing_agents['agents']
            print(f"Found agents list in dict, length: {len(agents_list)}")
        else:
            print(f"Unexpected format for list_agents response: {type(existing_agents)} - {existing_agents}")
            agents_list = existing_agents if isinstance(existing_agents, list) else []
            print(f"Using fallback agents list, length: {len(agents_list) if isinstance(agents_list, list) else 'not a list'}")
        
        agent_exists = False
        agent_id = None
        
        print("Checking if our agent exists in the list...")
        for agent in agents_list:
            if hasattr(agent, 'name') and agent.name == agent_name:
                agent_exists = True
                agent_id = agent.id
                print(f"Found existing agent {agent_name} with ID {agent_id}")
                break
            elif isinstance(agent, dict) and agent.get('name') == agent_name:
                agent_exists = True
                agent_id = agent.get('id')
                print(f"Found existing agent {agent_name} with ID {agent_id}")
                break
        
        if not agent_exists:
            print(f"Agent {agent_name} not found in existing agents")
        
        if agent_exists and agent_id:
            try:
                # Convert agent_id to UUID object if it's a string
                if isinstance(agent_id, str):
                    agent_id = uuid.UUID(agent_id)
                    print(f"Converted agent_id to UUID: {agent_id}")
                
                print(f"Attempting to delete agent with ID: {agent_id}")
                memgpt_client.delete_agent(agent_id)
                print(f"Successfully deleted existing agent {agent_name}")
            except Exception as e:
                print(f"Error deleting agent by ID: {e}")
                
                # As a fallback, try to force rename the existing agent
                print(f"Will create agent with a different name to avoid conflict")
                agent_name = f"{AGENT_NAME}_{config.default_llm_config.model}_{int(time.time())}"
    except Exception as e:
        print(f"Error checking/deleting agent: {e}")
        import traceback
        traceback.print_exc()
        # If we encounter an error, use a unique agent name to avoid conflicts
        agent_name = f"{AGENT_NAME}_{config.default_llm_config.model}_{int(time.time())}"

    # Create a new Agent that models the scenario setup
    print(f"Creating agent with name: {agent_name}")
    try:
        agent_state = memgpt_client.create_agent(
            name=agent_name,
            persona=NESTED_PERSONA,
            human=NESTED_HUMAN,
        )
        print(f"Successfully created agent with ID: {agent_state.id}")
    except Exception as e:
        print(f"Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        raise

    # get agent
    try:
        print(f"Attempting to get agent with ID: {agent_state.id}")
        agent = memgpt_client.server._get_or_load_agent(user_id, agent_state.id)
        print("Successfully got agent")
        print("Adding custom archival_memory_search function")
        agent.functions_python["archival_memory_search"] = archival_memory_text_search
        print("Successfully added custom function")

        # insert into archival
        print(f"Inserting {len(kv_dict)} key-value pairs into archival memory")
        for i, (k, v) in tqdm(enumerate(kv_dict.items())):
            if i % 20 == 0:
                print(f"Inserted {i} items so far")
            document_string = f"Key-value pair: key = {k}, value = {v}"
            # The insert method no longer accepts compute_embedding parameter
            agent.persistence_manager.archival_memory.insert(document_string)
        print(f"Inserted {len(agent.persistence_manager.archival_memory)} into archival memory.")
    except AttributeError as e:
        print(f"Warning: Could not directly access agent. Error: {e}")
        import traceback
        traceback.print_exc()
        # Try alternative approach - insert archival memory through client API
        print("Falling back to client API for inserting archival memory")
        for i, (k, v) in tqdm(enumerate(kv_dict.items())):
            if i % 20 == 0:
                print(f"Inserted {i} items so far via API")
            document_string = f"Key-value pair: key = {k}, value = {v}"
            memgpt_client.insert_archival_memory(agent_state.id, document_string)
        print("Inserted key-value pairs into archival memory through API.")

    print(f"Sending user message: {user_message[:100]}...")
    response = memgpt_client.user_message(agent_id=agent_state.id, message=user_message)
    print(f"Received response of type: {type(response)}")

    # for open models, make extra clear we need th response
    if config.default_llm_config.model_endpoint_type != "openai":
        print("Model endpoint is not OpenAI, sending follow-up question")
        followup_message = "What is your final answer? Respond with only the answer."
        print(f"Sending follow-up message: {followup_message}")
        response = memgpt_client.user_message(agent_id=agent_state.id, message=followup_message)
        print(f"Received final response of type: {type(response)}")
    
    print("Task completed successfully, returning response")
    return response


def run_baseline(model_id, query_key, kv_dict):
    def create_prompt(query_key, kv_dict):
        prompt = " ".join(
            [
                "Below is a JSON object containing key-value pairings, all keys and values are 128-bit UUIDs, and your task is to return the value associated with the specified key.",
                "If a value itself is also a key, return the value of that key (do a nested lookup).",
                "For example, if the value of 'x' is 'y', but 'y' is also a key, return the value of key 'y'.",
            ]
        )

        data_string = ",\n".join(f'"{k}": "{v}"' for k, v in kv_dict.items())
        prompt += f"\n\nJSON data: {{\n{data_string}\n}}"

        prompt += f'\n\nYour task is to provide the value for the following key: "{query_key}". Answer only with the value, nothing else.'

        return prompt

    user_message = create_prompt(query_key, kv_dict)
    print(user_message)

    model_dict = {
        "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
        "gpt-3.5": "gpt-3.5-turbo-16k",  # 140 K-Vs is approximately ~7/8k tokens, so it doesn't fit inside 3.5 base (4k limit)
        "gpt-4": "gpt-4",
        "gpt-4-1106-preview": "gpt-4-1106-preview",
        "gpt-4-0613": "gpt-4-0613",
    }
    model = model_dict[model_id] if model_id in model_dict else model_id

    if model_id == "ehartford/dolphin-2.5-mixtral-8x7b":
        # openai.base_url = "https://api.openai.com/v1/"
        openai.base_url = "https://api.memgpt.ai/v1/"

    print("base url", openai.base_url)
    # client = OpenAI()
    response = openai.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    # response = openai.ChatCompletion.create(
    #    model=model_dict[model_id],
    #    messages=[
    #        {"role": "user", "content": user_message},
    #    ]
    # )
    # print(response)
    print(response)
    content = response.choices[0].message.content
    print(content)
    return content
    # value_response = response['choices'][0]['message']['content']
    # return value_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("--model", type=str, help="The model to use")
    parser.add_argument("--nesting_levels", default=1, type=int, help="Nesting levels")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--task", default="kv", required=False, type=str, help="Task")
    parser.add_argument("--kv_data", default=DEFAULT_FILE, required=False, type=str, help="KV data")
    parser.add_argument("--baseline", default="memgpt", required=False, type=str, help="Baseline model (memgpt + model vs. model)")
    parser.add_argument("--rerun", default=False, action="store_true", help="Rerun task")

    args = parser.parse_args()
    assert args.task in ["kv", "kv_nested"], "Task must be one of 'kv' or 'kv_nested'"
    if args.baseline != "memgpt":
        # baseline should be the same as the model name
        assert args.baseline == args.model, "Baseline should be the same as the model name"

    # get provider
    if args.model == "ehartford/dolphin-2.5-mixtral-8x7b":
        provider = "local"
    else:
        provider = "openai"

    # skip if exists
    model_formatted = args.model.replace("/", "-")
    model_formatted = args.model.replace("/", "-")
    baseline_formatted = args.baseline.replace("/", "-")
    filename = f"results/nested_kv/nested_kv_results_{baseline_formatted}_nesting_{args.nesting_levels}_model_{model_formatted}_seed_{args.seed}.json"
    if not args.rerun and os.path.exists(filename):
        print("Skipping, file exists")
        print(filename)
        # exist program
        exit(0)

    if args.task in ["kv", "kv_nested"]:
        all_data = load_gzipped_file(args.kv_data)
        for example in all_data:
            data = example
            break

        ordered_kv_records = data["ordered_kv_records"]
        key_to_search = data["key"]

        # kv_dict = {k: v for k, v in ordered_kv_records}
        kv_dict = OrderedDict(ordered_kv_records)
        print(f"total number of keys: {len(ordered_kv_records)}")

        def print_kv(kv_d, limit=None):
            print("JSON data: {")
            count = 0
            for k, v in kv_d.items():
                print(f'"{k}": "{v}",')
                count += 1
                if limit and count > limit:
                    break
            print("}")

        def create_nested_kv_data(kv_d, nest_indices):
            """In-place operation"""
            assert isinstance(kv_d, OrderedDict)
            kv_d_list = list(kv_d)

            for i in range(len(nest_indices) - 1):
                current_idx = nest_indices[i]
                current_key = kv_d_list[current_idx]  # (key,value) -> key
                current_value = kv_d[current_key]  # this gets thrown away

                next_idx = nest_indices[i + 1]
                next_key = kv_d_list[next_idx]
                # overwrite
                kv_d[current_key] = next_key

                print(f"Nested {i+1}")
            print(f"Done")

        def get_nested_key(original_key, kv_d):
            key = original_key
            value = kv_d[key]

            print(f"Doing a lookup for key {key}")
            while value in kv_d:
                print(f"\t{key} -> {value} (value is a key, doing nested lookup)")
                key = value
                value = kv_d[key]
            return value

        if args.task == "kv_nested":
            data_filename = (
                f"paper_experiments/nested_kv_task/data/random_orderings_100_samples_140_indices_{args.nesting_levels}_levels.jsonl"
            )
            print(data_filename)
            loaded_data = load_jsonl_to_list(data_filename)
            print("LOADED", loaded_data, args.seed)
            swap_indices = loaded_data[args.seed]

            key_to_search_idx = swap_indices[0]
            key_to_search = list(kv_dict)[key_to_search_idx]
            key_to_search_init_value = kv_dict[key_to_search]

            # swap_indices = [0,16,100]
            create_nested_kv_data(kv_dict, swap_indices)
            # print_kv(kv_dict, limit=None)

            first_user_message = " ".join(
                [
                    # "I've given you a list of key-value pairs (keys are values are both UUIDs), which you can find in your archival memory.",
                    # "If a value itself is also a key, return the value of that key (do a nested lookup).",
                    "I've given you a list of key-value pairs which you can find in your archival memory, all keys and values are 128-bit UUIDs, and your task is to return the value associated with the specified key.",
                    "If a value itself is also a key, return the value of that key (do a nested lookup).",
                    "For example, if the value of 'x' is 'y', but 'y' is also a key, return the value of key 'y'.",
                    "Your task is to provide the value for the following key:",
                    # f"{key_to_search}"
                    f"{key_to_search}. Answer only with the value, nothing else.",
                ]
            )
        else:
            first_user_message = " ".join(
                [
                    "I've given you a list of key-value pairs, which you can find in your archival memory.",
                    "Your task is to provide the value for the following key:",
                    # f"{key_to_search}"
                    f"{key_to_search}. Answer only with the value, nothing else.",
                ]
            )

    if args.baseline == "memgpt":
        # create config
        postgres_uri = os.environ.get("MEMGPT_PGURI") or os.environ.get("PGVECTOR_TEST_DB_URL")
        if not postgres_uri:
            postgres_uri = "postgresql+pg8000://postgres:password@localhost:8888/postgres"
            print(f"No database URI found in environment variables. Using default: {postgres_uri}")
        else:
            print(f"Using database URI from environment: {postgres_uri}")
        
        # Check if the model is available
        print(f"Attempting to use model: {args.model}")
        
        try:
            config = get_experiment_config(postgres_uri, endpoint_type=provider, model=args.model)
            config.save()  # save config to file
            print(f"Configuration saved successfully with model {config.default_llm_config.model}")
            
            # create client
            print("Creating client...")
            memgpt_client = create_client()
            print("Client created successfully")
            
            # run task
            print("Starting task execution...")
            results = run_nested_kv_task(config, memgpt_client, kv_dict, first_user_message)
            print("Task completed successfully")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        results = run_baseline(args.model, key_to_search, kv_dict)

    final_result = {
        "model": args.model,
        "query_key": key_to_search,
        "query_key_value": get_nested_key(key_to_search, kv_dict),
        "nesting": args.nesting_levels,
        "results": results,
    }

    # Create directory if it doesn't exist
    os.makedirs("results/nested_kv", exist_ok=True)
    
    # write to JSON file
    with open(filename, "w") as f:
        json.dump(final_result, f, indent=4)
    
    print(f"Results saved to: {filename}")
