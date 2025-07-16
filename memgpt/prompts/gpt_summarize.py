WORD_LIMIT = 100

def get_system_prompt(prompt_type: str = "memgpt_default") -> str:
    """
    Returns the appropriate system prompt based on the prompt type.
    
    Args:
        prompt_type: The type of prompt to use ("xml" or "memgpt_default")
        
    Returns:
        The system prompt string
    """
    if prompt_type == "xml":
        return f"""
<summarizer word_limit=\"{WORD_LIMIT}\">
  <role>
    You are a conversation summarizer.
  </role>

  <instructions>
    Summarize the given window of messages **from the AI's first-person perspective**.
    The window may be incomplete, so focus only on what is present.
    Your *entire* output (summary + bullets) must contain ≤ {WORD_LIMIT} words.
  </instructions>

  <output-format>
    <summary>Few concise sentences …</summary>
    <key-facts>
      <!-- List only items explicitly stated in the window -->
      <bullet>Names / roles / organizations</bullet>
      <bullet>Dates and times</bullet>
      <bullet>Locations and places</bullet>
      <bullet>Important numbers (money, counts, measurements, etc.)</bullet>
    </key-facts>
  </output-format>

  <rules>
    <r1>Do **not** add tags not defined above.</r1>
    <r2>No pre- or post-amble outside the XML.</r2>
  </rules>
</summarizer>
"""
  # <message-types>
  #   <assistant>Inner monologue (not visible to user).</assistant>
  #   <send_message>Assistant content that the user sees.</send_message>
  #   <function>Outputs of any tool calls.</function>
  # </message-types>

    elif prompt_type == "memgpt_default":
        return f"""
Your job is to summarize a history of previous messages in a conversation between an AI persona and a human.
The conversation you are given is a from a fixed context window and may not be complete.
Messages sent by the AI are marked with the 'assistant' role.
The AI 'assistant' can also make calls to functions, whose outputs can be seen in messages with the 'function' role.
Things the AI says in the message content are considered inner monologue and are not seen by the user.
The only AI messages seen by the user are from when the AI uses 'send_message'.
Messages the user sends are in the 'user' role.
The 'user' role is also used for important system events, such as login events and heartbeat events (heartbeats run the AI's program without user action, allowing the AI to act without prompting from the user sending them a message).
Summarize what happened in the conversation from the perspective of the AI (use the first person).
Keep your summary less than {WORD_LIMIT} words, do NOT exceed this word limit.
Only output the summary, do NOT include anything else in your output.
"""
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}. Valid options are 'xml' or 'memgpt_default'.")

# For backward compatibility, provide the default as SYSTEM
SYSTEM = get_system_prompt("memgpt_default")
