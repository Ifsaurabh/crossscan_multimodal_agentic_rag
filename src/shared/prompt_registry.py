from shared import langfuse_client

PROMPT_LABEL = "production"


def get_prompt(name: str, fallback: str) -> str:
    """Fetch the production-labelled version of a prompt from Langfuse Prompt
    Management, falling back to the local constant if Langfuse is disabled,
    unreachable, or has no such prompt yet. Same opportunistic
    attempt-with-fallback pattern as the Gemini prompt caching.

    The local prompts and their syncing to Langfuse are in retrieval/prompt_sync.py."""
    client = langfuse_client.get_client()
    if client is None:
        return fallback
    try:
        prompt = client.get_prompt(name, label=PROMPT_LABEL, type="text", fallback=fallback)
        return prompt.prompt
    except Exception:
        return fallback
