import hashlib

import langfuse_client

PROMPT_LABEL = "production"

# Local source of truth: (langfuse prompt name) -> loader returning the text.
# Imported lazily to avoid a circular import with the agent modules, which
# themselves call get_prompt().
_LOCAL_PROMPT_SOURCES = {
    "transform-route": ("agent_transform_route", "SYSTEM_INSTRUCTION"),
    "quality-check": ("agent_quality_generate", "QUALITY_CHECK_SYSTEM_INSTRUCTION"),
    "generate-answer": ("agent_quality_generate", "GENERATE_SYSTEM_INSTRUCTION"),
    "general-knowledge": ("agent_quality_generate", "GENERAL_KNOWLEDGE_SYSTEM_INSTRUCTION"),
    "conversation-summary": ("memory", "SUMMARY_SYSTEM_INSTRUCTION"),
}


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def get_prompt(name: str, fallback: str) -> str:
    """Fetch the production-labelled version of a prompt from Langfuse Prompt
    Management, falling back to the local constant if Langfuse is disabled,
    unreachable, or has no such prompt yet. Same opportunistic
    attempt-with-fallback pattern as the Gemini prompt caching."""
    client = langfuse_client.get_client()
    if client is None:
        return fallback
    try:
        prompt = client.get_prompt(name, label=PROMPT_LABEL, type="text", fallback=fallback)
        return prompt.prompt
    except Exception:
        return fallback


def local_prompts() -> dict:
    """name -> local prompt text for every managed prompt."""
    import importlib

    return {
        name: getattr(importlib.import_module(module), attr)
        for name, (module, attr) in _LOCAL_PROMPT_SOURCES.items()
    }


def local_prompt_versions() -> dict:
    """name -> short content hash of the local prompt text. Used to tag
    evaluation runs so results trace back to the exact prompts used."""
    return {name: prompt_hash(text) for name, text in local_prompts().items()}


def sync_prompts() -> dict:
    """Push local prompts to Langfuse as new versions, only where the content
    actually changed (Langfuse versions are immutable, so unchanged prompts
    are skipped to avoid pointless version bumps). Returns name -> status."""
    client = langfuse_client.get_client()
    if client is None:
        raise RuntimeError("Langfuse is not configured/enabled - cannot sync prompts.")

    results = {}
    for name, text in local_prompts().items():
        existing = None
        try:
            existing = client.get_prompt(name, label=PROMPT_LABEL, type="text")
        except Exception:
            pass

        if existing is not None and existing.prompt == text:
            results[name] = f"unchanged (v{existing.version})"
            continue

        created = client.create_prompt(name=name, prompt=text, labels=[PROMPT_LABEL], type="text")
        results[name] = f"created v{created.version}"
    return results


if __name__ == "__main__":
    for prompt_name, status in sync_prompts().items():
        print(f"{prompt_name}: {status}")
