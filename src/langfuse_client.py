import os

from dotenv import load_dotenv

load_dotenv()

_client = None
_client_failed = False


def is_enabled() -> bool:
    """Langfuse is opt-out: on when keys are configured, off if
    LANGFUSE_DISABLED is set (tests) or keys are missing."""
    if os.environ.get("LANGFUSE_DISABLED"):
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_client():
    """Shared Langfuse client, or None if disabled/unavailable. Never raises -
    observability must not be able to break the pipeline."""
    global _client, _client_failed
    if not is_enabled() or _client_failed:
        return None
    if _client is None:
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
                timeout=5,
            )
        except Exception as e:
            print(f"   (Langfuse unavailable, continuing without it: {e})")
            _client_failed = True
            return None
    return _client


def get_callbacks() -> list:
    """LangChain/LangGraph callback handlers that trace a graph run to
    Langfuse. Empty list when Langfuse is off, so callers can always pass
    the result straight into graph.invoke(config={"callbacks": ...})."""
    if get_client() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as e:
        print(f"   (Langfuse tracing unavailable, continuing without it: {e})")
        return []


def flush():
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
