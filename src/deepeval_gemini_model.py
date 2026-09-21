from deepeval.models.base_model import DeepEvalBaseLLM

import llm_connection
from retrieval_config import GEMINI_MODEL


class GeminiDeepEvalModel(DeepEvalBaseLLM):
    """Lets DeepEval components (Synthesizer, metrics) run on the project's
    LLM connection instead of requiring an OpenAI key. The name is historical:
    it now goes through llm_connection.generate() on the "evaluation" tier,
    so it uses that tier's models and fallback like the rest of the app.

    Deliberately implements `generate`/`a_generate` WITHOUT a `schema`
    parameter. DeepEval treats this as a "non-native" custom model: it
    first tries calling with `schema=...`, catches the resulting
    TypeError, then falls back to plain-text generation + its own JSON
    parsing. That fallback is exactly what we want.
    """

    def __init__(self, model: str = GEMINI_MODEL):
        self.model_name = model
        super().__init__(model)

    def load_model(self):
        # No client to build: llm_connection creates provider clients lazily.
        # (Tests may set .model to a fake Gemini client.)
        return None

    def generate(self, prompt: str) -> str:
        return llm_connection.generate("", prompt, task="deepeval_judge", client=self.model).text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name
