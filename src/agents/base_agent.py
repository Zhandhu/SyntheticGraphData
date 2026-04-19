"""Base class for all LLM-powered agents."""

from src.utils.llm_client import LLMClient


class BaseAgent:
    """Base agent with prompt template and LLM client."""

    def __init__(self, llm_client: LLMClient, prompt_path: str, call_params_key: str):
        self.llm = llm_client
        with open(prompt_path) as f:
            self.prompt_template = f.read()
        self.call_params_key = call_params_key

    def format_prompt(self, **kwargs) -> list[dict]:
        """Format the prompt template with provided variables.
        Uses simple {variable} substitution."""
        system_msg = self.prompt_template.format(**kwargs)
        return [{"role": "system", "content": system_msg}]

    async def run(self, **kwargs):
        raise NotImplementedError
