"""Interactive API key setup.

Called at the start of every entry-point script.  For each required key,
checks os.environ first (covers `source configs/env.sh` or a real env),
then prompts interactively if missing.  Keys are written into os.environ
for the duration of the process; they are never written to disk.
"""

import os
import getpass

_KEYS: list[tuple[str, str, bool]] = [
    # (env_var, description, required)
    ("OPENAI_API_KEY",           "OpenAI API key (platform.openai.com)",       True),
    ("ANTHROPIC_API_KEY",        "Anthropic API key (console.anthropic.com)",   False),
    ("HF_TOKEN",                 "HuggingFace token (huggingface.co/settings/tokens)", False),
    ("SEMANTIC_SCHOLAR_API_KEY", "Semantic Scholar API key (api.semanticscholar.org)", False),
    ("BRAVE_API_KEY",            "Brave Search API key (brave.com/search/api)", False),
]


def ensure_keys(require_openai: bool = True) -> None:
    """Prompt for any missing API keys and inject them into os.environ.

    Args:
        require_openai: If True, raise if OPENAI_API_KEY is still empty
                        after the prompt (i.e. user just hit Enter).
    """
    missing_any = False
    for var, desc, required in _KEYS:
        val = os.environ.get(var, "").strip()
        if not val:
            if not missing_any:
                print("\n-- API Key Setup --")
                print("Some keys are not set. Press Enter to skip optional keys.\n")
                missing_any = True
            entered = getpass.getpass(f"{'[required] ' if required else '[optional] '}{desc}\n  {var}: ").strip()
            if entered:
                os.environ[var] = entered
            elif required and require_openai:
                raise RuntimeError(
                    f"{var} is required but was not provided. "
                    "Set it in configs/env.sh or as an environment variable."
                )
