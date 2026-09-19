import os
from importlib import import_module

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

logger = get_logger(__name__)

_BACKENDS = {
    "CacheOpenAI": ".openai_gpt",
    "BaseLLM": ".base",
    "BedrockLLM": ".bedrock_llm",
    "TransformersLLM": ".transformers_llm",
}
__all__ = list(_BACKENDS) + ["_get_llm_class"]


def __getattr__(name):
    """Preserve package imports while loading only the requested backend."""
    if name not in _BACKENDS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_BACKENDS[name], __name__), name)
    globals()[name] = value
    return value


def _get_llm_class(config: BaseConfig):
    # DeepSeek → OpenAI
    if os.getenv("OPENAI_API_KEY") is None:
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key is not None:
            os.environ["OPENAI_API_KEY"] = deepseek_key

    if config.llm_base_url is not None and \
       'localhost' in config.llm_base_url and \
       os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        return __getattr__("BedrockLLM")(config)

    if config.llm_name.startswith('Transformers/'):
        return __getattr__("TransformersLLM")(config)

    return __getattr__("CacheOpenAI").from_experiment_config(config)
    
