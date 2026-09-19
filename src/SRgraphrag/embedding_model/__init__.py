from importlib import import_module

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_BACKENDS = {
    "EmbeddingConfig": ".base",
    "BaseEmbeddingModel": ".base",
    "ContrieverModel": ".Contriever",
    "GritLMEmbeddingModel": ".GritLM",
    "NVEmbedV2EmbeddingModel": ".NVEmbedV2",
    "OpenAIEmbeddingModel": ".OpenAI",
    "CohereEmbeddingModel": ".Cohere",
    "TransformersEmbeddingModel": ".Transformers",
    "VLLMEmbeddingModel": ".VLLM",
}
__all__ = list(_BACKENDS) + ["_get_embedding_model_class"]


def __getattr__(name):
    """Import an embedding implementation only when it is requested."""
    if name not in _BACKENDS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_BACKENDS[name], __name__), name)
    globals()[name] = value
    return value


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return __getattr__("GritLMEmbeddingModel")
    elif "NV-Embed-v2" in embedding_model_name:
        return __getattr__("NVEmbedV2EmbeddingModel")
    elif "contriever" in embedding_model_name:
        return __getattr__("ContrieverModel")
    elif "text-embedding" in embedding_model_name:
        return __getattr__("OpenAIEmbeddingModel")
    elif "cohere" in embedding_model_name:
        return __getattr__("CohereEmbeddingModel")
    elif embedding_model_name.startswith("Transformers/"):
        return __getattr__("TransformersEmbeddingModel")
    elif embedding_model_name.startswith("VLLM/"):
        return __getattr__("VLLMEmbeddingModel")
    raise ValueError(f"Unknown embedding model name: {embedding_model_name}")
