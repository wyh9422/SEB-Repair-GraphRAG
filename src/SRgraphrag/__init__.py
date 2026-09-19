"""Model backends are loaded only when the main SRgraphrag class is requested."""

__all__ = ["SRgraphrag"]


def __getattr__(name):
    if name == "SRgraphrag":
        from .SRgraphrag import SRgraphrag
        globals()[name] = SRgraphrag
        return SRgraphrag
    raise AttributeError(name)
