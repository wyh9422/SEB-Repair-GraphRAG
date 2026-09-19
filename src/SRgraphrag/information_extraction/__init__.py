"""Lazy OpenIE exports; offline runtimes are loaded by their selected mode."""

from importlib import import_module

__all__ = ["OpenIE"]


def __getattr__(name):
    if name != "OpenIE":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(".openie_openai", __name__).OpenIE
    globals()[name] = value
    return value
