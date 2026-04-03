"""Forge engine adapters for inference and training backends."""


def __getattr__(name: str):
    if name == "MonarchVLLMEngine":
        from forge.engine.monarch_inf import MonarchVLLMEngine

        return MonarchVLLMEngine
    raise AttributeError(f"module 'forge.engine' has no attribute {name!r}")


__all__ = ["MonarchVLLMEngine"]
