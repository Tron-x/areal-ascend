"""Forge Generator actor — vLLM inference on Monarch.

Import directly from submodules to avoid deadlocks::

    from forge.actors.generator.generator import Generator
    from forge.actors.generator.worker import WorkerRegistry
"""


def __getattr__(name: str):
    _lazy_map = {
        "Generator": "forge.actors.generator.generator",
        "WorkerRegistry": "forge.actors.generator.worker",
        "WorkerWrapper": "forge.actors.generator.worker",
    }
    if name in _lazy_map:
        import importlib

        mod = importlib.import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'forge.actors.generator' has no attribute {name!r}")


__all__ = ["Generator", "WorkerRegistry", "WorkerWrapper"]
