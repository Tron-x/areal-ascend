"""
Pure TorchForge declarative actor abstractions for AReaL Monarch Plugin.
Replaces the complex DAG ActorRegistry and explicit topology specs.
"""

from typing import TypeVar, Type, Any
from collections.abc import Callable
from monarch.actor import Actor
from areal.monarch_plugin.provisioner import get_proc_mesh

T = TypeVar("T", bound="AReaLMonarchActor")

class _ActorConfigContext:
    def __init__(self, cls: Type['AReaLMonarchActor'], **kwargs):
        self.cls = cls
        self.kwargs = kwargs
        
    async def as_service(self, *args, **kwargs) -> Any:
        return await self.cls._as_service_impl(self.kwargs, *args, **kwargs)
        
    async def as_actor(self, *args, **kwargs) -> Any:
        return await self.cls._as_actor_impl(self.kwargs, *args, **kwargs)


class AReaLMonarchActor(Actor):
    """
    TorchForge-style declarative actor.
    Declare processes and GPUs at the class level instead of writing complex topology code.
    """
    
    procs: int = 1
    with_gpus: bool = False
    num_replicas: int = 1
    
    _extra_config: dict[str, Any] = {}

    from monarch.actor import endpoint

    @classmethod
    def options(
        cls: Type[T],
        procs: int | None = None,
        with_gpus: bool | None = None,
        num_replicas: int | None = None,
        bootstrap: Callable | None = None,
        **kwargs
    ) -> _ActorConfigContext:
        """
        Dynamically override declaring resources, mirroring ForgeActor.options().
        """
        cfg = {}
        if procs is not None:
            cfg['procs'] = procs
        if with_gpus is not None:
            cfg['with_gpus'] = with_gpus
        if num_replicas is not None:
            cfg['num_replicas'] = num_replicas
        if bootstrap is not None:
            cfg['bootstrap'] = bootstrap
            
        cfg.update(kwargs)
        return _ActorConfigContext(cls, **cfg)

    @classmethod
    async def _as_service_impl(cls, overrides: dict, *args, **kwargs) -> Any:
        """
        Spawns `num_replicas` separate Actor instances inside ProcMeshes,
        then binds them into a Service orchestrator.
        """
        procs = overrides.get('procs', cls.procs)
        with_gpus = overrides.get('with_gpus', cls.with_gpus)
        num_replicas = overrides.get('num_replicas', cls.num_replicas)
        bootstrap = overrides.get('bootstrap', None)

        name = cls.__name__.lower()

        meshes = [
            get_proc_mesh(f"{name}_rep_{i}", procs, with_gpus, bootstrap=bootstrap)
            for i in range(num_replicas)
        ]

        actor_refs = [
            mesh.spawn(f"{name}_{i}", cls, *args, **kwargs)
            for i, mesh in enumerate(meshes)
        ]

        from areal.monarch_plugin.service import Service
        return Service(actor_refs, cls)


    @classmethod
    async def _as_actor_impl(cls, overrides: dict, *args, **kwargs) -> Any:
        """
        Spawns a single Actor instance for things like the master TrainerActor
        which doesn't need load-balanced Service routing.
        """
        procs = overrides.get('procs', cls.procs)
        with_gpus = overrides.get('with_gpus', cls.with_gpus)
        bootstrap = overrides.get('bootstrap', None)

        name = cls.__name__.lower()
        mesh = get_proc_mesh(name, procs, with_gpus, bootstrap=bootstrap)

        actor_ref = mesh.spawn(name, cls, *args, **kwargs)
        return actor_ref
