"""MonarchForgeActor: Forge actor backed by Monarch ProcMesh.

Bridges the abstract ``ForgeActor`` ABC with Monarch's ``Actor`` class
and ProcMesh-based process management.
"""

from __future__ import annotations

from typing import Any

from monarch.actor import Actor, endpoint

from forge.core.actor import ActorConfig, ForgeActor
from forge.core.service import Service


class MonarchForgeActor(Actor, ForgeActor):
    """ForgeActor implementation backed by Monarch.

    Subclasses declare resource requirements as class attributes::

        class MyActor(MonarchForgeActor):
            procs = 4
            with_gpus = True

            @endpoint
            async def my_method(self, data):
                ...

    Spawning::

        actor = await MyActor.options(procs=8).as_actor(arg1, arg2)
        service = await MyActor.options(num_replicas=3).as_service()
    """

    @classmethod
    async def _spawn_actor(cls, config: ActorConfig, *args: Any, **kwargs: Any) -> Any:
        """Spawn a single actor on a Monarch ProcMesh."""
        from forge.adapters.monarch.provisioner import get_proc_mesh

        name = cls.__name__.lower()
        mesh = get_proc_mesh(
            name=name,
            procs=config.procs,
            with_gpus=config.with_gpus,
            bootstrap=config.bootstrap,
        )
        return mesh.spawn(name, cls, *args, **kwargs)

    @classmethod
    async def _spawn_service(
        cls, config: ActorConfig, *args: Any, **kwargs: Any
    ) -> Any:
        """Spawn multiple replicas as a Forge Service."""
        from forge.adapters.monarch.provisioner import get_proc_mesh

        name = cls.__name__.lower()
        actor_refs = []
        for i in range(config.num_replicas):
            mesh = get_proc_mesh(
                name=f"{name}_rep_{i}",
                procs=config.procs,
                with_gpus=config.with_gpus,
                bootstrap=config.bootstrap,
            )
            ref = mesh.spawn(f"{name}_{i}", cls, *args, **kwargs)
            actor_refs.append(ref)

        router = config.extra.get("router", "least_loaded")
        return Service(actor_refs, router=router)

    async def setup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @endpoint
    def health(self) -> str:
        """Health check endpoint used by Service replica monitoring."""
        return "ok"
