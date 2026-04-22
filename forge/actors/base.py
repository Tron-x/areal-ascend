"""ForgeActor -- TorchForge-style declarative actor base class.

Provides the ``options() -> as_service() / as_actor()`` pattern for
declarative resource configuration and actor lifecycle management.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import TYPE_CHECKING, Any, TypeVar

from monarch.actor import Actor, current_rank, current_size, endpoint

if TYPE_CHECKING:
    from monarch._src.actor.actor_mesh import ActorMesh

    from forge.service.interface import ServiceInterface

from forge.provisioner import (
    get_proc_mesh,
    register_actor,
    register_service,
    stop_proc_mesh,
)
from forge.types import ProcessConfig, ServiceConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ForgeActor")


class ForgeActor(Actor):
    """Base class for all AReaL Monarch actors with declarative resource config.

    Subclasses declare their resource needs as class attributes::

        class MyTrainer(ForgeActor):
            procs = 4
            with_gpus = True

    Then spawn with::

        trainer = await MyTrainer.options(procs=4, with_gpus=True).as_actor(...)
        service = await MyGenerator.options(num_replicas=2).as_service(...)

    The ``options()`` classmethod returns a configured subclass, preserving
    all classmethods for ``as_service`` / ``as_actor``.
    """

    procs: int = 1
    gpus_per_proc: int = 1
    hosts: int | None = None
    with_gpus: bool = False
    num_replicas: int = 1
    mesh_name: str | None = None
    _extra_config: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_rank"):
            self._rank = current_rank().rank
        if not hasattr(self, "_size"):
            self._size = math.prod(current_size().values())

        BLUE = "\033[34m"
        RESET = "\033[0m"
        formatter = logging.Formatter(
            f"{BLUE}[{self.__class__.__name__}-{self._rank}/{self._size}] "
            f"%(asctime)s %(levelname)s{RESET} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)

        self._proc_mesh = None
        self.logger.root.setLevel(logging.INFO)
        self.logger.root.addHandler(stdout_handler)
        super().__init__(*args, **kwargs)

    @classmethod
    def options(
        cls: type[T],
        *,
        procs: int = 1,
        gpus_per_proc: int = 1,
        hosts: int | None = None,
        with_gpus: bool = False,
        num_replicas: int = 1,
        mesh_name: str | None = None,
        **kwargs,
    ) -> type[T]:
        """Return a configured subclass with the given resource attributes.

        Each call creates a new subclass so multiple configurations can
        coexist without interfering with each other::

            gen_cfg = Generator.options(procs=1, with_gpus=True, num_replicas=2)
            service = await gen_cfg.as_service(model="Qwen/Qwen2.5-1.5B")

        ``gpus_per_proc`` controls how many accelerator devices each
        spawned proc should see (via
        ``CUDA_VISIBLE_DEVICES`` / ``ASCEND_RT_VISIBLE_DEVICES``). The
        default 1 matches the historical one-device-per-proc layout.
        Use >1 when a single proc drives internal TP workers
        (e.g. a vLLM engine with ``tensor_parallel_size=N`` should be
        launched as ``procs=1, gpus_per_proc=N``).
        """
        attrs = {
            "procs": procs,
            "gpus_per_proc": gpus_per_proc,
            "hosts": hosts,
            "with_gpus": with_gpus,
            "num_replicas": num_replicas,
            "mesh_name": mesh_name,
            "_extra_config": kwargs,
        }
        return type(cls.__name__, (cls,), attrs)

    @classmethod
    async def as_service(cls: type[T], *actor_args, **actor_kwargs) -> ServiceInterface:
        """Spawn this actor as a replicated Service with load balancing.

        Uses configuration from ``options()`` to build a ServiceConfig,
        then creates Service -> ServiceInterface.
        """
        from forge.service import Service, ServiceInterface

        cfg_kwargs = {
            "procs": cls.procs,
            "gpus_per_proc": cls.gpus_per_proc,
            "hosts": cls.hosts,
            "with_gpus": cls.with_gpus,
            "num_replicas": cls.num_replicas,
            "mesh_name": cls.mesh_name,
            **cls._extra_config,
        }
        cfg = ServiceConfig(**cfg_kwargs)

        logger.info(f"Spawning service {cls.__name__}")
        service = Service(cfg, cls, actor_args, actor_kwargs)
        await service.__initialize__()
        service_interface = ServiceInterface(service, cls)
        await register_service(service_interface)
        return service_interface

    @endpoint
    def setup(self):
        """Heavy-weight initialization hook.

        Best practice: pass data via constructor, do expensive work in setup().
        This ensures initialization failures propagate back to the caller.

        Subclasses can override as sync or async depending on their endpoint style.
        """
        pass

    @classmethod
    async def launch(cls, *args, **kwargs) -> ActorMesh:
        """Provision a ProcMesh and deploy the actor.

        Override this in subclasses that need custom launch logic
        (e.g., Generator spawns WorkerRegistry + GPU procs separately).
        """
        cfg = ProcessConfig(
            procs=cls.procs,
            gpus_per_proc=cls.gpus_per_proc,
            hosts=cls.hosts,
            with_gpus=cls.with_gpus,
            mesh_name=cls.mesh_name,
        )

        proc_mesh = await get_proc_mesh(process_config=cfg)

        actor_name = kwargs.pop("name", cls.__name__)
        actor = proc_mesh.spawn(actor_name, cls, *args, **kwargs)
        actor._proc_mesh = proc_mesh
        await actor.setup.call()
        return actor

    @classmethod
    async def as_actor(cls: type[T], *args, **actor_kwargs) -> T:
        """Spawn a single actor using configuration from ``options()``."""
        logger.info(f"Spawning actor {cls.__name__}")
        actor = await cls.launch(*args, **actor_kwargs)
        await register_actor(actor)
        return actor

    @classmethod
    async def shutdown(cls, actor: ForgeActor):
        """Shut down an actor by stopping its ProcMesh."""
        if actor._proc_mesh is None:
            raise AssertionError("Called shutdown on a replica with no proc_mesh.")
        await stop_proc_mesh(actor._proc_mesh)
