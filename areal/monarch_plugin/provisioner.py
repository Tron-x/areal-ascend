"""
TorchForge-inspired declarative resource provisioner.
Replaces explicit topology calculations by providing a get_proc_mesh() primitive
that interprets `procs` and `with_gpus` declarations.
"""

from collections.abc import Callable

from monarch._src.actor.host_mesh import this_host

from areal.monarch_plugin.bootstraps import make_cpu_bootstrap


def get_proc_mesh(
    name: str,
    procs: int,
    with_gpus: bool,
    bootstrap: Callable | None = None,
):
    """Create a ProcMesh on the local host with the requested resources.

    Parameters
    ----------
    name : str
        Unique name for the ProcMesh.
    procs : int
        Number of processes to spawn.
    with_gpus : bool
        Whether the processes need GPU (NPU) devices.
    bootstrap : Callable | None
        Optional bootstrap function run in each spawned process before
        the actor is constructed. For GPU actors this *must* set device
        visibility (e.g. ASCEND_RT_VISIBLE_DEVICES). For CPU actors a
        default no-op bootstrap is used when *None*.
    """
    host = this_host()

    per_host = {}
    if with_gpus:
        per_host["npu"] = procs
    else:
        per_host["cpu"] = procs

    if bootstrap is None and not with_gpus:
        bootstrap = make_cpu_bootstrap()

    proc_mesh = host.spawn_procs(
        per_host=per_host,
        bootstrap=bootstrap,
        name=name,
    )
    return proc_mesh
