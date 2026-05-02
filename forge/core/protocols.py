"""Engine protocols -- pluggable contracts for training, inference, and reward.

Three protocol families, each tied to a path in the dual-path architecture
(see ``.cursor/rules/framework-first-principles.mdc``):

1. **Pure Path protocols**: ``TrainEngine`` (per-step controllable trainers
   like Titan / Megatron / FSDP2), ``InferenceEngine`` (in-process generators
   like AReaL ``Generator``).  Used when Forge owns the algorithm and
   composes pure backends.

2. **Adapter Path protocols**: ``SPMDTrainerProtocol`` (run-blocking trainer
   actors that wrap a whole RL framework like ms-swift / LF / TRL),
   ``InferenceServerProtocol`` (out-of-process FastAPI rollout servers like
   ms-swift's ``SwiftRolloutDeploy`` / future SGLang server).  Used when a
   third-party framework owns the algorithm and Forge only orchestrates.

3. **Cross-path protocols**: ``RewardFn`` / ``RewardModelEngine`` /
   ``BatchAdapter`` / ``AgentLogic`` -- common components both paths use.

4. **Legacy backend protocols**: ``TrainBackend``, ``InferenceBridge``,
   ``RewardBackend``, ``DataProvider`` -- retained for backward compatibility
   with ``forge/engines/areal/`` (which architecturally is an Adapter Path
   member used as a functional-correctness oracle for the Pure Path; do not
   model new adapters on these legacy shapes).

All protocols use ``typing.Protocol`` with ``@runtime_checkable``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from forge.core.weight_sync import WeightsSpec

# ======================================================================
# New clean protocols (framework-agnostic)
# ======================================================================


@runtime_checkable
class TrainEngine(Protocol):
    """**Pure Path** -- per-step controllable training engine.

    Implementations wrap a *pure* training framework (TorchTitan, Megatron,
    FSDP2) and expose a uniform per-step interface that the Forge
    orchestration layer (``TrainerActor``, ``apps/grpo_titan.py``, etc.)
    can drive while owning the algorithm itself.

    Adapter Path frameworks (ms-swift / LF / TRL / AReaL-as-adapter) which
    own their own training loop should implement
    :class:`SPMDTrainerProtocol` instead -- forcing them through this
    per-step protocol would require fork-level changes to their internals.

    Key design decisions:

    - ``train_step(batch, step)`` receives an **externally-provided** batch,
      decoupling data production from training.
    - ``get_weights_spec()`` + ``state_dict_for_sync()`` separate weight
      *description* from weight *data*, letting ``WeightSyncStrategy``
      decide how to transfer weights.
    - Weight pushing and rollout are **not** part of this protocol --
      they belong to the orchestration layer.

    Lifecycle::

        engine = FSDPTrainEngine(config)
        meta = engine.initialize()        # load model, optimizer, scheduler
        for step in range(meta["max_steps"]):
            result = engine.train_step(batch, step)
            # Orchestrator handles weight sync separately
        engine.shutdown()
    """

    def initialize(self) -> dict:
        """Load model, optimizer, and scheduler.

        Returns:
            Metadata dict with at least:
            ``{"max_steps": int, "start_step": int, "model_path": str}``.
        """
        ...

    def train_step(self, batch: dict, step: int) -> dict:
        """Run one optimization step on an externally-provided batch.

        Args:
            batch: Engine-specific tensor dict produced by a ``BatchAdapter``.
            step: Global training step number.

        Returns:
            Result dict with at least ``{"loss": float}``.
            May also include ``"grad_norm"``, ``"lr"``, etc.
        """
        ...

    def get_weights_spec(self) -> WeightsSpec:
        """Describe the current model parameters for weight sync.

        Returns a ``WeightsSpec`` containing parameter names, shapes,
        dtypes, and optional sharding metadata -- everything a
        ``WeightSyncStrategy`` needs to set up a transfer channel.
        """
        ...

    def state_dict_for_sync(self) -> dict:
        """Return a state dict (or shard) for weight sync to inference.

        The returned dict maps parameter names to tensors. For sharded
        models (FSDP, Megatron), this may return only the local shard;
        the ``WeightSyncStrategy`` handles reassembly.
        """
        ...

    def get_metadata(self) -> dict:
        """Return engine metadata.

        Returns:
            Dict with ``"max_steps"``, ``"current_step"``,
            ``"model_path"``, and any engine-specific info.
        """
        ...

    def shutdown(self) -> None:
        """Release resources (model, optimizer, CUDA memory)."""
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """**Pure Path** -- in-process generator (Forge ``Generator``-style).

    Implementations wrap vLLM / SGLang / TGI **as a Python object** living
    inside a ``ForgeActor`` (e.g. AReaL's :class:`forge.actors.generator.Generator`).
    This is the protocol when Forge drives generation directly via Python
    method calls, not over HTTP.

    For out-of-process FastAPI rollout servers (the Adapter Path default),
    use :class:`InferenceServerProtocol` instead -- see that class for the
    distinction.
    """

    async def generate(self, prompt: str, **kwargs: Any) -> dict:
        """Generate text for a prompt. Returns completion dict."""
        ...

    def update_weights(self, version: int) -> None:
        """Pull updated weights from the training engine."""
        ...

    def get_version(self) -> int:
        """Current policy version."""
        ...


# ======================================================================
# Adapter Path protocols
# ======================================================================
# These protocols capture the "actor shape" for backends that wrap a whole
# third-party RL framework.  The framework owns its own training loop /
# rollout / algorithm; Forge just orchestrates lifecycle + cross-mesh
# weight sync + reward/dataset adapters.
#
# Why separate from TrainEngine / InferenceEngine: a coarse-grained run()
# endpoint that blocks until training is done has fundamentally different
# semantics than a per-step train_step() call.  Forcing them into the same
# interface either loses the algorithm-control granularity (Pure Path) or
# requires fork-level changes to every adapted framework (Adapter Path).


@runtime_checkable
class SPMDTrainerProtocol(Protocol):
    """**Adapter Path** -- coarse-grained SPMD trainer actor.

    The "actor shape" for trainers that wrap a third-party framework's
    own ``train.py`` (or equivalent).  Subclasses
    :class:`monarch._src.spmd.actor.SPMDActor` to inherit ``RANK`` /
    ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` / ``MASTER_PORT``
    env wiring + ``setup_env(master_addr, master_port)`` endpoint, then
    adds:

    * ``run(config, **kwargs)`` -- block until the framework's training
      loop completes, return a result dict.
    * ``teardown(...)`` -- recursive process-tree cleanup so dataloader
      workers / temporary subprocs don't leak across runs.

    Conformant implementations (current):

    * :class:`forge.actors.titan_trainer.TitanTrainerActor` (TorchTitan)
      *(missing teardown -- TODO)*
    * :class:`forge.actors.llamafactory_trainer.LlamaFactoryTrainerActor` (LF)
      *(missing teardown -- TODO)*
    * :class:`forge.actors.msswift_trainer.MsSwiftTrainerActor` (ms-swift)

    The ``config`` arg's shape is framework-specific by design (TT takes a
    toml path + overrides list, LF takes a dict-of-dicts, ms-swift takes a
    flat CLI dict).  The driver layer knows which adapter it's calling and
    builds the right shape from the YAML.  We deliberately do **not** force
    a uniform config schema -- that would either water down each
    framework's expressiveness or require parser hacks at the actor.

    Lifecycle::

        actor = mesh.spawn("trainer", MyTrainerActor)
        await actor.setup_env.call(master_addr, master_port)  # SPMDActor
        results = await actor.run.call(config=..., **kwargs)
        # ... drive other paths if needed (rollout server etc.) ...
        await actor.teardown.call()
    """

    def setup_env(self, master_addr: str, master_port: int) -> dict:
        """Set RANK/LOCAL_RANK/.../MASTER_ADDR/MASTER_PORT.

        Inherited from :class:`monarch._src.spmd.actor.SPMDActor`; listed
        here so the protocol is self-contained for type-checking purposes.
        """
        ...

    def run(self, config: Any, **kwargs: Any) -> dict:
        """Block until the wrapped framework's training loop completes.

        Args:
            config: Framework-specific training config.  Shape varies
                per adapter (toml path / dict / argv list).  See each
                implementation's docstring for its expected shape.
            **kwargs: Cross-cutting overrides commonly shared across
                adapters: ``cwd``, ``ws_master_addr`` / ``ws_master_port``
                (cross-mesh weight-sync TCPStore), ``extra_env``,
                ``use_modelscope``.

        Returns:
            ``{"rank": int, "host": str, "elapsed_s": float, "ok": bool, ...}``
            -- per-rank result.  ``ok=False`` means this rank's training
            crashed; the driver should still call ``teardown`` to clean up.
        """
        ...

    def teardown(
        self,
        *,
        term_grace_s: float = 5.0,
        kill_grace_s: float = 3.0,
    ) -> dict:
        """Recursive process-tree cleanup + DDP group destruction.

        Always best-effort, never raises.  Should:

        1. Destroy ``torch.distributed`` process group if still initialized.
        2. Recursively SIGTERM/SIGKILL every descendant of this actor
           (dataloader workers, framework-spawned subprocs).  Reuse
           :func:`forge.utils.process_tree.kill_descendants`.

        Returns ``{"rank", "host", "dist_destroyed", "terminated",
        "killed", "survivors"}``.  ``survivors > 0`` is the actionable
        failure (stuck NPU/GPU driver call -- needs host reboot).
        """
        ...


@runtime_checkable
class InferenceServerProtocol(Protocol):
    """**Adapter Path / cross-path** -- out-of-process FastAPI rollout server.

    The "actor shape" for inference backends that run as a long-lived
    HTTP server hit by trainers over the network.  This is the default
    inference shape for both Pure Path (Titan + vLLM-server) and Adapter
    Path (ms-swift's ``SwiftRolloutDeploy``, future SGLang RLHF server),
    because cross-mesh weight sync over HCCL/NCCL needs the inference
    workers to be addressable by the trainer's :class:`WeightSyncClient`.

    Distinction from :class:`InferenceEngine`:

    * ``InferenceEngine`` = in-process Python object inside a Forge actor
      (e.g. :class:`forge.actors.generator.Generator`).  Trainer drives
      generation via Python method calls.
    * ``InferenceServerProtocol`` = separate process tree exposing
      FastAPI ``/generate`` + ``/areal_*`` weight-sync routes.  Trainer
      hits it over HTTP (vLLM client / requests).

    Conformant implementations (current):

    * :class:`forge.actors.msswift_rollout.MsSwiftRolloutActor`

    Future:

    * SGLang-RLHF rollout actor (Pure Path inference for grpo_titan.py)
    * Plain vLLM-only rollout actor (Pure Path lighter-weight than ms-swift)

    Lifecycle::

        actor = mesh.spawn("rollout", MyRolloutActor)
        info = await actor.host_info.call_one()      # get NIC IP
        await actor.start.call_one(args=..., port=8000)
        await actor.wait_ready.call_one(timeout_s=300)
        # ... trainer drives weight sync via http://info["ip"]:8000 ...
        await actor.teardown.call_one()
    """

    def host_info(self) -> dict:
        """Return ``{"ip": str, "hostname": str}`` for this actor's host.

        Cheap (no model load).  Driver calls this BEFORE ``start`` so it
        knows which URL to hand to the trainer's ``vllm_server_base_url``.
        """
        ...

    def start(self, *, port: int = 8000, **kwargs: Any) -> dict:
        """Boot the FastAPI server in a daemon thread, return immediately.

        Heavy imports (vLLM, torch_npu, framework-specific) happen here
        so the module stays importable on CPU-only orchestration hosts.
        Apply any monkey-patches BEFORE constructing the server (e.g.
        :func:`forge.engines.msswift.glue.install_server_patches`).

        Returns ``{"host": str, "port": int, "world_size": int,
        "pid": int}``.  ``world_size`` is what the trainer's weight-sync
        client should expect on the inference side.
        """
        ...

    def wait_ready(
        self,
        *,
        timeout_s: float = 300.0,
        interval_s: float = 2.0,
    ) -> dict:
        """Poll ``/health/`` until 200 OK or timeout.

        Returns ``{"ready": True, "elapsed_s": float}``.  Raises
        ``TimeoutError`` if not ready in budget; raises ``RuntimeError``
        if the server thread died during startup (more useful than
        polling forever).
        """
        ...

    def teardown(
        self,
        *,
        uvicorn_timeout_s: float = 10.0,
        term_grace_s: float = 8.0,
        kill_grace_s: float = 5.0,
    ) -> dict:
        """Best-effort uvicorn + recursive child-tree cleanup.

        Cleanup order:

        1. Flip ``uvicorn.Server.should_exit``, join the FastAPI thread.
        2. Recursively SIGTERM/SIGKILL every descendant of this actor
           (vLLM ``EngineCore`` + ``Worker_TP*`` grandchildren that hold
           HCCL/NCCL ports).  Reuse
           :func:`forge.utils.process_tree.kill_descendants`.
        3. Reap direct-child zombies via ``waitpid``.

        Returns ``{"uvicorn_joined", "host", "port", "terminated",
        "killed", "survivors", ...}``.
        """
        ...


@runtime_checkable
class RewardFn(Protocol):
    """Pluggable reward function.

    Can be a simple callable or a full model-based reward.
    """

    def __call__(
        self, prompt: str, response: str, target: Any = None, **kwargs: Any
    ) -> float:
        """Compute scalar reward for a prompt-response pair."""
        ...


@runtime_checkable
class RewardModelEngine(Protocol):
    """Pluggable neural reward model for scoring (prompt, response) pairs.

    Unlike ``RewardFn`` (a stateless callable), a ``RewardModelEngine``
    manages a loaded model with GPU memory, batched inference, and
    optional weight updates.

    Lifecycle::

        engine = HFRewardModelEngine(model_path="...")
        engine.load()                          # load checkpoint to GPU
        scores = engine.score_batch([...])     # batched inference
        engine.shutdown()                      # free GPU memory

    Implementations live in ``forge/engines/<backend>/reward_model.py``.
    """

    def load(self) -> dict:
        """Load model checkpoint and move to device. Returns metadata."""
        ...

    def score(self, prompt: str, response: str) -> float:
        """Score a single (prompt, response) pair. Returns scalar reward."""
        ...

    def score_batch(self, items: list[dict[str, str]]) -> list[float]:
        """Score a batch of (prompt, response) pairs.

        Each item dict must contain ``"prompt"`` and ``"response"`` keys.
        Returns list of scalar rewards.
        """
        ...

    def shutdown(self) -> None:
        """Free model and GPU memory."""
        ...


@runtime_checkable
class BatchAdapter(Protocol):
    """Convert framework-agnostic ``Episode`` objects to engine-specific batches.

    Each training engine (AReaL, TorchTitan, Slime, ...) expects a different
    tensor layout.  A ``BatchAdapter`` bridges the gap so that actors
    produce only ``Episode`` objects and the orchestrator converts them
    at the last moment before sending to the ``TrainerActor``.

    Implementations live in ``forge/engines/<backend>/batch_adapter.py``.
    """

    def adapt(self, episodes: list) -> dict:
        """Convert a list of Episodes to the engine's expected batch dict.

        Args:
            episodes: Framework-agnostic ``Episode`` objects from the
                rollout pipeline.

        Returns:
            A dict of lists/tensors that the training engine can consume
            directly (e.g. ``input_ids``, ``attention_mask``, etc.).
        """
        ...

    def required_fields(self) -> list[str]:
        """Return the list of Episode fields this adapter needs.

        Used for validation: the orchestrator can warn early if an
        Episode is missing a required field.
        """
        ...


@runtime_checkable
class AgentLogic(Protocol):
    """Pluggable agent strategy -- pure logic, no infrastructure awareness.

    Implementations define *what* the agent does (extract tools, decide
    when to stop, format feedback) while ``AgentActor`` handles *how*
    (call Generator via ModelProxy, execute tools, collect training data).

    Built-in implementations:
        - ``forge.agents.react.SimpleReActAgent``   (code-execution ReAct loop)
    """

    def process_response(
        self,
        response: str,
        messages: list[dict[str, str]],
    ) -> Any:
        """Analyse a generation response and decide what to do.

        Returns an ``AgentAction`` describing extracted tool calls and
        whether the episode is considered finished.
        """
        ...

    def should_continue(self, turn: int, reward: float) -> bool:
        """Decide whether to proceed to the next turn."""
        ...

    def format_feedback(
        self,
        action: Any,
        tool_results: list,
        reward: float,
    ) -> str:
        """Build the user-feedback message appended before the next turn."""
        ...


# ======================================================================
# Legacy backend protocols (backward compat for engines/areal)
# ======================================================================


@runtime_checkable
class TrainBackend(Protocol):
    """Legacy training backend with AReaL-style rollout+train interface.

    Kept for backward compatibility. New engines should implement
    ``TrainEngine`` instead.
    """

    def initialize(self) -> dict: ...
    def train_step(self, global_step: int) -> dict: ...
    def do_rollout(self, global_step: int) -> dict: ...
    def train_on_batch(self, batch_data: dict, global_step: int) -> dict: ...

    def train_on_buffered_batch(
        self, batch_data: dict, global_step: int, skip_weight_sync: bool = False
    ) -> dict: ...

    def sync_weights(self, global_step: int) -> dict: ...
    def get_train_metadata(self) -> dict: ...
    def shutdown(self) -> None: ...


@runtime_checkable
class InferenceBridge(Protocol):
    """Legacy inference bridge (AReaL-style, used inside TrainBackend)."""

    def initialize(self, **kwargs: Any) -> None: ...
    def destroy(self) -> None: ...
    async def agenerate(self, request: Any) -> Any: ...
    def set_version(self, version: int) -> None: ...
    def get_version(self) -> int: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


@runtime_checkable
class RewardBackend(Protocol):
    """Legacy reward backend (AReaL-style, used by RewardActor)."""

    def setup(self, reward_fn_path: str = "") -> dict: ...

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list | None = None,
        completion_ids: list | None = None,
        task_data: dict | None = None,
    ) -> float: ...

    def compute_rewards_batch(self, items: list[dict]) -> list[float]: ...
    def get_stats(self) -> dict: ...


@runtime_checkable
class DataProvider(Protocol):
    """Data provider for rollout (decoupled from training pipeline)."""

    def get_batch(self) -> list[dict]: ...
    def reset(self) -> None: ...
    def __len__(self) -> int: ...
