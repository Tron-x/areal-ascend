"""Universal data types for the Forge training pipeline.

These types are the common language between all components (actors, engines,
RL algorithms, orchestration).  They have ZERO framework dependencies --
no AReaL, no Slime, no TorchTitan.  Only stdlib + typing.

Inspired by TorchForge's ``Episode``, ``Completion``, and ``TrainBatch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Completion:
    """A model-generated completion for a given prompt.

    Attributes:
        text: Decoded generated text.
        prompt: The original prompt string.
        prompt_ids: Encoded prompt token IDs.
        token_ids: Encoded generated token IDs.
        logprobs: Per-token log-probabilities of the generated tokens.
        stop_reason: Why generation stopped (``"stop"``, ``"length"``, etc.).
        generator_version: Policy version that produced this completion.
        metadata: Extra info (timing, model name, etc.).
    """

    text: str = ""
    prompt: str = ""
    prompt_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str | None = None
    generator_version: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A single RL episode: prompt + completion + reward + training signals.

    This is the universal unit of data flowing through the pipeline::

        Generator -> Episode -> ReplayBuffer -> BatchAdapter -> Trainer

    ``Episode`` is framework-agnostic: actors produce it, the ``ReplayBuffer``
    stores it, and a per-engine ``BatchAdapter`` converts it to the format
    that the training engine expects.

    Attributes:
        episode_id: Unique identifier.
        prompt: Original prompt text.
        response: Generated response text.
        target: Ground truth answer (for reward computation).
        completion: Full generation metadata.
        reward: Scalar reward for this episode.
        reward_breakdown: Per-component reward scores.
        advantage: Computed advantage (GRPO/GAE).
        policy_version: Which policy version generated this.
        prompt_token_ids: Encoded prompt token IDs.
        token_ids: All token IDs (prompt + completion concatenated).
        generator_logprobs: Per-token logprobs from the generator.
        ref_logprobs: Per-token logprobs from the reference model.
        loss_mask: Binary mask indicating which tokens to train on.
        versions: Per-token policy version tags.
        metadata: Arbitrary extra data.
    """

    episode_id: str = ""
    prompt: str = ""
    response: str = ""
    target: Any = None
    completion: Completion | None = None
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    advantage: float | None = None
    policy_version: int = -1
    prompt_token_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    generator_logprobs: list[float] = field(default_factory=list)
    ref_logprobs: list[float] | None = None
    loss_mask: list[int] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def seq_len(self) -> int:
        """Total sequence length (prompt + completion tokens)."""
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for Monarch RPC / ReplayBuffer)."""
        d: dict[str, Any] = {
            "episode_id": self.episode_id,
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "policy_version": self.policy_version,
            "prompt_token_ids": self.prompt_token_ids,
            "token_ids": self.token_ids,
            "generator_logprobs": self.generator_logprobs,
            "loss_mask": self.loss_mask,
            "versions": self.versions,
        }
        if self.target is not None:
            d["target"] = self.target
        if self.ref_logprobs is not None:
            d["ref_logprobs"] = self.ref_logprobs
        if self.advantage is not None:
            d["advantage"] = self.advantage
        if self.reward_breakdown:
            d["reward_breakdown"] = self.reward_breakdown
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        """Reconstruct an Episode from a plain dict."""
        return cls(
            episode_id=d.get("episode_id", ""),
            prompt=d.get("prompt", ""),
            response=d.get("response", ""),
            target=d.get("target"),
            reward=d.get("reward", 0.0),
            policy_version=d.get("policy_version", -1),
            prompt_token_ids=d.get("prompt_token_ids", []),
            token_ids=d.get("token_ids", []),
            generator_logprobs=d.get("generator_logprobs", []),
            ref_logprobs=d.get("ref_logprobs"),
            loss_mask=d.get("loss_mask", []),
            versions=d.get("versions", []),
            advantage=d.get("advantage"),
            reward_breakdown=d.get("reward_breakdown", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass
class TrainBatch:
    """Universal training batch consumed by any TrainEngine.

    Separates model inputs from loss inputs, so the trainer can do::

        logits = model(**batch.model_inputs)
        loss = loss_fn(logits, **batch.loss_inputs)

    Attributes:
        model_inputs: Inputs for the forward pass (input_ids, attention_mask, ...).
        loss_inputs: Inputs for loss computation (advantages, ref_logprobs, ...).
        meta: Non-training metadata (for logging, checkpointing, etc.).
    """

    model_inputs: dict[str, Any] = field(default_factory=dict)
    loss_inputs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


Group = list[Episode]
"""A group of episodes for the same prompt (GRPO: G completions per prompt)."""


# ======================================================================
# Agent types (moved from core/agent.py for a flatter core/ structure)
# ======================================================================


@dataclass
class ToolCall:
    """A single tool invocation requested by the agent.

    Attributes:
        type: Tool identifier (e.g. ``"code_execution"``, ``"web_search"``).
        content: Payload for the tool (source code, query string, etc.).
        metadata: Optional extra data for the tool.
    """

    type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Outcome of executing a ``ToolCall``.

    Attributes:
        success: Whether the tool executed without errors.
        output: The tool's stdout / return value.
        error: Error message if ``success`` is False.
        tool_call: The original request that produced this result.
    """

    success: bool
    output: str = ""
    error: str = ""
    tool_call: ToolCall | None = None


@dataclass
class GenerationResult:
    """LLM generation output with RL training metadata.

    The ``text`` field is all an ``AgentLogic`` needs.  The remaining
    fields are collected by ``AgentActor`` for training data assembly
    and are opaque to agent logic implementations.

    Attributes:
        text: Generated text.
        token_ids: Output token IDs (for training).
        logprobs: Per-token log-probabilities (for PPO/GRPO advantage).
        version: Generator weight version at generation time.
        raw: Full upstream response dict (preserved for adapters).
    """

    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    version: int = -1
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Output of a single agent reasoning step.

    Attributes:
        response: The model's textual response for this turn.
        tool_calls: Tool invocations extracted from the response.
        done: If True, the agent considers the episode finished.
        metadata: Arbitrary data the agent logic wants to carry forward.
    """

    response: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Infrastructure types (merged from forge/types.py)
# ======================================================================


@dataclass
class ProcessConfig:
    """Configuration for allocating a Monarch ProcMesh.

    Fields:
        procs: number of Python processes to spawn on the mesh.
        gpus_per_proc: number of accelerator devices (e.g. NPUs) each
            proc should have visible. Default 1 matches the historical
            behavior where ``with_gpus=True`` hands one device per proc.
            Set to >1 when a single proc needs to drive tensor parallel
            workers internally (e.g. a vLLM engine running TP>1 inside
            one Python process). Total devices requested from the host's
            GpuManager is ``procs * gpus_per_proc``.
        with_gpus: whether to request accelerator isolation at all.
        hosts: remote host count; ``None`` means local.
        mesh_name: logical mesh name (used by the launcher to look up
            which physical host slice to land on).
    """

    procs: int = 1
    gpus_per_proc: int = 1
    with_gpus: bool = False
    hosts: int | None = None
    mesh_name: str | None = None


@dataclass
class ServiceConfig:
    """Configuration for a replicated Forge service."""

    procs: int = 1
    gpus_per_proc: int = 1
    num_replicas: int = 1
    with_gpus: bool = False
    hosts: int | None = None
    health_poll_rate: float = 0.2
    replica_max_concurrent_requests: int = 10
    return_first_rank_result: bool = True
    mesh_name: str | None = None

    def to_process_config(self) -> ProcessConfig:
        return ProcessConfig(
            procs=self.procs,
            gpus_per_proc=self.gpus_per_proc,
            with_gpus=self.with_gpus,
            hosts=self.hosts,
            mesh_name=self.mesh_name,
        )


class Launcher(Enum):
    LOCAL = "local"
    SLURM = "slurm"
    PREALLOCATED = "preallocated"
    BARE_METAL = "bare_metal"


@dataclass
class BcastBackendConfig:
    """Per-backend tuning for the ``collective_broadcast`` weight-sync backend.

    All fields have sane defaults; typical users don't need to touch this
    block at all.  Exposed mostly so operators can pin deterministic
    ports / vol indices when co-locating multiple forge jobs on the same
    cluster.
    """

    src_vol_idx: int = 0
    """Index of the storage volume that acts as the HCCL broadcast
    source.  Must be in ``[0, train_world)``.  Always 0 unless an
    operator has a specific reason to pick another vol."""

    master_port: int | None = None
    """TCP port used for the HCCL TCPStore rendezvous.  ``None`` = auto-
    allocate a free port at initialize time; set to a fixed port for
    deterministic deployments or firewall-restricted environments."""


@dataclass
class WeightSyncBlock:
    """Data-plane weight-sync configuration (nested under ``launcher``).

    Per the env-to-YAML mapping doc (`forge/docs/env_to_yaml_mapping.md`),
    this block is the authoritative source for weight-sync knobs: envs
    (``FORGE_WEIGHT_SYNC_BACKEND`` / ``FORGE_SHARD_PUBLISH`` / ``...``)
    remain available for one release window as overrides, then go away.

    Precedence: ``yaml > env > default``.
    """

    # ---- Method / backend -------------------------------------------
    method: str | None = None
    """``"nccl"`` (legacy) / ``"checkpoint"`` / ``"hixl"`` (legacy) /
    ``"torchstore"`` (recommended).  Default ``None`` = read
    ``FORGE_WEIGHT_SYNC`` env (itself defaults to ``"nccl"`` for backward
    compat).  The ``torchstore`` path routes through
    ``WeightSyncService`` + pluggable backends (see ``backend`` below)."""

    backend: str | None = None
    """Backend name when ``method=torchstore``.  Registered values:
    ``torchstore_multi_vol`` (default, TP=1), ``collective_broadcast``
    (TP>1), ``dedicated_ps`` (future), ``areal_xccl`` (fallback).
    Registration lives in
    ``forge.engines.weight_sync.backends.create_backend``."""

    # ---- Shared across multi-vol / collective backends --------------
    storage_mesh: str | None = None
    """Name of a ``launcher.meshes.*`` entry to host storage volumes.
    Default ``None`` = ``"trainer"`` (colocated with the training mesh,
    which is the legacy behavior).  Set to ``"generator"`` or a custom
    dedicated host for alternative topologies."""

    storage_npu_base: int | None = None
    """First NPU id used by storage volumes on the chosen host.
    ``None`` = auto (uses ``train_world_size`` as offset so storage
    doesn't collide with the trainer's NPUs on a colocated host)."""

    storage_spawn_mode: str = "driver"
    """``"driver"`` (default) — the grpo driver spawns the storage mesh
    and injects it into the backend.  ``"backend"`` = backend spawns its
    own storage mesh internally (legacy path; simpler but harder to
    compose with other meshes).  Most users should leave at default."""

    pool_mb: int | None = None
    """MonarchRDMA staging pool size per storage volume (MiB).  ``None`` =
    use backend default (currently 8192).  Lower this (to e.g. 4096) if
    NPU memory is tight on the storage host."""

    eager_d2h: bool | None = None
    """``True`` eagerly moves staged tensors to host memory; ``False``
    keeps them on NPU.  Default ``None`` = use torchstore's default,
    which is ``False`` (on-device staging gives better RDMA bandwidth).
    Only tune if you're debugging HiXL memory registration issues."""

    # ---- Trainer-side (shard-parallel publish) ----------------------
    shard_publish: bool | None = None
    """Enables 4-NIC parallel ``ts.put`` on the trainer side (every FSDP
    rank writes its own byte-shard of the flat tensor).  Measured 4-NIC
    agg ~32 GB/s at TP=1 but incompatible with
    ``collective_broadcast`` (which needs the full tensor on one vol).
    Default ``None`` = read ``FORGE_SHARD_PUBLISH`` env (defaults to
    off)."""

    # ---- collective_broadcast-specific ------------------------------
    bcast: BcastBackendConfig = field(default_factory=BcastBackendConfig)
    """Nested tuning for the collective_broadcast backend.  See
    ``BcastBackendConfig`` docstring."""

    # ---- dedicated_ps-specific (future) -----------------------------
    ps_world: int = 0
    """Number of dedicated parameter-server procs (future
    ``dedicated_ps`` backend).  0 = feature disabled."""


@dataclass
class LauncherConfig:
    """Cluster launcher configuration.

    Modes:
        - ``local``: All actors on the current machine (default).
        - ``slurm``: Allocate machines via Slurm.
        - ``preallocated``: K8s / external scheduler pre-allocated machines.
    """

    launcher: Launcher = Launcher.LOCAL
    job_name: str = ""
    services: dict[str, Any] = field(default_factory=dict)
    actors: dict[str, Any] = field(default_factory=dict)
    gpus_per_node: int = 8
    master_addr: str = ""
    master_port: int = 0
    nnodes: int = 1
    node_rank: int = 0
    workers: list[str] = field(default_factory=list)
    worker_port: int = 22222

    # Which machinery manages the remote worker lifecycle for bare-metal
    # (``launcher == Launcher.BARE_METAL``) deployments.
    #
    # - ``"bash"`` (default, legacy): ``forge launch`` shells out to
    #   ``forge/scripts/worker_manager.sh start/stop``.  This is the
    #   300-line bash path that has been battle-tested on the 2-node
    #   NPU cluster since the first multi-node run.
    # - ``"ssh_job"``: ``forge launch`` uses Monarch's native
    #   :class:`SSHJob` (via our :class:`forge.provisioner_ssh.ForgeSSHJob`
    #   subclass) -- same SSH command, same ``run_worker_loop_forever``,
    #   but lifecycle managed by Monarch's ``JobTrait`` protocol.  This
    #   is the forward-compat path that lines up with ``SlurmJob`` /
    #   ``KubernetesJob`` for future migrations.
    #
    # Opt-in during the migration window so existing CI and muscle
    # memory keep working.  Flip the default to ``"ssh_job"`` once a
    # few full training runs have come back clean.  See
    # ``forge/docs/launcher_impl.md`` for the migration notes.
    launcher_impl: str = "bash"

    # Explicit name -> placement map for launcher.get_host_mesh(name).
    #
    # Today (bare-metal) each value is the integer index into `workers`,
    # e.g. {"trainer": 0, "generator": 1, "storage": 0} pins storage to
    # the same host as trainer.  Leaving this empty keeps the legacy
    # round-robin fallback for backward compat.
    #
    # Future launchers (slurm, k8s) are expected to keep the same
    # "name -> placement" shape: the value type will grow into a dict
    # ({"host_idx": 0, "slurm_mesh_name": ...}) at that point and the
    # bare-metal launcher will accept both int and dict for a migration
    # window.  Using a plain int today keeps the schema approachable;
    # the bare-metal launcher normalizes it to the richer shape
    # internally before consuming it.
    meshes: dict[str, Any] = field(default_factory=dict)

    # Weight-sync data-plane policy.  See ``WeightSyncBlock`` docstring
    # for the full field reference and
    # ``forge/docs/env_to_yaml_mapping.md`` for the env-to-YAML
    # precedence rules.  Unset (default-constructed) block is the
    # legacy state: every knob falls through to its env var (or to
    # the wired-in default).
    weight_sync: WeightSyncBlock = field(default_factory=WeightSyncBlock)

    def __post_init__(self):
        if isinstance(self.launcher, str):
            self.launcher = Launcher(self.launcher)
        # OmegaConf / YAML loaders typically hand us ``dict`` here
        # rather than the target dataclasses -- normalize so
        # downstream ``self.weight_sync.backend`` access just works.
        if isinstance(self.weight_sync, dict):
            raw = dict(self.weight_sync)
            bcast_raw = raw.pop("bcast", None) or {}
            bcast = (
                BcastBackendConfig(**bcast_raw)
                if isinstance(bcast_raw, dict)
                else bcast_raw
            )
            self.weight_sync = WeightSyncBlock(bcast=bcast, **raw)


@dataclass
class ProvisionerConfig:
    """Configuration for the global resource provisioner."""

    launcher_config: LauncherConfig | None = None


Scalar = int | float
