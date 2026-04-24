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
    # Phase A3: per-turn rewards.  Aligned with multi-turn
    # trajectories; ``None`` (default) means this episode is
    # single-turn and only ``reward`` is meaningful.  When the
    # reward pipeline runs process-scope rewards, this holds the
    # aggregated per-turn vector.
    step_rewards: list[float] | None = None
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
        if self.step_rewards is not None:
            d["step_rewards"] = list(self.step_rewards)
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
            step_rewards=d.get("step_rewards"),
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
    storage_role: str | None = None
    """Name of a ``launcher.roles.*`` entry that hosts storage volumes.
    Default ``None`` = ``"trainer"`` (colocated with the training role,
    which is the legacy behavior).  Set to ``"storage"`` or another
    custom role for dedicated-PS topologies.

    R1.5 renamed this field from ``storage_mesh``.  ``storage_mesh`` is
    still accepted as an input alias (with a DeprecationWarning) for
    one release; YAMLs that set both fields to conflicting values get
    a hard error so "did the rename land?" isn't silent."""

    storage_mesh: str | None = None
    """DEPRECATED alias for :attr:`storage_role`.  Kept for one release
    so in-flight YAMLs keep working.  The canonical name is
    ``storage_role`` -- both sides of the reference are roles now, not
    the legacy flat ``meshes`` dict.  :meth:`LauncherConfig.__post_init__`
    normalizes this into ``storage_role`` and warns once."""

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


# Accepted values for :attr:`RoleConfig.hardware`.  Kept narrow on
# purpose: R1.5 only ships the ``npu`` tier that's verified on our
# 2-node Ascend cluster.  CPU roles (``replay_buffer`` / ``tool_server``
# etc.) and ``gpu`` will be added in Phase A2 when we have actual test
# hardware -- silently accepting them today would let YAMLs pass that
# have no corresponding launcher path yet.  See
# ``forge/docs/role_abstraction_design.md`` §4 for the tier roadmap.
_ROLE_HARDWARE_TIERS: frozenset[str] = frozenset({"npu"})


@dataclass
class RoleConfig:
    """Resource request for a single logical unit of work.

    Deliberately small: a role is a *resource bucket*, nothing more.
    Parallelism strategy (FSDP / TP / PP / DP) lives in the actor's
    own workload config (``allocation_mode`` for the trainer,
    ``engine_args.tensor_parallel_size`` for the generator, ...).
    Actor-spawn code reads the workload config, derives
    ``(procs, gpus_per_proc)``, and asks the provisioner for a
    proc mesh sized against this role's ``devices`` budget.  The
    runtime check is::

        procs * gpus_per_proc <= role.devices

    which fails fast at actor init with a clear error if the
    workload config and the resource budget disagree.

    Fields:
        devices: Total number of accelerator cards this role needs.
            The launcher reserves ``devices`` cards across one or
            more hosts from the cluster pool.  Example: ``devices: 4``
            + ``hardware: npu`` asks for 4 NPUs on a single host (or
            spread across hosts with enough free NPUs if the pool
            doesn't have a 4-card host).  ``0`` is legal for CPU-only
            roles (tool server, replay buffer, ...).
        hardware: Device tier.  Currently only ``"npu"`` is accepted;
            see ``_ROLE_HARDWARE_TIERS`` for the rationale.
        colocate: Name of another role this one must share host(s)
            with.  The canonical use is ``storage.colocate: trainer``
            so torchstore's HiXL put leg stays on-host (HCCS) with
            the FSDP ranks it's reading from.  Launcher enforces by
            pinning to the same host(s) chosen for ``colocate``'s
            role after that role is scheduled.
        host_idx: Explicit worker index into ``bare_metal.workers[]``.
            Two legitimate uses:
              * small clusters where you want explicit pins instead
                of pool-scheduled placement (2-node dev setups);
              * legacy ``meshes: {X: host_idx: N}`` YAMLs that the
                R1.5 bridge auto-migrates.
            ``None`` (the default) lets the pool scheduler pick.
        extras: Free-form dict for workload-specific knobs that don't
            fit in the resource-bucket abstraction (tool-server
            ``timeout``, replay-buffer ``num_replicas``, ...).  The
            launcher itself never reads ``extras``; downstream actor
            code does.  This keeps ``RoleConfig`` small without
            forcing every A2/A3 feature to bump the dataclass.

    Intentionally *not* first-class fields (pushed to other layers):
        - ``procs`` / ``gpus_per_proc``: derived from workload config
          (``allocation_mode`` / ``tensor_parallel_size``) at
          actor-spawn time, not declared per-role.
        - ``node_selector`` / ``count`` / ``anti_colocate``: punted to
          Phase R2 (or never, if the 9-node bare-metal topology
          doesn't need them).  ``colocate`` is the one placement
          constraint we actually use today (storage-next-to-trainer),
          so it lives here.
        - ``role_type`` / ``agent`` / ``workflow`` / ``tools``: mixed
          "WHAT to run" with "WHERE to run".  If Phase A2 needs these
          back they can land in ``extras`` or in a sibling top-level
          block, but they don't belong in the resource schema.
    """

    devices: int = 0
    hardware: str = "npu"
    colocate: str | None = None
    host_idx: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.devices < 0:
            raise ValueError(
                f"RoleConfig.devices must be >= 0, got {self.devices!r}. "
                f"Use 0 for CPU-only roles or a positive integer for "
                f"accelerator roles."
            )
        if self.hardware not in _ROLE_HARDWARE_TIERS:
            accepted = ", ".join(sorted(_ROLE_HARDWARE_TIERS))
            raise ValueError(
                f"RoleConfig.hardware={self.hardware!r} is not supported "
                f"in R1.5 (accepted: {{{accepted}}}).  CPU / GPU tiers "
                f"arrive in Phase A2 once there's hardware to test them "
                f"against; silently accepting the value now would let "
                f"YAMLs pass that have no corresponding launcher path."
            )


_CORE_ROLE_FIELDS: frozenset[str] = frozenset(
    {"devices", "hardware", "colocate", "host_idx", "extras"}
)


def _normalize_role(raw: Any) -> RoleConfig:
    """Coerce a YAML-loaded dict (or existing ``RoleConfig``) into a
    :class:`RoleConfig`.

    Accepts legacy-nested shapes (``{"hardware": {"type": "npu"},
    "placement": {"host_idx": 0}}``) and transparently flattens them
    into the new flat schema.  Unknown keys are NOT an error -- they
    flow into :attr:`RoleConfig.extras` so A2 workload knobs
    (``num_replicas``, ``timeout``, ``tools``, ...) can be authored
    alongside resource fields without bumping this dataclass on every
    new feature.
    """
    if isinstance(raw, RoleConfig):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(
            f"RoleConfig entry must be dict or RoleConfig, got "
            f"{type(raw).__name__}: {raw!r}"
        )

    data = dict(raw)

    # Flatten the short-lived nested R1 schema (``hardware`` /
    # ``placement`` sub-blocks) into the new flat shape.  The nested
    # schema only ever existed in the working tree; we migrate it
    # transparently so any stale YAML that still uses it keeps
    # working.
    legacy_hw = data.pop("hardware", None)
    if isinstance(legacy_hw, dict):
        hw_type = legacy_hw.get("type")
        if hw_type is not None:
            data.setdefault("hardware", hw_type)
    elif legacy_hw is not None:
        data["hardware"] = legacy_hw

    legacy_placement = data.pop("placement", None)
    if isinstance(legacy_placement, dict):
        if "host_idx" in legacy_placement:
            data.setdefault("host_idx", legacy_placement["host_idx"])
        if "colocate_with" in legacy_placement and "colocate" not in data:
            data["colocate"] = legacy_placement["colocate_with"]

    # Everything that's not a core field joins ``extras`` so the
    # schema stays extensible for A2/A3 workload knobs
    # (``num_replicas`` / ``timeout`` / ``tools`` / ...).  Caller-
    # authored ``extras`` (if any) is merged on top so explicit
    # ``extras.X`` wins over loose ``X`` of the same name.
    extras: dict[str, Any] = {}
    for key in list(data):
        if key == "extras":
            continue
        if key not in _CORE_ROLE_FIELDS:
            extras[key] = data.pop(key)
    caller_extras = data.pop("extras", None) or {}
    if isinstance(caller_extras, dict):
        extras.update(caller_extras)

    return RoleConfig(**data, extras=extras)


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
    # ``forge/docs/weight_sync.md`` §8 for the migration notes.
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

    # Role-driven placement schema (Phase R1).  When set, each entry
    # is a :class:`RoleConfig` describing hardware, placement, and
    # role-type metadata for a logical unit of work (trainer,
    # generator, agent_runner, tool_server, ...).
    #
    # R1 behavior: ``__post_init__`` keeps ``roles`` and ``meshes`` in
    # sync -- either one can be authored and the other is
    # auto-populated.  Downstream consumers (``Provisioner``,
    # ``BareMetalLauncher``) still read ``meshes``; the migration to
    # reading ``roles`` directly happens in R2 alongside the placement
    # scheduler.  This keeps R1 a pure schema change with zero runtime
    # risk.  See ``forge/docs/role_abstraction_design.md`` §2 for the
    # target schema and ``forge/docs/agentic_rl_architecture.md`` §3
    # for the agentic-specific role types.
    roles: dict[str, RoleConfig] = field(default_factory=dict)

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

        # Role normalization + legacy meshes bridge.  Do this after
        # weight_sync normalization so downstream code can rely on
        # ``self.roles`` being fully-typed.
        if self.roles:
            self.roles = {
                name: _normalize_role(raw) for name, raw in dict(self.roles).items()
            }
        self._sync_roles_and_meshes()
        self._normalize_weight_sync_role_refs()
        self._validate_role_refs()

    def _normalize_weight_sync_role_refs(self) -> None:
        """Collapse the ``storage_mesh`` / ``storage_role`` deprecation
        alias into a single canonical attribute.

        R1.5 renamed ``storage_mesh`` -> ``storage_role`` because both
        sides of the reference (the field itself, and the dict it
        points into) are now ``roles``, not the legacy flat ``meshes``
        dict.  We accept both names for one release window:

        * ``storage_role`` alone (the new canonical): no-op.
        * ``storage_mesh`` alone (legacy): copy to ``storage_role``,
          warn once.
        * Both set to identical values: copy, warn once (idempotent
          YAML migration where someone kept both for safety).
        * Both set to DIFFERENT values: hard error.  Ambiguous input
          is worse than a broken config -- we'd rather fail at load
          time than have the user wondering which one took effect.
        """
        ws = self.weight_sync
        if ws is None or not isinstance(ws, WeightSyncBlock):
            return
        old, new = ws.storage_mesh, ws.storage_role
        if old is None:
            # User only set the new name (or neither).  No migration
            # warning needed; still mirror new -> old so legacy
            # readers that peek at ``storage_mesh`` see a value.
            if new is not None:
                ws.storage_mesh = new
            return
        if new is not None and new != old:
            raise ValueError(
                f"weight_sync.storage_mesh={old!r} conflicts with "
                f"weight_sync.storage_role={new!r}.  Pick one.  "
                f"(``storage_mesh`` is the deprecated alias; delete it "
                f"and keep ``storage_role``.)"
            )
        # At this point the user authored ``storage_mesh`` (either
        # alone, or alongside an identical ``storage_role``).  Either
        # way, the deprecated name is in the YAML and the user
        # deserves a warning.
        import warnings
        warnings.warn(
            "weight_sync.storage_mesh is deprecated (R1.5); rename to "
            "weight_sync.storage_role.  Both names point at the same "
            "``launcher.roles.*`` entry; the rename reflects that the "
            "dict being referenced is ``roles``, not the legacy "
            "``meshes`` dict.  ``storage_mesh`` will be removed after "
            "the next release.",
            DeprecationWarning,
            stacklevel=4,
        )
        if new is None:
            ws.storage_role = old
        # Keep both names pointing at the same value so legacy
        # readers don't go stale.
        ws.storage_mesh = ws.storage_role

    def _validate_role_refs(self) -> None:
        """Fail fast when a string-typed role reference names a role
        that doesn't exist.

        Today the only such reference is ``weight_sync.storage_role``,
        but the pattern will recur (``transports:`` blocks in Phase R3,
        for example), so we keep the validation generic.  We only
        validate when ``self.roles`` is non-empty -- legacy YAMLs that
        didn't declare any ``roles:`` block are allowed to use raw
        ``meshes`` names, which the Provisioner will resolve against
        ``meshes`` directly.
        """
        if not self.roles:
            return
        ws = self.weight_sync if isinstance(self.weight_sync, WeightSyncBlock) else None
        if ws is None or not ws.storage_role:
            return
        # "trainer" is a sentinel default that every forge run has,
        # even without an explicit ``roles.trainer`` entry -- it maps
        # to the TrainerActor's mesh name and the reverse bridge will
        # have synthesized it if the user didn't declare it.  So we
        # only flag refs that name something *not* in ``roles`` and
        # *not* in the legacy ``meshes`` dict.
        ref = ws.storage_role
        if ref not in self.roles and ref not in self.meshes:
            available = sorted(set(self.roles) | set(self.meshes))
            raise ValueError(
                f"weight_sync.storage_role={ref!r} does not match any "
                f"entry in launcher.roles or launcher.meshes.  "
                f"Available: {available!r}.  (Check for typos -- this "
                f"used to be a silent fallback that resolved to a "
                f"default mesh; we now fail fast to surface config "
                f"errors at YAML load time.)"
            )

    def _sync_roles_and_meshes(self) -> None:
        """Keep ``roles`` and legacy ``meshes`` consistent in BOTH
        directions, regardless of which side(s) the user authored.

        Migration window contract (R1.5):

        * Forward bridge (``roles`` -> ``meshes``): for any role whose
          ``host_idx`` is set and whose name is NOT already present in
          ``meshes``, write ``meshes[name] = {"host_idx": host_idx}``
          so ``BareMetalLauncher.get_host_mesh(name)`` honors
          role-authored placement without a second code path.  Roles
          without ``host_idx`` (the common case once the pool
          scheduler lands) skip this bridge -- the launcher reads
          ``devices`` directly instead.
        * Reverse bridge (``meshes`` -> ``roles``): for every entry in
          ``meshes`` whose name is NOT already present in ``roles``,
          synthesize a minimal :class:`RoleConfig(host_idx=N)` so new
          code that iterates ``launcher.roles`` still sees legacy-only
          meshes.

        Conflict rule: entries present on BOTH sides are never
        overwritten.  If the user authored ``roles.trainer.host_idx=0``
        AND ``meshes.trainer=1``, the ``meshes`` value wins (that's
        the hot data path for legacy code) and ``roles`` is left
        alone.  Legitimate use case: a migration where an agent
        injects ``roles`` entries alongside hand-written ``meshes``;
        we refuse to guess which wins, we just trust what the user
        wrote.
        """
        # Forward: roles -> meshes.  Only fills names NOT already in
        # ``meshes``; this preserves the "user intent wins" rule.
        for name, role in self.roles.items():
            if name in self.meshes:
                continue
            if role.host_idx is None:
                continue
            self.meshes[name] = {"host_idx": int(role.host_idx)}

        # Reverse: meshes -> roles.  Synthesize minimal entries so
        # roles-aware code sees the full picture.
        for name, placement in self.meshes.items():
            if name in self.roles:
                continue
            host_idx: int | None
            if isinstance(placement, int):
                host_idx = placement
            elif isinstance(placement, dict):
                raw = placement.get("host_idx")
                host_idx = int(raw) if raw is not None else None
            else:
                # Unknown shape (future launcher-specific dict).
                # Skip role synthesis; Provisioner will surface any
                # shape error at use time with a clearer message.
                continue
            self.roles[name] = RoleConfig(host_idx=host_idx)


@dataclass
class ProvisionerConfig:
    """Configuration for the global resource provisioner."""

    launcher_config: LauncherConfig | None = None


Scalar = int | float
