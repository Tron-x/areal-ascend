"""Tool-server placement resolver (Phase A2).

Translates ``LauncherConfig.roles["tool_server"]`` (role schema from
Phase R1) into the concrete spawn parameters that
``forge.actors.sandbox.SandboxActor`` needs:

* ``procs`` -- Monarch procs per actor
* ``num_replicas`` -- how many load-balanced copies to spin up
* ``as_service`` -- ``True`` when ``num_replicas > 1``, else
  single-actor mode (legacy path)
* ``mesh_name`` -- Monarch mesh name (defaults to ``"sandbox"``)
* ``tool_types`` -- names of tools to mount
* ``host_idx`` -- explicit placement hint for R2's
  ``BareMetalLauncher.get_host_mesh(role_name)`` (currently
  informational; R1/R2 bridge reads it)

Callers pass in an already-resolved :class:`LauncherConfig` (or
``None`` for "no YAML roles block authored, fall back to legacy
single-actor path").

Design notes:

* **Backward compatibility** is the hard constraint.  When ``roles``
  is empty or does not include ``tool_server``, this resolver
  returns the legacy default, which matches the pre-A2 hardcoded
  ``SandboxActor.options(procs=1, mesh_name="sandbox").as_actor()``.
* **Extras passthrough**: unknown keys under
  ``roles.tool_server.extras`` (e.g. ``timeout``, ``max_output_len``)
  are surfaced verbatim so ``SandboxActor``-side knobs don't need a
  dataclass bump on every new setting.
* **No imports from Monarch/forge.actors** -- this module is a pure
  config translator so it can be unit-tested without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge.core.types import LauncherConfig


_DEFAULT_MESH_NAME = "sandbox"
_DEFAULT_TOOL_TYPES = ("python_sandbox",)


@dataclass
class ToolServerOptions:
    """Resolved tool-server spawn parameters.

    Consumers (e.g. ``agent_rl.py``) use this to decide between
    ``SandboxActor.options(...).as_actor()`` and
    ``SandboxActor.options(...).as_service(num_replicas=N)``.
    """

    enabled: bool = True
    procs: int = 1
    num_replicas: int = 1
    mesh_name: str = _DEFAULT_MESH_NAME
    tool_types: tuple[str, ...] = _DEFAULT_TOOL_TYPES
    host_idx: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    _source: str = "default"

    @property
    def as_service(self) -> bool:
        """True when we should spawn a load-balanced replica set."""
        return self.num_replicas > 1


def resolve_tool_server_options(
    launcher_config: LauncherConfig | None,
    *,
    role_name: str = "tool_server",
) -> ToolServerOptions:
    """Resolve spawn parameters for the tool-server role.

    Precedence (highest to lowest):

    1. ``launcher_config.roles[role_name]`` -- explicit role entry.
    2. Legacy default: single actor, ``procs=1``, mesh ``sandbox``.

    This function NEVER raises on missing config; a bare
    :class:`ToolServerOptions` is always returned so callers can
    stay on the happy path.
    """

    if launcher_config is None:
        return ToolServerOptions(_source="default:no-launcher-config")

    roles = getattr(launcher_config, "roles", None) or {}
    role = roles.get(role_name)
    if role is None:
        return ToolServerOptions(_source="default:no-role-entry")

    # R1.5 simplified ``RoleConfig`` down to resource fields only
    # (``devices`` / ``hardware`` / ``colocate`` / ``host_idx``).
    # Everything else authored under ``roles.tool_server`` flows into
    # ``role.extras`` (see ``forge/core/types.py::_normalize_role``),
    # so this resolver pulls A2 knobs exclusively from there.
    extras = dict(getattr(role, "extras", {}) or {})
    procs = max(1, int(extras.pop("procs", 1) or 1))
    num_replicas = _extract_num_replicas(extras)
    tool_types = _extract_tool_types(extras)
    # Default mesh_name to the role_name itself -- that's the key the
    # R1.5 bridge writes ``meshes`` entries under, so
    # ``BareMetalLauncher.get_host_mesh(mesh_name)`` resolves the
    # right placement without extra wiring.  Explicit override wins.
    mesh_name = str(extras.pop("mesh_name", role_name))
    host_idx = getattr(role, "host_idx", None)

    return ToolServerOptions(
        enabled=bool(extras.pop("enabled", True)),
        procs=procs,
        num_replicas=num_replicas,
        mesh_name=mesh_name,
        tool_types=tool_types,
        host_idx=host_idx,
        extras=extras,
        _source=f"roles.{role_name}",
    )


def _extract_num_replicas(extras: dict[str, Any]) -> int:
    """Replica count from extras.

    Precedence:

    * ``extras.num_replicas`` (explicit user intent)
    * ``extras.count`` (legacy R1 field that the schema used to
      surface as ``placement.count``; R1.5 folds it into extras)
    * 1 (legacy single actor)
    """
    if "num_replicas" in extras:
        return max(1, int(extras.pop("num_replicas")))
    if "count" in extras:
        return max(1, int(extras.pop("count")))
    return 1


def _extract_tool_types(extras: dict[str, Any]) -> tuple[str, ...]:
    """Pull the list of tools the server should expose.

    Honors (in order):

    * ``extras.tools`` (list of short names, A2 canonical)
    * ``extras.tool_types`` (role-agnostic override)
    * default: ``("python_sandbox",)`` -- matches current
      ``SandboxActor`` behavior
    """
    tools = extras.pop("tools", None)
    if tools:
        if isinstance(tools, str):
            return (tools,)
        return tuple(str(t) for t in tools)
    raw = extras.pop("tool_types", None)
    if raw:
        if isinstance(raw, str):
            return (raw,)
        return tuple(str(t) for t in raw)
    return _DEFAULT_TOOL_TYPES
