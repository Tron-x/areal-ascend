"""Two-layer YAML composer: algo YAML (resource intent) + cluster preset.

See ``forge/docs/role_abstraction_design.md §2b`` and the *two-layer YAML*
discussion in the 2026-04 architecture review.  Today launching a
multi-node GRPO run requires the operator to author two YAMLs:

- **Algorithm YAML** (e.g. ``examples/math/gsm8k_grpo_npu.yaml``): reward
  / workflow / dataset / hyperparams.  Owned by the algorithm engineer.
- **Launcher YAML** (e.g. ``forge/configs/launcher_bare_metal_2node.yaml``):
  ``pool`` / ``roles`` / ``weight_sync`` / ``launcher_impl``.  Owned by
  infra / ops.

Algorithm engineers shouldn't have to read the launcher YAML just to
say "I want 4 cards for training and 4 for inference".  This module
bridges that gap via two primitives:

- :func:`resolve_cluster_preset` -- look up a named preset under
  ``forge/configs/clusters/`` (or a repo-local ``clusters/`` next to
  the algo YAML, or an absolute path).
- :func:`compose_launcher_yaml` -- merge the algo YAML's top-level
  ``roles:`` overrides into the preset, producing a single launcher
  block that :class:`LauncherConfig` can consume unchanged.

The composition is intentionally narrow: only ``devices`` and
``extras`` in ``roles.<name>`` can be overridden from the algo YAML.
``colocate`` / ``host_idx`` / ``hardware`` are infra decisions that
live in the preset.  A power-user escape hatch is documented in §3 of
``role_abstraction_design.md``; it's gated on the overrides being in
a small whitelist to prevent the algo YAML from silently changing
fabric choices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "PresetError",
    "resolve_cluster_preset",
    "merge_role_overrides",
    "compose_launcher_yaml",
    "ALGO_OVERRIDE_WHITELIST",
    "ALGO_OVERRIDE_ESCAPE_HATCH",
]


_REPO_ROOT = Path(__file__).resolve().parents[2]
"""``/root/AReaL`` in dev. Used as the anchor for ``forge/configs/clusters/``."""

_BUILTIN_CLUSTERS_DIR = _REPO_ROOT / "forge" / "configs" / "clusters"


ALGO_OVERRIDE_WHITELIST: frozenset[str] = frozenset({"devices", "extras"})
"""Fields the algo YAML is *always* allowed to override per role.

``devices``  : the algorithm's resource request (business decision).
``extras``   : workload-specific kv bag (parallelism metadata,
               tokenizer overrides, etc.).  Treated opaquely.
"""

ALGO_OVERRIDE_ESCAPE_HATCH: frozenset[str] = frozenset({"colocate", "host_idx"})
"""Fields the algo YAML *may* override but shouldn't under normal use.

These are placement decisions (topology) that properly belong to the
infra preset.  We allow them with a warning so a power user can still
pin a role without hand-editing the preset.  Anything outside both
sets raises :class:`PresetError`.
"""


class PresetError(ValueError):
    """Raised when a preset lookup fails or an algo override is illegal."""


def resolve_cluster_preset(
    name: str,
    *,
    algo_yaml_dir: Path | None = None,
    builtin_dir: Path = _BUILTIN_CLUSTERS_DIR,
) -> Path:
    """Resolve a preset reference to a concrete file path.

    Search order:

    1. Absolute path or path with a YAML extension -- use verbatim.
    2. ``{algo_yaml_dir}/clusters/{name}.yaml`` -- project-local
       override, lets a user vendor a cluster definition next to the
       experiment without forking the framework.
    3. ``{builtin_dir}/{name}.yaml`` -- the in-tree presets under
       ``forge/configs/clusters/``.

    Parameters
    ----------
    name:
        Either a bare preset name (``"2node_colocated"``) or an
        explicit path (absolute, or ending in ``.yaml``/``.yml``).
    algo_yaml_dir:
        Directory of the algo YAML that referenced this preset.
        Enables the project-local override tier.  ``None`` skips that
        tier (useful in unit tests).
    builtin_dir:
        Override for the in-tree preset directory.  Only changed in
        tests.

    Raises
    ------
    PresetError
        When the named preset is not found in any search tier.
    """
    # Tier 1: explicit path (absolute, or has a YAML suffix).  We
    # resolve here so the caller can pass a relative path and still
    # get a deterministic answer.
    candidate = Path(name)
    if candidate.is_absolute() or candidate.suffix in {".yaml", ".yml"}:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
        raise PresetError(
            f"launcher_preset={name!r} looks like a path but does not exist: {resolved}"
        )

    tried: list[Path] = []

    # Tier 2: project-local override next to the algo YAML.
    if algo_yaml_dir is not None:
        local = (algo_yaml_dir / "clusters" / f"{name}.yaml").resolve()
        tried.append(local)
        if local.is_file():
            return local

    # Tier 3: in-tree preset directory.  Try both .yaml and .yml so
    # operators don't stumble on the extension.
    for ext in (".yaml", ".yml"):
        builtin = (builtin_dir / f"{name}{ext}").resolve()
        tried.append(builtin)
        if builtin.is_file():
            return builtin

    raise PresetError(
        f"launcher_preset={name!r} not found.  Tried:\n  "
        + "\n  ".join(str(p) for p in tried)
        + f"\n\nAvailable built-in presets: {_list_builtin_presets(builtin_dir)}"
    )


def _list_builtin_presets(builtin_dir: Path) -> list[str]:
    """Best-effort enumeration of in-tree preset names (for error messages)."""
    if not builtin_dir.is_dir():
        return []
    return sorted(p.stem for p in builtin_dir.glob("*.yaml"))


def merge_role_overrides(
    preset_roles: dict[str, dict[str, Any]],
    algo_roles: dict[str, dict[str, Any]] | None,
    *,
    on_escape_hatch: callable | None = None,  # type: ignore[type-arg]
) -> dict[str, dict[str, Any]]:
    """Apply algo-layer per-role overrides onto a preset's roles block.

    Semantics (see module docstring + role_abstraction_design.md §3):

    - Fields in :data:`ALGO_OVERRIDE_WHITELIST` are applied silently.
    - Fields in :data:`ALGO_OVERRIDE_ESCAPE_HATCH` are applied but
      trigger ``on_escape_hatch(role, field, value)`` so the caller
      can emit a warning.
    - Any other field raises :class:`PresetError` -- this prevents
      e.g. the algo YAML silently flipping ``hardware: npu`` -> ``gpu``.
    - Algo roles that aren't declared in the preset raise
      :class:`PresetError`: role topology (who colocates with whom,
      CPU sidecar vs accelerator) is an infra decision, so the
      preset must at least acknowledge the role's existence.

    Parameters
    ----------
    preset_roles:
        The ``launcher.roles`` block from the preset YAML.  Deep-copied
        before mutation so the caller's dict is not touched.
    algo_roles:
        The top-level ``roles:`` block from the algo YAML (or ``None``
        when the algo YAML declares no overrides).
    on_escape_hatch:
        Optional callback invoked as ``on_escape_hatch(role, field,
        value)`` whenever an override from
        :data:`ALGO_OVERRIDE_ESCAPE_HATCH` is applied.  Typically wired
        to ``warnings.warn`` or a launcher-side stderr print.

    Returns
    -------
    dict
        New dict mirroring ``preset_roles`` with overrides applied.
    """
    import copy

    merged = copy.deepcopy(preset_roles)
    if not algo_roles:
        return merged

    for role_name, overrides in algo_roles.items():
        if role_name not in merged:
            raise PresetError(
                f"algo YAML declares role {role_name!r} but the preset "
                f"has no such role.  Known roles in preset: "
                f"{sorted(merged)}.  Add the role to the preset first "
                f"(infra decision) before overriding its device count."
            )
        if not isinstance(overrides, dict):
            raise PresetError(
                f"algo YAML roles.{role_name!r} must be a mapping, got "
                f"{type(overrides).__name__}"
            )
        for field, value in overrides.items():
            if field in ALGO_OVERRIDE_WHITELIST:
                merged[role_name][field] = value
            elif field in ALGO_OVERRIDE_ESCAPE_HATCH:
                merged[role_name][field] = value
                if on_escape_hatch is not None:
                    on_escape_hatch(role_name, field, value)
            else:
                raise PresetError(
                    f"algo YAML cannot override roles.{role_name}.{field} "
                    f"(field belongs to the cluster preset, not the algo).  "
                    f"Allowed from algo: {sorted(ALGO_OVERRIDE_WHITELIST | ALGO_OVERRIDE_ESCAPE_HATCH)}. "
                    f"Edit the preset YAML directly if you really need to change this."
                )
    return merged


def compose_launcher_yaml(
    algo_yaml: dict[str, Any],
    preset_yaml: dict[str, Any],
    *,
    on_escape_hatch: callable | None = None,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Build a single launcher YAML dict from algo overrides + preset.

    The returned dict is in the same shape as the legacy launcher YAML
    (``launcher_bare_metal_2node.yaml``): a top-level ``launcher:`` block
    with ``pool``, ``roles``, ``weight_sync``, ``launcher_impl``, etc.
    This lets the existing :class:`LauncherConfig` consumer stay
    unchanged -- the composition is purely a pre-processing step.

    Parameters
    ----------
    algo_yaml:
        Parsed algo YAML (e.g. ``gsm8k_grpo_npu.yaml``).  Only the
        top-level ``roles:`` block is consumed.  Everything else
        (reward / workflow / hyperparams / ...) is ignored by this
        function and passed through to grpo.py unchanged.
    preset_yaml:
        Parsed cluster preset.  Must have a top-level ``launcher:``
        block containing ``roles:`` and ``pool:``; other fields
        (``weight_sync``, ``launcher_impl``, ...) pass through.
    on_escape_hatch:
        Forwarded to :func:`merge_role_overrides`.

    Returns
    -------
    dict
        Composed launcher dict ready to be written to a temp YAML and
        fed to ``grpo.py --forge-config``.
    """
    import copy

    if "launcher" not in preset_yaml or not isinstance(preset_yaml["launcher"], dict):
        raise PresetError(
            "cluster preset is missing a top-level 'launcher:' block; "
            "presets must declare pool + roles under launcher:"
        )

    composed = copy.deepcopy(preset_yaml)
    preset_roles = composed["launcher"].get("roles") or {}
    if not isinstance(preset_roles, dict):
        raise PresetError(
            f"cluster preset's launcher.roles must be a mapping, got "
            f"{type(preset_roles).__name__}"
        )

    algo_roles = algo_yaml.get("roles")
    composed["launcher"]["roles"] = merge_role_overrides(
        preset_roles,
        algo_roles if isinstance(algo_roles, dict) else None,
        on_escape_hatch=on_escape_hatch,
    )
    return composed
