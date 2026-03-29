"""
Cluster topology and device placement for Monarch AReaL.

Abstracts away single-node vs multi-node differences so the launcher
can treat all configurations uniformly:

    topology = ClusterTopology.from_config(config, alloc_mode)
    placement = topology.placement

    # Works identically for 1+1, 4+4, 2+6, multi-node, etc.
    gen_host   = placement.inference_host_mesh
    train_host = placement.training_host_mesh
    cpu_host   = placement.cpu_host_mesh

Multi-node setup requires workers on each node running
``monarch.actor.bootstrap.run_worker_loop_forever``.  Pass their
addresses via the ``MONARCH_WORKERS`` env var (comma-separated) or
the ``cluster.monarch_workers`` config field.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from areal.api import AllocationMode, AllocationType

logger = logging.getLogger("MonarchPlugin")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NodeDevices:
    """Devices available on a single physical node."""

    host_addr: str
    device_ids: List[str]

    @property
    def count(self) -> int:
        return len(self.device_ids)


@dataclass
class RolePlacement:
    """Which node(s) and device(s) are assigned to a role."""

    nodes: List[NodeDevices]

    @property
    def total_devices(self) -> int:
        return sum(n.count for n in self.nodes)

    @property
    def all_device_ids(self) -> List[str]:
        result = []
        for n in self.nodes:
            result.extend(n.device_ids)
        return result

    @property
    def is_multi_node(self) -> bool:
        return len(self.nodes) > 1

    @property
    def host_addrs(self) -> List[str]:
        return [n.host_addr for n in self.nodes]

    def devices_per_node(self) -> int:
        if not self.nodes:
            return 0
        counts = {n.count for n in self.nodes}
        if len(counts) != 1:
            raise ValueError(
                f"Non-uniform device counts across nodes: "
                f"{[n.count for n in self.nodes]}. "
                f"Monarch requires uniform per_host allocation."
            )
        return self.nodes[0].count


@dataclass
class DevicePlacement:
    """Complete placement plan for all actor roles."""

    inference: RolePlacement
    training: RolePlacement
    cpu_services: RolePlacement
    master_addr: str = "127.0.0.1"

    @property
    def is_multi_node(self) -> bool:
        all_addrs = set(self.inference.host_addrs
                        + self.training.host_addrs
                        + self.cpu_services.host_addrs)
        return len(all_addrs) > 1

    @property
    def total_npu_count(self) -> int:
        return self.inference.total_devices + self.training.total_devices


# ---------------------------------------------------------------------------
# Topology builder
# ---------------------------------------------------------------------------


class ClusterTopology:
    """Builds a DevicePlacement from AReaL config + allocation_mode.

    Single-node:
      Devices are linearly partitioned: [0..gen_count) for inference,
      [gen_count..gen_count+train_count) for training.

    Multi-node:
      Devices are partitioned per-node according to the allocation_mode
      and the worker list.  Currently supports two patterns:

      (a) Symmetric: every node contributes to both inference and training.
          e.g. 2 nodes × 8 NPU, allocation d4p1t1+d4p1t1:
               node0: NPU 0-3 inference, NPU 4-7 training
               node1: NPU 0-3 inference, NPU 4-7 training

      (b) Role-split: some nodes do inference, others do training.
          e.g. 2 nodes × 8 NPU, node0=inference, node1=training
          (controlled by ``cluster.monarch_node_roles``)
    """

    def __init__(
        self,
        n_nodes: int,
        n_devices_per_node: int,
        alloc_mode: AllocationMode,
        worker_addrs: Optional[List[str]] = None,
        node_roles: Optional[List[str]] = None,
        env_var: str = "ASCEND_RT_VISIBLE_DEVICES",
    ):
        self.n_nodes = n_nodes
        self.n_devices_per_node = n_devices_per_node
        self.alloc_mode = alloc_mode
        self.env_var = env_var

        self.worker_addrs = worker_addrs or [self._local_addr()]
        if len(self.worker_addrs) < n_nodes:
            self.worker_addrs.extend(
                [self._local_addr()] * (n_nodes - len(self.worker_addrs))
            )

        self.node_roles = node_roles
        self._placement: Optional[DevicePlacement] = None

    @classmethod
    def from_config(
        cls,
        config,
        alloc_mode: AllocationMode,
        env_var: str = "ASCEND_RT_VISIBLE_DEVICES",
    ) -> "ClusterTopology":
        n_nodes = getattr(config.cluster, "n_nodes", 1)
        n_devices = config.cluster.n_gpus_per_node

        worker_addrs = cls._parse_worker_addrs(config)
        node_roles = cls._parse_node_roles(config)

        return cls(
            n_nodes=n_nodes,
            n_devices_per_node=n_devices,
            alloc_mode=alloc_mode,
            worker_addrs=worker_addrs,
            node_roles=node_roles,
            env_var=env_var,
        )

    @property
    def placement(self) -> DevicePlacement:
        if self._placement is None:
            self._placement = self._compute_placement()
        return self._placement

    @property
    def gen_device_count(self) -> int:
        am = self.alloc_mode
        return am.gen.pp_size * am.gen.tp_size * am.gen.dp_size

    @property
    def train_device_count(self) -> int:
        am = self.alloc_mode
        if am.type_ == AllocationType.LLM_SERVER_ONLY:
            return 0
        return am.train.world_size

    @property
    def is_multi_node(self) -> bool:
        return self.n_nodes > 1

    # ----- placement computation -----

    def _compute_placement(self) -> DevicePlacement:
        if self.n_nodes == 1:
            return self._single_node_placement()
        if self.node_roles:
            return self._role_split_placement()
        return self._symmetric_placement()

    def _single_node_placement(self) -> DevicePlacement:
        all_ids = self._resolve_device_ids_for_node(0)
        gen_count = self.gen_device_count
        train_count = self.train_device_count

        local = self._local_addr()
        inf_ids = all_ids[:gen_count]
        train_ids = all_ids[gen_count: gen_count + train_count]

        return DevicePlacement(
            inference=RolePlacement([NodeDevices(local, inf_ids)]),
            training=RolePlacement([NodeDevices(local, train_ids)]),
            cpu_services=RolePlacement([NodeDevices(local, [])]),
            master_addr="127.0.0.1",
        )

    def _symmetric_placement(self) -> DevicePlacement:
        """Each node mirrors the same inference/training split."""
        gen_per_node = self.gen_device_count // self.n_nodes
        train_per_node = self.train_device_count // self.n_nodes

        inf_nodes = []
        train_nodes = []
        cpu_nodes = []

        for i in range(self.n_nodes):
            addr = self.worker_addrs[i]
            all_ids = self._resolve_device_ids_for_node(i)
            inf_nodes.append(NodeDevices(addr, all_ids[:gen_per_node]))
            train_nodes.append(
                NodeDevices(addr, all_ids[gen_per_node: gen_per_node + train_per_node])
            )
            cpu_nodes.append(NodeDevices(addr, []))

        master = self.worker_addrs[0]

        return DevicePlacement(
            inference=RolePlacement(inf_nodes),
            training=RolePlacement(train_nodes),
            cpu_services=RolePlacement(cpu_nodes),
            master_addr=master,
        )

    def _role_split_placement(self) -> DevicePlacement:
        """Nodes are designated as 'inference' or 'training'."""
        inf_nodes = []
        train_nodes = []

        for i, role in enumerate(self.node_roles):
            addr = self.worker_addrs[i]
            all_ids = self._resolve_device_ids_for_node(i)
            if role == "inference":
                inf_nodes.append(NodeDevices(addr, all_ids))
            elif role == "training":
                train_nodes.append(NodeDevices(addr, all_ids))
            else:
                raise ValueError(f"Unknown node role: {role}")

        if not inf_nodes:
            raise ValueError("No inference nodes in role assignment")
        if not train_nodes:
            raise ValueError("No training nodes in role assignment")

        master = train_nodes[0].host_addr
        cpu_host = inf_nodes[0].host_addr

        return DevicePlacement(
            inference=RolePlacement(inf_nodes),
            training=RolePlacement(train_nodes),
            cpu_services=RolePlacement([NodeDevices(cpu_host, [])]),
            master_addr=master,
        )

    # ----- host mesh creation -----

    def create_host_mesh(self):
        """Create the appropriate Monarch HostMesh for this topology.

        Single-node  -> this_host()
        Multi-node   -> attach_to_workers(workers=...)
        """
        if self.n_nodes == 1:
            from monarch.actor import this_host
            return this_host()
        else:
            from monarch._src.actor.bootstrap import attach_to_workers
            logger.info(
                f"Attaching to {self.n_nodes} worker nodes: "
                f"{self.worker_addrs[:self.n_nodes]}"
            )
            return attach_to_workers(
                name="areal_cluster",
                ca="trust_all_connections",
                workers=self.worker_addrs[:self.n_nodes],
            )

    # ----- helpers -----

    def _resolve_device_ids_for_node(self, node_idx: int) -> List[str]:
        if node_idx == 0:
            raw = os.environ.get(self.env_var, "")
            if raw:
                return [d.strip() for d in raw.split(",") if d.strip()]
        return [str(i) for i in range(self.n_devices_per_node)]

    @staticmethod
    def _local_addr() -> str:
        try:
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception:
            return "127.0.0.1"

    @staticmethod
    def _parse_worker_addrs(config) -> Optional[List[str]]:
        env_addrs = os.environ.get("MONARCH_WORKERS", "")
        if env_addrs:
            return [a.strip() for a in env_addrs.split(",") if a.strip()]

        try:
            addrs = config.cluster.get("monarch_workers", None)
            if addrs:
                return list(addrs)
        except Exception:
            pass

        return None

    @staticmethod
    def _parse_node_roles(config) -> Optional[List[str]]:
        env_roles = os.environ.get("MONARCH_NODE_ROLES", "")
        if env_roles:
            return [r.strip() for r in env_roles.split(",") if r.strip()]

        try:
            roles = config.cluster.get("monarch_node_roles", None)
            if roles:
                return list(roles)
        except Exception:
            pass

        return None

    def summary(self) -> str:
        p = self.placement
        lines = [
            f"Cluster: {self.n_nodes} node(s), "
            f"{self.n_devices_per_node} devices/node",
            f"  Inference: {p.inference.total_devices} devices "
            f"on {len(p.inference.nodes)} node(s) "
            f"{p.inference.all_device_ids}",
            f"  Training:  {p.training.total_devices} devices "
            f"on {len(p.training.nodes)} node(s) "
            f"{p.training.all_device_ids}",
            f"  MASTER_ADDR: {p.master_addr}",
            f"  Multi-node: {p.is_multi_node}",
        ]
        return "\n".join(lines)
