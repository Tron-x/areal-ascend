"""Standalone network utilities for Forge (no areal dependency)."""

from __future__ import annotations

import socket


def gethostip() -> str:
    """Get the IP address of the current host."""
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return "127.0.0.1"


def find_free_port(low: int = 10000, high: int = 50000) -> int:
    """Find a single free TCP port in the given range."""
    import random

    for _ in range(100):
        port = random.randint(low, high)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range [{low}, {high}]")


def find_free_ports(
    count: int, port_range: tuple[int, int] = (10000, 50000)
) -> list[int]:
    """Find ``count`` free TCP ports."""
    ports = []
    for _ in range(count):
        port = find_free_port(port_range[0], port_range[1])
        ports.append(port)
    return ports
