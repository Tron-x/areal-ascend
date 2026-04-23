"""Allow ``python -m forge`` as an alias for ``python -m forge.cli``.

Users naturally type ``python -m forge launch config.yaml`` before
learning about the ``forge.cli`` sub-namespace; forwarding keeps both
paths working.
"""

from __future__ import annotations

from forge.cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
