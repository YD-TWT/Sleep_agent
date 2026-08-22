from __future__ import annotations

import logging
import os

logger = logging.getLogger("gdesigner")

def agent_debug_print(*args, **kwargs) -> None:
    if os.environ.get("GDESIGNER_VERBOSE", "").lower() in ("1", "true", "yes"):
        print(*args, **kwargs)
