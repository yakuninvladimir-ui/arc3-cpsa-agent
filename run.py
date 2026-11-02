#!/usr/bin/env python3
"""Root entrypoint to run the ARC-AGI-3 CPSA 2.3 Agent."""

import os
import sys
from pathlib import Path

# Add project root to sys.path so inference, taaf, and viewer import natively
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env.example")
load_dotenv(_root / ".env", override=True)

if __name__ == "__main__":
    from inference.framework.run import main
    sys.exit(main())
