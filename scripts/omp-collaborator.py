#!/usr/bin/env python3
"""Launch an assigned collaborator Issue under the project's pricing policy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from development.collaboration import main

if __name__ == "__main__":
    main()
