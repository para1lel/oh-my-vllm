#!/usr/bin/env python3
"""Explicit entry point for the optional, heavy real-agent stability collector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.agentic_soak import main

if __name__ == "__main__":
    main()
