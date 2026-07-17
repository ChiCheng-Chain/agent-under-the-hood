"""支持 `python -m agent_harness.cli ...`。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
