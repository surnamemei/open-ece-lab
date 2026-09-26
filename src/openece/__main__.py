"""``python -m openece``: same as the ``open-ece`` command (useful when scripts are not on PATH)."""
import sys

from .cli import main

sys.exit(main())
