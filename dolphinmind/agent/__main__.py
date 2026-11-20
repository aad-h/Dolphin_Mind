"""Allow ``python -m dolphinmind.agent "question" --doc book.txt``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
