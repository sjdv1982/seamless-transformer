import os
import sys

# Ensure local test helpers are importable even when this module is imported from a spawned child.
sys.path.insert(0, os.path.dirname(__file__))
from helpers import func

func()
