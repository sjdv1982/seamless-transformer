import multiprocessing as mp
import os
import sys
import seamless

# Ensure local test helpers are importable regardless of invocation cwd.
sys.path.insert(0, os.path.dirname(__file__))


def main():
    import spawn_import_module  # noqa: F401


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=main)
    p.start()
    p.join()
    seamless.close()
    sys.exit(p.exitcode)
