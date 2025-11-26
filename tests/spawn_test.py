import os
import sys
import multiprocessing as mp
import seamless

# Ensure local test helpers are importable even when the test is invoked from elsewhere.
sys.path.insert(0, os.path.dirname(__file__))
from helpers import func


func()  # We want to block this from execution inside a spawned process.
#  We want to tell the user that it should have beein inside a 'if __name__ == "__main__"' wrapper


def main():
    print("MAIN", "spawned process:", mp.parent_process() is not None)
    func()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("SPAWN")
    spawned_process = mp.Process(target=main)
    spawned_process.start()
    spawned_process.join()

seamless.close()
