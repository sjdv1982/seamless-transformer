import time
import seamless
import seamless.config
seamless.config.init()
import seamless_transformer
from seamless_transformer.transformation_class import _dask_available
from seamless.transformer import direct, delayed
print("seamless_transformer from", seamless_transformer.__file__)
print("dask_available:", _dask_available())

####

SALT = int(time.time() * 1000) % 100000

@direct
def add_direct(a, b):
    return a + b

@delayed
def add(a, b):
    return a + b

def report(label, func):
    t = time.perf_counter()
    try:
        r = func()
        print(f"{label}: OK {r!r} ({time.perf_counter()-t:.2f}s)")
    except Exception as exc:
        text = str(exc).strip()
        print(f"{label}: {type(exc).__name__}: {text.splitlines()[-1] if text else ''}")

####

report("direct", lambda: add_direct(2, SALT))

####

report("run", lambda: add(20, SALT).run())

####

report("dependency", lambda: add(add(1, SALT), add(3, 4)).run())

####

started = add(7, SALT).start()

####

report("start-then-run", lambda: started.run())

####

started2 = add(70, SALT).start()
print("start-then-await:", await started2.computation(), started2.value)

####

print("await task:", await add(700, SALT).task())

####

@delayed
def outer(label):
    from seamless.transformer import delayed, direct

    @direct
    def middle(label):
        from seamless.transformer import delayed

        def leaf(label):
            return label

        leaf = delayed(leaf)
        left = leaf(label + "-a").start()
        right = leaf(label + "-b").start()
        return left.run(), right.run()

    return middle(label + "-1"), middle(label + "-2")

def _nested():
    tasks = [outer(f"job-{SALT}-{idx}").start() for idx in range(4)]
    return len([tf.run() for tf in tasks])
report("nested x4", _nested)

####

seamless.close()
