from seamless.transformer.util import in_spawned_main_toplevel, is_spawned


def func():
    spawned = is_spawned()
    top_level = in_spawned_main_toplevel()
    print("FUNC", "spawned process:", spawned)
    print("From spawned toplevel", top_level)
