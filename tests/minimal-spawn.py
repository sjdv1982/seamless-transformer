from seamless_transformer import spawn, shutdown_workers

if __name__ == "__main__":
    spawn(5)
    print("STOP")
    shutdown_workers()
