"""Fix E: Endpoint._send falls back to a direct send after executor shutdown."""

import asyncio


def test_endpoint_send_falls_back_after_executor_shutdown():
    from seamless_transformer.process.channel import Endpoint

    sent = []

    class Conn:
        def send(self, message):
            sent.append(message)

    class Loop:
        def run_in_executor(self, *_a):
            raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    async def go():
        ep = object.__new__(Endpoint)
        ep._conn, ep._loop, ep._name = Conn(), Loop(), "ep"
        ep._send_lock = asyncio.Lock()
        await ep._send({"x": 1})

    asyncio.run(go())
    assert sent == [{"x": 1}]
