from siteloom.dispatch import Job, LocalBackend


class EchoModule:
    def process(self, job):
        return {"echo": job.payload["value"]}


class BoomModule:
    def process(self, job):
        raise RuntimeError("boom")


def test_submit_and_wait_roundtrip():
    backend = LocalBackend()
    backend.register("echo", EchoModule())
    result = backend.submit_and_wait(Job(module="echo", payload={"value": 42}))
    assert result.ok
    assert result.result == {"echo": 42}


def test_unregistered_module_fails_gracefully():
    backend = LocalBackend()
    result = backend.submit_and_wait(Job(module="nope", payload={}))
    assert not result.ok
    assert "no module registered" in result.error


def test_module_exception_captured_not_raised():
    backend = LocalBackend()
    backend.register("boom", BoomModule())
    result = backend.submit_and_wait(Job(module="boom", payload={}))
    assert not result.ok
    assert "RuntimeError: boom" in result.error


def test_submit_batch_preserves_order():
    backend = LocalBackend()
    backend.register("echo", EchoModule())
    jobs = [Job(module="echo", payload={"value": i}) for i in range(5)]
    handles = backend.submit_batch(jobs)
    values = [h.wait().result["echo"] for h in handles]
    assert values == [0, 1, 2, 3, 4]
