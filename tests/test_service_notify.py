"""sd_notify, and the promise that it is never load-bearing (CLD-105).

`Type=notify` is opt-in, so the overwhelmingly common case is that
`$NOTIFY_SOCKET` is unset — every hand-run server, every test, every
launchd job. That path has to be a silent no-op, for the reason the MQTT
publisher degrades to a log line: telemetry must never break the thing it
is reporting on (NFR1).
"""

from __future__ import annotations

import socket

from siteloom.service import notify


def test_no_socket_is_a_silent_no_op(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert notify.ready() is False
    assert notify.stopping() is False


def test_a_dead_socket_does_not_raise(monkeypatch, tmp_path):
    """systemd went away mid-run; the server must not care."""
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "gone.sock"))
    assert notify.ready() is False


def test_ready_and_stopping_reach_the_socket(monkeypatch, tmp_path):
    path = tmp_path / "notify.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
        server.bind(str(path))
        server.settimeout(2)
        monkeypatch.setenv("NOTIFY_SOCKET", str(path))

        assert notify.ready() is True
        assert server.recv(64) == b"READY=1"

        assert notify.stopping() is True
        assert server.recv(64) == b"STOPPING=1"


def test_abstract_namespace_prefix_is_translated(monkeypatch):
    """systemd spells an abstract socket '@name'; the sockaddr wants a
    leading NUL. Getting this wrong looks like a socket that is simply
    not there, which is indistinguishable from the no-op case."""
    name = "\0siteloom-test-notify"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
        server.bind(name)
        server.settimeout(2)
        monkeypatch.setenv("NOTIFY_SOCKET", "@" + name[1:])
        assert notify.ready() is True
        assert server.recv(64) == b"READY=1"
