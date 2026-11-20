"""End-to-end tests for the ``phaseveil`` argparse CLI (in-process)."""

from __future__ import annotations

import json
import threading
import time

from typing import List

import numpy as np
import pytest

from phaseveil.cli import main
from phaseveil.crypto import make_blob
from phaseveil import audio_stego
from phaseveil.timing_channel import encode_transmission
from phaseveil.detector import VERDICT_CLEAN, VERDICT_SUSPICIOUS

from conftest import synth_cover, write_wav

TEXT = "CLI covert payload for end-to-end testing."
PASS = "cli-passphrase"


@pytest.fixture()
def wav(tmp_path):
    rate, samples = synth_cover(seconds=2.0)
    path = tmp_path / "cover.wav"
    write_wav(path, rate, samples)
    return path


# ---------------------------------------------------------------------------
# audio hide / extract
# ---------------------------------------------------------------------------


def test_cli_audio_hide_and_extract(tmp_path, wav, capsys) -> None:
    out = tmp_path / "stego.wav"
    code = main(["audio", "hide", "--in", str(wav), "--out", str(out), "--text", TEXT])
    assert code == 0
    captured = capsys.readouterr().out
    assert "SNR" in captured

    code = main(["audio", "extract", "--in", str(out)])
    assert code == 0
    assert TEXT in capsys.readouterr().out


def test_cli_audio_encrypted(tmp_path, wav, capsys) -> None:
    out = tmp_path / "stego.wav"
    code = main(
        ["audio", "hide", "--in", str(wav), "--out", str(out), "--text", TEXT,
         "--passphrase", PASS]
    )
    assert code == 0
    assert "AES-256-GCM" in capsys.readouterr().out

    code = main(["audio", "extract", "--in", str(out), "--passphrase", PASS])
    assert code == 0
    assert TEXT in capsys.readouterr().out


def test_cli_audio_extract_wrong_passphrase_fails(tmp_path, wav, capsys) -> None:
    out = tmp_path / "stego.wav"
    main(["audio", "hide", "--in", str(wav), "--out", str(out), "--text", TEXT,
          "--passphrase", PASS])
    capsys.readouterr()
    code = main(["audio", "extract", "--in", str(out), "--passphrase", "nope"])
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_audio_binary_file_payload(tmp_path) -> None:
    # a 256-byte payload needs two RS codewords -> use a 3 s cover
    rate, samples = synth_cover(seconds=3.0)
    cover = tmp_path / "cover3s.wav"
    write_wav(cover, rate, samples)
    blob = tmp_path / "blob.bin"
    data = bytes(range(256))
    blob.write_bytes(data)
    out = tmp_path / "stego.wav"
    assert main(["audio", "hide", "--in", str(cover), "--out", str(out), "--file", str(blob)]) == 0
    dest = tmp_path / "recovered.bin"
    assert main(["audio", "extract", "--in", str(out), "--out", str(dest)]) == 0
    assert dest.read_bytes() == data


def test_cli_audio_hide_reports_metrics(tmp_path, wav, capsys) -> None:
    out = tmp_path / "stego.wav"
    main(["audio", "hide", "--in", str(wav), "--out", str(out), "--text", "x" * 50])
    out_text = capsys.readouterr().out
    for token in ("SNR", "PSNR", "capacity", "used bins", "erasures", "RS(255"):
        assert token in out_text, token


def test_cli_missing_payload_args_fails(tmp_path, wav, capsys) -> None:
    """argparse rejects a missing --text/--file source with exit code 2."""
    with pytest.raises(SystemExit) as exc:
        main(["audio", "hide", "--in", str(wav), "--out", str(tmp_path / "o.wav")])
    assert exc.value.code == 2


def test_cli_bad_input_fails_cleanly(tmp_path, capsys) -> None:
    code = main(["audio", "hide", "--in", str(tmp_path / "missing.wav"),
                 "--out", str(tmp_path / "o.wav"), "--text", "abc"])
    assert code == 2
    assert "phaseveil: error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# analyze audio
# ---------------------------------------------------------------------------


def test_cli_analyze_audio_clean(tmp_path, wav, capsys) -> None:
    code = main(["analyze", "audio", "--in", str(wav)])
    assert code == 0
    text = capsys.readouterr().out
    assert VERDICT_CLEAN in text
    assert "VERDICT" in text


def test_cli_analyze_audio_stego(tmp_path, wav, capsys) -> None:
    stego = tmp_path / "stego.wav"
    main(["audio", "hide", "--in", str(wav), "--out", str(stego), "--text", "secret payload " * 3])
    capsys.readouterr()
    code = main(["analyze", "audio", "--in", str(stego)])
    assert code == 0
    text = capsys.readouterr().out
    assert VERDICT_SUSPICIOUS in text


def test_cli_analyze_audio_with_baseline(tmp_path, wav, capsys) -> None:
    stego = tmp_path / "stego.wav"
    main(["audio", "hide", "--in", str(wav), "--out", str(stego), "--text", "payload " * 4])
    capsys.readouterr()
    code = main(["analyze", "audio", "--in", str(stego), "--baseline", str(wav)])
    assert code == 0
    text = capsys.readouterr().out
    assert "baseline delta" in text
    assert VERDICT_SUSPICIOUS in text


# ---------------------------------------------------------------------------
# timing send / listen
# ---------------------------------------------------------------------------


def _free_udp_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _wait_for_listener(holder: dict, timeout: float = 60.0) -> None:
    """Block until the listener thread finished (it exits on its idle timeout)."""
    deadline = time.time() + timeout
    while "code" not in holder and time.time() < deadline:
        time.sleep(0.02)
    assert "code" in holder, "listener thread did not finish"


def _cli_listen(port: int, extra: List[str], holder: dict) -> threading.Event:
    """Start the listener subcommand in a thread; returns its readiness event.

    The listener sets the event as soon as its socket is bound, so the sender
    can start immediately afterwards without losing the leading packets.
    """
    import threading

    from phaseveil import timing_channel

    listening = threading.Event()

    def _listen() -> None:
        orig = timing_channel.receive_udp

        def _patched(*a, **kw):
            kw["ready"] = listening
            return orig(*a, **kw)

        timing_channel.receive_udp = _patched
        try:
            holder["code"] = main(["timing", "listen", "--host", "127.0.0.1",
                                   "--port", str(port), *extra])
        finally:
            timing_channel.receive_udp = orig

    listener = threading.Thread(target=_listen, daemon=True)
    listener.start()
    return listening


def test_cli_timing_send_and_listen(tmp_path, capsys) -> None:
    """Live loopback exchange through both CLI subcommands."""
    import threading

    port = _free_udp_port()
    trace = tmp_path / "trace.json"
    out = tmp_path / "received.bin"
    text = "cli live loopback payload"
    holder: dict = {}

    listening = _cli_listen(
        port, ["--delay-ms", "4,12", "--save-trace", str(trace), "--out", str(out)], holder
    )
    assert listening.wait(timeout=10.0), "listener never bound"

    code = main(["timing", "send", "--host", "127.0.0.1", "--port", str(port),
                 "--delay-ms", "4,12", "--text", text])
    assert code == 0
    _wait_for_listener(holder)
    assert holder["code"] == 0
    assert trace.exists()
    assert out.read_bytes() == text.encode("utf-8")


def test_cli_timing_listen_encrypted(tmp_path, capsys) -> None:
    port = _free_udp_port()
    text = "encrypted cli timing payload"
    holder: dict = {}

    listening = _cli_listen(port, ["--delay-ms", "4,12", "--passphrase", PASS], holder)
    assert listening.wait(timeout=10.0), "listener never bound"

    main(["timing", "send", "--host", "127.0.0.1", "--port", str(port),
          "--delay-ms", "4,12", "--text", text, "--passphrase", PASS])
    _wait_for_listener(holder)
    assert holder["code"] == 0
    assert text in capsys.readouterr().out


# ---------------------------------------------------------------------------
# analyze timing / benchmark / version
# ---------------------------------------------------------------------------


def test_cli_analyze_timing(tmp_path, capsys) -> None:
    from phaseveil.timing_channel import save_trace

    rng = np.random.default_rng(1)
    cfg_delays = encode_transmission(
        make_blob(b"covert timing payload for cli analysis", None,
                  compress=True).blob,
        0x00,
        __import__("phaseveil.config", fromlist=["TimingConfig"]).TimingConfig(),
        __import__("phaseveil.config", fromlist=["ReedSolomonConfig"]).ReedSolomonConfig(),
    )
    arrival = np.cumsum(np.r_[0.0, cfg_delays + rng.uniform(-5, 5, size=len(cfg_delays))])
    trace = tmp_path / "covert.json"
    save_trace(str(trace), np.maximum.accumulate(arrival))
    code = main(["analyze", "timing", "--trace", str(trace)])
    assert code == 0
    text = capsys.readouterr().out
    assert VERDICT_SUSPICIOUS in text
    assert "bimodal R" in text


def test_cli_benchmark(tmp_path, capsys) -> None:
    json_out = tmp_path / "bench.json"
    code = main(["benchmark", "--json", str(json_out)])
    capsys.readouterr()
    assert code == 0
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert data["detector"]["audio_true_positives"] == 6
    assert data["detector"]["audio_false_positives"] == 0
    assert data["detector"]["timing_false_positives_poisson"] == 0
    assert data["detector"]["timing_false_positives_gaussian"] == 0
    assert data["peak_rss_mb"] < 150.0


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "phaseveil" in capsys.readouterr().out.lower()


def test_cli_no_subcommand_fails() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_help_mentions_disclaimer() -> None:
    with pytest.raises(SystemExit):
        main(["--help"])