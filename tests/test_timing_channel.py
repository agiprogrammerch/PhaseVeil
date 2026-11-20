"""Tests for the inter-packet-delay (IPD) covert channel codec.

Pure codec tests inject deterministic jitter; live tests use ephemeral
``127.0.0.1`` UDP/TCP sockets.
"""

from __future__ import annotations

import numpy as np
import pytest
import time

from phaseveil.config import PhaseVeilConfig, TimingConfig
from phaseveil.crypto import make_blob, parse_blob
from phaseveil.payload import rs_encode_stream, frame_packet
from phaseveil.timing_channel import (
    SyncError,
    TimingError,
    adaptive_threshold,
    arrival_to_iats,
    bits_to_bytes,
    bits_to_delays,
    bytes_to_bits,
    build_bitstream,
    decode_transmission,
    encode_transmission,
    load_trace,
    receive_tcp,
    receive_udp,
    save_trace,
    simulate_jitter,
    transmit_tcp,
    transmit_udp,
)

from conftest import free_tcp_port, free_udp_port

PAYLOAD = b"Timing channel covert payload of moderate length."
PASS = "lab passphrase"


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------


def test_bit_packing_round_trip() -> None:
    data = bytes(range(37))
    assert bits_to_bytes(bytes_to_bits(data)) == data


def test_bits_to_delays_uses_cfg_symbols() -> None:
    cfg = TimingConfig()
    d = bits_to_delays([0, 1, 0, 1], cfg)
    assert list(d) == [cfg.bit0_ms, cfg.bit1_ms] * 2


def test_preamble_is_recoverable_from_encoded_stream(cfg) -> None:
    stream = build_bitstream(b"payload", 0x00, cfg.timing, cfg.rs)
    assert bytes_to_bits(b"payload")[: len(cfg.timing.preamble_bits)] is not None
    pre = np.asarray(stream[: len(cfg.timing.preamble_bits)], dtype=np.uint8)
    assert list(pre) == list(cfg.timing.preamble_bits)


# ---------------------------------------------------------------------------
# Deterministic jitter round trips
# ---------------------------------------------------------------------------


def test_zero_jitter_round_trip(cfg) -> None:
    blob = make_blob(PAYLOAD, None, compress=True, cfg=cfg.crypto)
    delays = encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    result = decode_transmission(delays, cfg.timing, cfg.rs)
    assert parse_blob(result.payload, None, expect_flags=result.flags) == PAYLOAD
    assert result.threshold_ms == pytest.approx(cfg.timing.threshold_ms, abs=1.0)
    assert result.preamble_matches == len(cfg.timing.preamble_bits)


def test_encrypted_jitter_round_trip(cfg) -> None:
    blob = make_blob(PAYLOAD, PASS, compress=True, cfg=cfg.crypto)
    delays = encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    for seed in (1, 2, 3):
        iats = simulate_jitter(delays, np.random.default_rng(seed), 4.0)
        result = decode_transmission(iats, cfg.timing, cfg.rs)
        assert parse_blob(result.payload, PASS, expect_flags=result.flags) == PAYLOAD


def test_max_specified_jitter_8ms(cfg) -> None:
    blob = make_blob(PAYLOAD, None, compress=True, cfg=cfg.crypto)
    delays = encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    for seed in range(8):
        iats = simulate_jitter(delays, np.random.default_rng(seed), 8.0)
        result = decode_transmission(iats, cfg.timing, cfg.rs)
        assert parse_blob(result.payload, None, expect_flags=result.flags) == PAYLOAD


def test_jitter_above_design_envelope_fails_cleanly(cfg) -> None:
    """Jitter that swallows half the gap cannot be decoded; must not crash."""
    blob = make_blob(PAYLOAD, None, compress=True, cfg=cfg.crypto)
    delays = encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    iats = simulate_jitter(delays, np.random.default_rng(0), 30.0)
    with pytest.raises(Exception) as exc:
        decode_transmission(iats, cfg.timing, cfg.rs)
    assert isinstance(exc.value, (TimingError, SyncError))


def test_adaptive_threshold_recovers_symbol_means(cfg) -> None:
    blob = make_blob(b"threshold estimation payload", None, compress=True, cfg=cfg.crypto)
    delays = encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    iats = simulate_jitter(delays, np.random.default_rng(11), 7.0)
    threshold, start, matches, m0, m1 = adaptive_threshold(iats, cfg.timing)
    assert start == 0
    assert matches >= 14
    assert m0 == pytest.approx(cfg.timing.bit0_ms, abs=cfg.timing.jitter_tolerance_ms)
    assert m1 == pytest.approx(cfg.timing.bit1_ms, abs=cfg.timing.jitter_tolerance_ms)
    assert min(m0, m1) < threshold < max(m0, m1)


def test_all_zero_payload_decodes(cfg) -> None:
    """Threshold refinement must not depend on seeing both symbols in the payload."""
    from phaseveil.payload import rs_encode_stream, frame_packet

    frame = frame_packet(b"\x00" * 3, 0x00)
    stream = rs_encode_stream(frame, cfg.rs)
    bits = np.concatenate(
        [np.asarray(cfg.timing.preamble_bits, dtype=np.uint8), bytes_to_bits(stream)]
    )
    delays = bits_to_delays(bits, cfg.timing)
    iats = simulate_jitter(delays, np.random.default_rng(5), 5.0)
    result = decode_transmission(iats, cfg.timing, cfg.rs)
    assert result.payload == b"\x00" * 3


def test_all_ones_payload_decodes(cfg) -> None:
    from phaseveil.payload import rs_encode_stream, frame_packet

    frame = frame_packet(b"\xff" * 3, 0x00)
    stream = rs_encode_stream(frame, cfg.rs)
    bits = np.concatenate(
        [np.asarray(cfg.timing.preamble_bits, dtype=np.uint8), bytes_to_bits(stream)]
    )
    delays = bits_to_delays(bits, cfg.timing)
    iats = simulate_jitter(delays, np.random.default_rng(6), 5.0)
    result = decode_transmission(iats, cfg.timing, cfg.rs)
    assert result.payload == b"\xff" * 3


def test_short_trace_raises_sync_error(cfg) -> None:
    with pytest.raises(SyncError):
        decode_transmission(np.array([20.0, 60.0, 20.0]), cfg.timing, cfg.rs)


def test_pure_noise_trace_rejected(cfg) -> None:
    iats = np.random.default_rng(0).exponential(35.0, size=300)
    with pytest.raises(Exception):
        decode_transmission(iats, cfg.timing, cfg.rs)


def test_iats_from_arrivals() -> None:
    iats = arrival_to_iats([0.0, 20.0, 80.0, 140.0])
    assert list(iats) == [20.0, 60.0, 60.0]


# ---------------------------------------------------------------------------
# Live loopback sockets
# ---------------------------------------------------------------------------


def _fast_timing(cfg: PhaseVeilConfig) -> PhaseVeilConfig:
    """Test-only symbol timing: the same codec, 10x faster on the wire.

    The production defaults (20 ms / 60 ms) would make a 1000-bit transmission
    take ~40 s of wall clock; tests use 2 ms / 6 ms symbols so the live socket
    suite stays quick while exercising identical encoding/decoding logic.
    """
    return cfg.with_(
        timing=TimingConfig(
            bit0_ms=4.0,
            bit1_ms=12.0,
            jitter_tolerance_ms=2.0,
            listen_idle_timeout_ms=400.0,
            listen_max_seconds=30.0,
        )
    )


def _live_round_trip(cfg, payload, passphrase, protocol: str, tmp_path) -> None:
    """Run a live loopback transmission: listener in a thread, sender inline."""
    import threading

    fast = _fast_timing(cfg)
    blob = make_blob(payload, passphrase, compress=True, cfg=fast.crypto)
    delays = encode_transmission(blob.blob, blob.flags, fast.timing, fast.rs)

    if protocol == "tcp":
        port = free_tcp_port()
    else:
        port = free_udp_port()

    holder: dict = {}

    def _listen() -> None:
        if protocol == "tcp":
            arrival, clock = receive_tcp("127.0.0.1", port, fast.timing)
        else:
            arrival, clock = receive_udp("127.0.0.1", port, fast.timing)
        holder["arrival"] = arrival

    listener = threading.Thread(target=_listen, daemon=True)
    listener.start()
    time.sleep(0.05)  # let the listener bind/serve before the first packet

    if protocol == "tcp":
        summary = transmit_tcp("127.0.0.1", port, delays, fast.timing)
    else:
        summary = transmit_udp("127.0.0.1", port, delays, fast.timing)
    listener.join(timeout=30.0)

    assert summary.packets == len(delays) + 1  # one leading packet, then len(delays) gaps
    arrival = holder.get("arrival")
    assert arrival is not None and arrival.size >= 10

    iats = arrival_to_iats(arrival)
    if tmp_path is not None:
        trace = tmp_path / "trace.json"
        save_trace(str(trace), arrival, protocol=protocol, host="127.0.0.1", port=port)
        assert trace.exists()
        assert '"iats_ms"' in trace.read_text()
    result = decode_transmission(iats, fast.timing, fast.rs)
    assert parse_blob(result.payload, passphrase, expect_flags=result.flags) == payload
    # measured symbol means should track the scaled nominal delays closely
    assert result.symbol0_ms == pytest.approx(fast.timing.bit0_ms, abs=3.0)
    assert result.symbol1_ms == pytest.approx(fast.timing.bit1_ms, abs=3.0)


def test_live_udp_round_trip(cfg, tmp_path) -> None:
    _live_round_trip(cfg, b"live udp payload", None, "udp", tmp_path)


def test_live_udp_encrypted_round_trip(cfg) -> None:
    _live_round_trip(cfg, PAYLOAD, PASS, "udp", None)


def test_live_tcp_round_trip(cfg) -> None:
    _live_round_trip(cfg, b"live tcp payload", None, "tcp", None)


def test_live_udp_trace_save_load(cfg, tmp_path) -> None:
    port = free_udp_port()
    fast = _fast_timing(cfg)
    blob = make_blob(b"trace save test payload", None, compress=True, cfg=fast.crypto)
    delays = encode_transmission(blob.blob, blob.flags, fast.timing, fast.rs)

    import threading

    holder: dict = {}

    def _listen() -> None:
        arrival, _clock = receive_udp("127.0.0.1", port, fast.timing)
        holder["arrival"] = arrival

    listener = threading.Thread(target=_listen, daemon=True)
    listener.start()
    time.sleep(0.05)
    transmit_udp("127.0.0.1", port, delays, fast.timing)
    listener.join(timeout=30.0)

    trace = tmp_path / "capture.json"
    save_trace(str(trace), holder["arrival"], protocol="udp", host="127.0.0.1", port=port)
    iats, meta = load_trace(trace)
    assert meta["protocol"] == "udp"
    assert meta["port"] == port
    result = decode_transmission(iats, fast.timing, fast.rs)
    assert parse_blob(result.payload, None, expect_flags=result.flags) == b"trace save test payload"


def test_receive_udp_with_no_traffic_returns_empty(cfg) -> None:
    """An idle listener terminates on its idle timeout rather than hanging."""
    port = free_udp_port()
    cfg_short = cfg.with_(
        timing=TimingConfig(listen_idle_timeout_ms=200.0, listen_max_seconds=2.0)
    )
    arrival, clock = receive_udp("127.0.0.1", port, cfg_short.timing)
    assert arrival.size == 0


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_timing_config_validation() -> None:
    with pytest.raises(ValueError):
        TimingConfig(bit0_ms=0)
    with pytest.raises(ValueError):
        TimingConfig(bit0_ms=60.0, bit1_ms=20.0)
    with pytest.raises(ValueError):
        TimingConfig(preamble_bits=(1, 0, 1))


def test_timing_config_defaults(cfg) -> None:
    assert cfg.timing.bit0_ms == 20.0
    assert cfg.timing.bit1_ms == 60.0
    assert cfg.timing.jitter_tolerance_ms == 8.0
    assert cfg.timing.threshold_ms == 40.0
    assert cfg.timing.jitter_headroom_ms == 20.0