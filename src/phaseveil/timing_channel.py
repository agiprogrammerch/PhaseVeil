"""Network inter-packet-delay (IPD) covert channel.

Architecture
------------
The module is split deliberately in two halves so the *codec* is testable
without sockets and OS noise:

* :func:`encode_transmission` / :func:`decode_transmission` - pure functions
  mapping bytes <-> millisecond inter-arrival delay sequences.  Everything is
  deterministic, so unit tests can inject exact OS-scheduling jitter.
* :func:`transmit_udp` / :func:`transmit_tcp` / :func:`receive_udp` /
  :func:`receive_tcp` - live loopback (``127.0.0.1``) socket I/O for
  authorised lab experiments.  The transmitted datagrams are a fixed-size
  benign heartbeat; **all** covert information lives in the *gaps* between
  consecutive packets.

Bit coding
----------
A 0 bit is a short gap (``bit0_ms``, default 20 ms), a 1 bit a long gap
(``bit1_ms``, default 60 ms).  The transmitted bit stream is::

    [preamble: 16 alternating bits][RS-protected PhaseVeil packet bytes]

The preamble lets the receiver synchronise before any clock bias can matter.
Decoding is *adaptive*: the decision threshold is searched over candidate
values and chosen by preamble match, then refined to the midpoint of the two
cluster means estimated from the preamble itself, so the decoder needs no
assumption about the sender's absolute clock.

Safety: by default every socket is bound to / connected to ``127.0.0.1`` and
only unprivileged UDP/TCP sockets are used.  No raw packets, no CAP_NET_ADMIN.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ReedSolomonConfig, TimingConfig
from .payload import embed_payload, parse_embedded

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TimingError(Exception):
    """Base class for timing-channel failures."""


class SyncError(TimingError):
    """Raised when the preamble cannot be located in a trace."""


# ---------------------------------------------------------------------------
# Pure codec: bits <-> delays
# ---------------------------------------------------------------------------


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Pack bytes MSB-first into a bit array."""
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    return (
        ((arr[:, None] >> np.arange(7, -1, -1, dtype=np.uint8)) & 1)
        .astype(np.uint8)
        .reshape(-1)
    )


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    """Inverse of :func:`bytes_to_bits`; trailing partial byte dropped."""
    n = (len(bits) // 8) * 8
    if n == 0:
        return b""
    b = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, 8)
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint8)).astype(np.int32)
    return (b.astype(np.int32) * weights).sum(axis=1).astype(np.uint8).tobytes()


def build_bitstream(payload: bytes, flags: int, cfg: TimingConfig, rs_cfg: ReedSolomonConfig) -> np.ndarray:
    """Full transmitted bit sequence: preamble + RS-protected packet bytes."""
    stream = embed_payload(payload, flags, rs_cfg)
    return np.concatenate(
        [np.asarray(cfg.preamble_bits, dtype=np.uint8), bytes_to_bits(stream)]
    )


def bits_to_delays(bits: Sequence[int], cfg: TimingConfig) -> np.ndarray:
    """Map a bit sequence to the target inter-arrival delays in milliseconds."""
    b = np.asarray(bits, dtype=np.uint8)
    return np.where(b == 1, cfg.bit1_ms, cfg.bit0_ms).astype(np.float64)


def encode_transmission(
    payload: bytes, flags: int, cfg: TimingConfig, rs_cfg: ReedSolomonConfig
) -> np.ndarray:
    """Encode a payload blob into the list of inter-arrival delays (ms)."""
    bits = build_bitstream(payload, flags, cfg, rs_cfg)
    return bits_to_delays(bits, cfg)


# ---------------------------------------------------------------------------
# Adaptive decoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeResult:
    """Everything the receiver learned from a timestamp trace."""

    payload: bytes
    flags: int
    codewords: int
    bits: np.ndarray          # decoded bit stream (preamble + data)
    iats_ms: np.ndarray       # the inter-arrival times used for decoding
    threshold_ms: float       # final decision threshold
    symbol0_ms: float         # estimated mean IAT of 0 bits
    symbol1_ms: float         # estimated mean IAT of 1 bits
    preamble_index: int       # index of the first payload bit
    preamble_matches: int     # matching preamble bits at the chosen threshold
    erasure_fraction: float   # fraction of bytes handed to RS as erasures


def _preamble_align(bits: np.ndarray, preamble: Sequence[int], window: int) -> Tuple[int, int]:
    """Best ``(start, matches)`` alignment of ``preamble`` inside ``bits[:window]``.

    Ties prefer the earliest start so the search is deterministic.
    """
    pre = np.asarray(preamble, dtype=np.uint8)
    n = len(pre)
    best_start, best_matches = -1, -1
    for start in range(0, max(1, window - n + 1)):
        seg = bits[start : start + n]
        if len(seg) < n:
            break
        matches = int(np.sum(seg == pre))
        if matches > best_matches:
            best_start, best_matches = start, matches
    return best_start, best_matches


def adaptive_threshold(
    iats_ms: np.ndarray, cfg: TimingConfig, fade: int = 120
) -> Tuple[float, int, int, float, float]:
    """Search the decision threshold using the known preamble.

    Returns ``(threshold, preamble_index, matches, symbol0_ms, symbol1_ms)``.
    The search is a deterministic scan; for every candidate threshold the bits
    are thresholded and the preamble is re-aligned.  Among the top-scoring
    candidates the *median* threshold is taken (a plateau, not a knife edge),
    then refined to the midpoint of the two preamble cluster means.
    """
    iats = np.asarray(iats_ms, dtype=np.float64)
    n = len(cfg.preamble_bits)
    search_window = min(len(iats), fade)
    if search_window < n:
        raise SyncError("trace too short to contain a preamble")

    hi = max(cfg.bit1_ms + cfg.jitter_tolerance_ms + 5.0, float(np.median(iats)) * 1.5)
    best_score = -1
    best_thresholds: List[float] = []
    best_start = -1
    best_m0, best_m1 = cfg.bit0_ms, cfg.bit1_ms
    t = 1.0
    while t <= hi:
        bits = (iats[:search_window] > t).astype(np.uint8)
        start, matches = _preamble_align(bits, cfg.preamble_bits, search_window)
        m0 = m1 = float("nan")
        if start >= 0:
            pre_vals = iats[start : start + n]
            zeros = pre_vals[pre_vals < t]
            ones = pre_vals[pre_vals >= t]
            if len(zeros) and len(ones):
                m0, m1 = float(np.mean(zeros)), float(np.mean(ones))
        if matches > best_score:
            best_score = matches
            best_thresholds = [t]
            if start >= 0 and m0 == m0:
                best_start, best_m0, best_m1 = start, m0, m1
        elif matches == best_score:
            best_thresholds.append(t)
            if start >= 0 and m0 == m0 and best_start < 0:
                best_start, best_m0, best_m1 = start, m0, m1
        t += 0.5

    if best_score < n - 2:
        raise SyncError(f"preamble not found (best match {best_score}/{n})")
    plateau = float(np.median(best_thresholds))
    # Refine towards the midpoint of the two cluster means estimated from the
    # preamble.  The guard band keeps the threshold inside the middle 30% of
    # the estimated gap, which works for ANY symbol pair (the listener's config
    # may differ from the sender's, e.g. custom --delay-ms) and is bounded by
    # the observed clusters rather than by an assumed nominal value.
    gap = best_m1 - best_m0
    if gap > 0.5:
        lower = min(best_m0 + 0.35 * gap, best_m1 - 0.35 * gap)
        upper = max(best_m0 + 0.35 * gap, best_m1 - 0.35 * gap)
        refined = float(np.clip(0.5 * (best_m0 + best_m1), lower, upper))
    else:
        refined = plateau
    return refined, best_start, best_score, best_m0, best_m1


def unreliability_mask(
    iats_ms: np.ndarray, symbol0_ms: float, symbol1_ms: float, cfg: TimingConfig
) -> np.ndarray:
    """Per-bit reliability: bits far from either cluster mean are suspect.

    These bits are turned into Reed-Solomon *erasures*, which the decoder
    repairs for free (up to ``nsym`` per codeword).  The cut-off is the larger
    of ``3.5 * jitter_tolerance_ms`` and four robust sigmas of the observed
    residuals, so a channel whose *actual* jitter is wider than the configured
    design figure does not blanket-mark every byte as an erasure (which would
    exhaust the RS budget) while genuinely wild packets still are.
    """
    iats = np.asarray(iats_ms, dtype=np.float64)
    dist = np.minimum(np.abs(iats - symbol0_ms), np.abs(iats - symbol1_ms))
    if iats.size:
        # robust scale from the median absolute deviation of the residuals
        sigma = 1.4826 * float(np.median(dist)) if iats.size else 0.0
    else:
        sigma = 0.0
    cutoff = max(3.5 * cfg.jitter_tolerance_ms, 4.0 * sigma, 1e-3)
    return dist > cutoff


def decode_transmission(
    iats_ms: Sequence[float],
    cfg: TimingConfig,
    rs_cfg: ReedSolomonConfig,
    *,
    require_preamble: bool = True,
) -> DecodeResult:
    """Decode an inter-arrival-time sequence back into the payload blob.

    ``iats_ms`` holds the gaps between *consecutive received packets*, in ms.
    """
    iats = np.asarray(iats_ms, dtype=np.float64)
    if iats.size < (len(cfg.preamble_bits) + 8):
        raise SyncError(f"trace too short: {iats.size} IATs")

    if not require_preamble:
        threshold = cfg.threshold_ms
        bits_all = (iats > threshold).astype(np.uint8)
        start, matches = 0, len(cfg.preamble_bits)
        m0, m1 = cfg.bit0_ms, cfg.bit1_ms
    else:
        threshold, start, matches, m0, m1 = adaptive_threshold(iats, cfg)
        bits_all = (iats > threshold).astype(np.uint8)

    payload_bits = bits_all[start + len(cfg.preamble_bits) :]
    if payload_bits.size < 64:
        raise SyncError("no payload bits after the preamble")

    unreliable = unreliability_mask(iats, m0, m1, cfg)
    erasure_rows = unreliable[start + len(cfg.preamble_bits) :]

    n_bytes_bits = (payload_bits.size // 8) * 8
    payload_bits = payload_bits[:n_bytes_bits]
    stream = bits_to_bytes(payload_bits)
    # Per-byte erasure flag: any unreliable bit in the byte marks the byte.
    mask = [
        bool(np.any(erasure_rows[i : i + 8])) for i in range(0, n_bytes_bits, 8)
    ]
    try:
        payload, flags, codewords = parse_embedded(stream, mask, rs_cfg)
    except Exception as exc:
        raise TimingError(f"no PhaseVeil packet recovered: {exc}") from exc

    return DecodeResult(
        payload=payload,
        flags=flags,
        codewords=codewords,
        bits=bits_all,
        iats_ms=iats,
        threshold_ms=threshold,
        symbol0_ms=m0,
        symbol1_ms=m1,
        preamble_index=start,
        preamble_matches=matches if require_preamble else len(cfg.preamble_bits),
        erasure_fraction=float(np.mean(mask)) if mask else 0.0,
    )


# ---------------------------------------------------------------------------
# Deterministic jitter simulation (test support)
# ---------------------------------------------------------------------------


def simulate_jitter(
    delays: Sequence[float], rng: np.random.Generator, jitter_ms: float
) -> np.ndarray:
    """Add i.i.d. uniform jitter ``+/- jitter_ms`` to a delay sequence (ms)."""
    d = np.asarray(delays, dtype=np.float64)
    if jitter_ms <= 0:
        return d.copy()
    return d + rng.uniform(-jitter_ms, jitter_ms, size=d.shape)


# ---------------------------------------------------------------------------
# Live loopback I/O (authorised lab use only: 127.0.0.1, unprivileged sockets)
# ---------------------------------------------------------------------------

_TRACE_SCHEMA = "phaseveil-trace/1"


@dataclass
class SendSummary:
    """Result of a live transmission."""

    packets: int
    duration_ms: float
    delays_ms: List[float] = field(default_factory=list)
    protocol: str = "udp"


def _sleep_until(target: float) -> None:
    """Spin/sleep until ``target`` (``time.perf_counter`` clock).

    Bulk sleeping keeps CPU use low; the final sub-millisecond is a spin wait
    so OS scheduler granularity cannot blur the covert timing.
    """
    while True:
        now = time.perf_counter()
        remaining = target - now
        if remaining <= 0:
            return
        if remaining > 0.0015:
            time.sleep(remaining - 0.0015)


def transmit_udp(
    host: str,
    port: int,
    delays_ms: Sequence[float],
    cfg: TimingConfig,
    *,
    timeout_ms: Optional[float] = None,
    recorder: Optional[List[float]] = None,
) -> SendSummary:
    """Send heartbeats over UDP, encoding bits in the inter-packet gaps.

    Packet 0 leaves immediately; packet *i* is scheduled at
    ``t0 + sum(delays[:i])`` so that the gap observed by the receiver between
    packets ``i-1`` and ``i`` equals ``delays[i-1]`` - exactly the value the
    decoder expects (``iats[k]`` carries bit ``k``).
    """
    addr = (host, int(port))
    heartbeat = bytes(cfg.heartbeat_bytes) or b"\x00"
    timeout = (timeout_ms if timeout_ms is not None else cfg.send_socket_timeout_ms) / 1000.0
    sent: List[float] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        t0 = time.perf_counter()
        target = t0
        sock.sendto(heartbeat, addr)
        sent.append(time.perf_counter())
        for d in delays_ms:
            target += float(d) / 1000.0
            _sleep_until(target)
            sock.sendto(heartbeat, addr)
            sent.append(time.perf_counter())
        duration = (time.perf_counter() - t0) * 1000.0
    if recorder is not None:
        recorder.extend(sent)
    return SendSummary(
        packets=len(sent), duration_ms=duration, delays_ms=[float(d) for d in delays_ms]
    )


def transmit_tcp(
    host: str,
    port: int,
    delays_ms: Sequence[float],
    cfg: TimingConfig,
    *,
    timeout_ms: Optional[float] = None,
    recorder: Optional[List[float]] = None,
) -> SendSummary:
    """Send heartbeats over TCP (``TCP_NODELAY`` defeats Nagle coalescing).

    Packet 0 leaves immediately; see :func:`transmit_udp` for the gap contract.
    """
    addr = (host, int(port))
    heartbeat = bytes(cfg.heartbeat_bytes) or b"\x00"
    timeout = (timeout_ms if timeout_ms is not None else cfg.send_socket_timeout_ms) / 1000.0
    sent: List[float] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect(addr)
        t0 = time.perf_counter()
        target = t0
        sock.sendall(heartbeat)
        sent.append(time.perf_counter())
        for d in delays_ms:
            target += float(d) / 1000.0
            _sleep_until(target)
            sock.sendall(heartbeat)
            sent.append(time.perf_counter())
        duration = (time.perf_counter() - t0) * 1000.0
    if recorder is not None:
        recorder.extend(sent)
    return SendSummary(
        packets=len(sent), duration_ms=duration, protocol="tcp",
        delays_ms=[float(d) for d in delays_ms],
    )


def receive_udp(
    host: str,
    port: int,
    cfg: TimingConfig,
    *,
    idle_timeout_ms: Optional[float] = None,
    max_seconds: Optional[float] = None,
    recorder: Optional[List[float]] = None,
    ready: Optional[threading.Event] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Receive UDP heartbeats on ``127.0.0.1``; returns ``(arrival_ms, clock)``.

    ``arrival_ms`` are arrival timestamps relative to the first packet (the
    covert channel's own time base); ``clock`` is the raw
    ``perf_counter`` list for trace export.  If ``ready`` is given it is set as
    soon as the socket is bound, so a caller can start a sender without losing
    the leading packets (a plain ``sleep`` is a race on a loaded machine).
    """
    idle = (idle_timeout_ms if idle_timeout_ms is not None else cfg.listen_idle_timeout_ms) / 1000.0
    max_s = max_seconds if max_seconds is not None else cfg.listen_max_seconds
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
        if ready is not None:
            ready.set()
        sock.settimeout(idle)
        clock: List[float] = []
        start = time.perf_counter()
        while True:
            if time.perf_counter() - start > max_s:
                break
            try:
                data, _addr = sock.recvfrom(cfg.heartbeat_bytes + 64)
            except socket.timeout:
                break
            clock.append(time.perf_counter())
    if not clock:
        return np.empty(0, dtype=np.float64), clock
    arrival = [(t - clock[0]) * 1000.0 for t in clock]
    if recorder is not None:
        recorder.extend(clock)
    return np.asarray(arrival, dtype=np.float64), clock


def receive_tcp(
    host: str,
    port: int,
    cfg: TimingConfig,
    *,
    idle_timeout_ms: Optional[float] = None,
    max_seconds: Optional[float] = None,
    recorder: Optional[List[float]] = None,
    ready: Optional[threading.Event] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Receive TCP heartbeats on ``127.0.0.1``; returns ``(arrival_ms, clock)``.

    ``ready`` is set once the listening socket exists; the kernel backlog keeps
    an early connection from being lost while ``accept`` still runs.
    """
    idle = (idle_timeout_ms if idle_timeout_ms is not None else cfg.listen_idle_timeout_ms) / 1000.0
    max_s = max_seconds if max_seconds is not None else cfg.listen_max_seconds
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, int(port)))
        srv.listen(1)
        if ready is not None:
            ready.set()
        srv.settimeout(max_s)
        deadline = time.perf_counter() + max_s
        conn, _addr = srv.accept()
        conn.settimeout(idle)
        clock: List[float] = []
        while True:
            if time.perf_counter() > deadline:
                break
            try:
                data = conn.recv(cfg.heartbeat_bytes + 64)
            except socket.timeout:
                break
            if not data:
                break
            clock.append(time.perf_counter())
        conn.close()
    if not clock:
        return np.empty(0, dtype=np.float64), clock
    arrival = np.asarray([(t - clock[0]) * 1000.0 for t in clock], dtype=np.float64)
    if recorder is not None:
        recorder.extend(clock)
    return arrival, clock


def arrival_to_iats(arrival_ms: Sequence[float]) -> np.ndarray:
    """Inter-arrival times (ms) from absolute arrival times (ms)."""
    a = np.asarray(arrival_ms, dtype=np.float64)
    if a.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(a)


def save_trace(
    path: str | Path,
    arrival_ms: Sequence[float],
    *,
    protocol: str = "udp",
    host: str = "127.0.0.1",
    port: int = 0,
    meta: Optional[Dict[str, object]] = None,
) -> Path:
    """Persist a capture for offline steganalysis."""
    arr = [float(x) for x in arrival_ms]
    doc = {
        "schema": _TRACE_SCHEMA,
        "protocol": protocol,
        "host": host,
        "port": int(port),
        "packets": len(arr),
        "time_base": "time.perf_counter() relative to first packet",
        "arrival_ms": arr,
        "iats_ms": [float(x) for x in arrival_to_iats(arr)],
    }
    if meta:
        doc["meta"] = meta
    out = Path(path)
    out.write_text(json.dumps(doc, indent=2))
    return out


def load_trace(path: str | Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load a trace saved by :func:`save_trace`; returns ``(iats_ms, metadata)``."""
    doc = json.loads(Path(path).read_text())
    if "iats_ms" in doc and doc["iats_ms"]:
        iats = np.asarray(doc["iats_ms"], dtype=np.float64)
    else:
        iats = arrival_to_iats(doc["arrival_ms"])
    meta = {k: v for k, v in doc.items() if k not in ("iats_ms", "arrival_ms")}
    return iats, meta


def send_payload_blob(
    host: str,
    port: int,
    blob: bytes,
    flags: int,
    cfg: TimingConfig,
    rs_cfg: ReedSolomonConfig,
    *,
    protocol: str = "udp",
    timeout_ms: Optional[float] = None,
) -> Tuple[SendSummary, np.ndarray]:
    """Convenience wrapper: encode a blob and transmit it live over loopback.

    Returns ``(send_summary, delays_ms)``.  The listener side calls
    :func:`receive_udp`/:func:`receive_tcp` and then
    :func:`decode_transmission` on the measured inter-arrival times.
    """
    delays = encode_transmission(blob, flags, cfg, rs_cfg)
    if protocol == "tcp":
        summary = transmit_tcp(host, port, delays, cfg, timeout_ms=timeout_ms)
    else:
        summary = transmit_udp(host, port, delays, cfg, timeout_ms=timeout_ms)
    return summary, np.asarray(delays, dtype=np.float64)