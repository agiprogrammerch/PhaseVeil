"""``phaseveil`` command line interface (argparse, no third-party CLI frameworks).

Subcommands
-----------
``audio hide``     embed a text/file payload into a 16-bit PCM wav (phase coding)
``audio extract``  recover a payload from a stego wav
``timing send``    transmit a payload over a loopback inter-packet-delay channel
``timing listen``  receive a transmission over loopback and save the capture
``analyze audio`` statistical phase-spectrum steganalysis of a wav
``analyze timing`` statistical IAT steganalysis of a saved packet trace
``benchmark``      synthetic capacity/SNR/jitter/detector benchmark + RAM report

All network operations are restricted to unprivileged UDP/TCP sockets on
``127.0.0.1`` (authorised research lab use only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .config import PhaseVeilConfig, TimingConfig
from .crypto import make_blob, parse_blob
from .detector import analyze_audio_file, analyze_timing_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err(msg: str) -> int:
    print(f"phaseveil: error: {msg}", file=sys.stderr)
    return 2


def _table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    """Render a plain ASCII table."""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) for row in rows]
    return "\n".join([line, sep, *body])


def _payload_bytes(args: argparse.Namespace) -> bytes:
    if getattr(args, "text", None) is not None:
        return args.text.encode("utf-8")
    if getattr(args, "file", None) is not None:
        return Path(args.file).read_bytes()
    raise ValueError("either --text or --file must be given")


def _config_from_args(args: argparse.Namespace) -> PhaseVeilConfig:
    cfg = PhaseVeilConfig()
    if getattr(args, "delay_ms", None):
        try:
            b0, b1 = (float(x) for x in args.delay_ms.split(",")[:2])
        except ValueError:
            raise ValueError("--delay-ms expects 'BIT0,BIT1' in milliseconds")
        cfg = cfg.with_(timing=TimingConfig(bit0_ms=b0, bit1_ms=b1))
    return cfg


def _print(text: str) -> None:
    print(text, flush=True)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_audio_hide(args: argparse.Namespace) -> int:
    from . import audio_stego

    payload = _payload_bytes(args)
    cfg = _config_from_args(args)
    blob = make_blob(
        payload, getattr(args, "passphrase", None), compress=not getattr(args, "no_compress", False),
        cfg=cfg.crypto,
    )
    metrics = audio_stego.hide_file(
        args.input, args.output, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
    )
    _print(
        _table(
            [
                ("cover", Path(args.input).name),
                ("stego", Path(args.output).name),
                ("payload bytes", len(payload)),
                ("mode", "AES-256-GCM + zlib" if blob.encrypted else "plaintext (CRC32)"),
                ("FEC", f"RS({cfg.rs.codeword_length},{cfg.rs.data_symbols})"),
                ("embedded bytes", metrics.payload_bytes),
                ("capacity bytes", metrics.capacity_bytes),
                ("capacity used", f"{metrics.capacity_used * 100:.1f}%"),
                ("used bins", metrics.used_bins),
                ("chunks", f"{metrics.chunks_used}/{metrics.total_segments // cfg.audio.chunk_segments}"),
                ("SNR", f"{metrics.snr_db:.2f} dB"),
                ("PSNR", f"{metrics.psnr_db:.2f} dB"),
                ("erasures", f"{metrics.erasures_total} bits ({metrics.erasure_fraction * 100:.1f}%)"),
            ],
            ["phaseveil audio hide", "value"],
        )
    )
    return 0


def cmd_audio_extract(args: argparse.Namespace) -> int:
    from . import audio_stego

    cfg = _config_from_args(args)
    payload, flags, codewords = audio_stego.extract_payload(
        *_read_stego(args), audio_cfg=cfg.audio, rs_cfg=cfg.rs,
    )
    text = parse_blob(payload, getattr(args, "passphrase", None), expect_flags=flags)
    out = args.out
    if out:
        Path(out).write_bytes(text)
        _print(f"extracted {len(text)} bytes -> {out}")
    else:
        try:
            printable = text.decode("utf-8")
        except UnicodeDecodeError:
            printable = text.hex()
        _print(printable)
    _print(f"(payload {len(payload)} B, flags 0x{flags:02x}, {codewords} RS codewords)")
    return 0


def _read_stego(args: argparse.Namespace):
    from . import audio_stego

    rate, samples = audio_stego.read_wav(args.input)
    return samples, rate


def cmd_timing_send(args: argparse.Namespace) -> int:
    from . import timing_channel

    payload = _payload_bytes(args)
    cfg = _config_from_args(args)
    blob = make_blob(
        payload, getattr(args, "passphrase", None), compress=not getattr(args, "no_compress", False),
        cfg=cfg.crypto,
    )
    delays = timing_channel.encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    protocol = "tcp" if getattr(args, "tcp", False) else "udp"
    if protocol == "tcp":
        summary = timing_channel.transmit_tcp(args.host, args.port, delays, cfg.timing)
    else:
        summary = timing_channel.transmit_udp(args.host, args.port, delays, cfg.timing)
    _print(
        _table(
            [
                ("protocol", f"{protocol} -> {args.host}:{args.port}"),
                ("payload bytes", len(payload)),
                ("mode", "AES-256-GCM + zlib" if blob.encrypted else "plaintext (CRC32)"),
                ("packets sent", summary.packets),
                ("channel seconds", f"{summary.duration_ms / 1000.0:.3f}"),
                ("bit delay", f"{cfg.timing.bit0_ms:.0f} ms / {cfg.timing.bit1_ms:.0f} ms"),
                ("jitter tolerance", f"±{cfg.timing.jitter_tolerance_ms:.0f} ms"),
            ],
            ["phaseveil timing send", "value"],
        )
    )
    return 0


def cmd_timing_listen(args: argparse.Namespace) -> int:
    from . import timing_channel

    cfg = _config_from_args(args)
    protocol = "tcp" if getattr(args, "tcp", False) else "udp"
    if protocol == "tcp":
        arrival, _clock = timing_channel.receive_tcp(args.host, args.port, cfg.timing)
    else:
        arrival, _clock = timing_channel.receive_udp(args.host, args.port, cfg.timing)
    if arrival.size < 2:
        return _err("no packets captured (is a transmission running?)")
    if getattr(args, "save_trace", None):
        timing_channel.save_trace(
            args.save_trace, arrival, protocol=protocol, host=args.host, port=args.port
        )
        _print(f"saved packet trace -> {args.save_trace} ({arrival.size} packets)")
    iats = timing_channel.arrival_to_iats(arrival)
    result = timing_channel.decode_transmission(iats, cfg.timing, cfg.rs)
    text = parse_blob(
        result.payload, getattr(args, "passphrase", None), expect_flags=result.flags
    )
    out = args.out
    if out:
        Path(out).write_bytes(text)
        _print(f"extracted {len(text)} bytes -> {out}")
    else:
        try:
            printable = text.decode("utf-8")
        except UnicodeDecodeError:
            printable = text.hex()
        _print(printable)
    _print(
        f"(sync at bit {result.preamble_index}, threshold {result.threshold_ms:.1f} ms, "
        f"symbols {result.symbol0_ms:.1f}/{result.symbol1_ms:.1f} ms, "
        f"RS erasures {result.erasure_fraction * 100:.1f}%)"
    )
    return 0


def cmd_analyze_audio(args: argparse.Namespace) -> int:
    from . import audio_stego  # noqa: F401  (ensures WAV reader is importable)

    cfg = _config_from_args(args)
    report = analyze_audio_file(
        args.input, baseline_path=getattr(args, "baseline", None),
        cfg=cfg.audio, det=cfg.detector,
    )
    _print(
        _table(
            [
                ("file", report.sample_path),
                ("baseline", report.baseline_path or "-"),
                ("phase entropy", f"{report.metrics['phase_hist_entropy']:.4f}"),
                ("grid z-max", f"{report.metrics['cluster_zmax']:.3f}"),
                ("grid z-threshold", f"{report.metrics.get('score_cluster_threshold', 0.0):.3f}"),
                ("mean grid dist", f"{report.metrics['mean_grid_distance']:.4f}"),
                ("pi/2 clustering", f"{report.metrics['half_pi_clustering']:.4f}"),
                ("usable bins", f"{report.metrics['usable_bins']:.0f}"),
                ("carriers", f"{report.metrics['carrier_segments']:.0f}"),
                ("entropy score", f"{report.metrics.get('score_entropy_score', 0.0):.3f}"),
                ("cluster score", f"{report.metrics.get('score_cluster_score', 0.0):.3f}"),
                ("ANOMALY SCORE", f"{report.anomaly_score:.3f}"),
                ("VERDICT", report.verdict),
            ],
            ["phaseveil analyze audio", "value"],
        )
    )
    if report.baseline_delta:
        _print(
            "baseline delta: "
            + ", ".join(f"{k}={v:.4f}" for k, v in report.baseline_delta.items())
        )
    return 0


def cmd_analyze_timing(args: argparse.Namespace) -> int:
    from . import timing_channel  # noqa: F401

    cfg = _config_from_args(args)
    report = analyze_timing_file(args.trace, det=cfg.detector)
    m = report.metrics
    _print(
        _table(
            [
                ("trace", report.trace_path),
                ("IAT samples", report.n_samples),
                ("mean IAT", f"{m.get('mean_ms', float('nan')):.2f} ms"),
                ("std IAT", f"{m.get('std_ms', float('nan')):.2f} ms"),
                ("cluster means", f"{m.get('cluster_low_ms', float('nan')):.1f} / "
                                 f"{m.get('cluster_high_ms', float('nan')):.1f} ms"),
                ("bimodal R", f"{m.get('bimodal_separation', float('nan')):.3f}"),
                ("IAT entropy", f"{m.get('iat_entropy_bits', float('nan')):.3f} bits"),
                ("lag-1 autocorr", f"{m.get('lag1_autocorr', float('nan')):.4f}"),
                ("KS p-value", f"{m.get('ks_pvalue', float('nan')):.4g}"),
                ("ANOMALY SCORE", f"{report.anomaly_score:.3f}"),
                ("VERDICT", report.verdict),
            ],
            ["phaseveil analyze timing", "value"],
        )
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from . import benchmark

    cfg = PhaseVeilConfig()
    output = benchmark.run_benchmark(cfg)
    _print(output.table)
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(output.data, indent=2))
        _print(f"benchmark data -> {args.json}")
    return 0 if output.success else 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the full two-level argparse CLI."""
    p = argparse.ArgumentParser(
        prog="phaseveil",
        description=(
            "PhaseVeil - covert-channel / audio phase-spectrum steganography / "
            "statistical steganalysis engine (deterministic, zero AI/ML). "
            "For authorised research and educational lab use only."
        ),
        epilog=(
            "Network operations use unprivileged UDP/TCP sockets on 127.0.0.1 "
            "loopback by default. Never use this tool on networks you do not own "
            "or lack written authorisation to test."
        ),
    )
    p.add_argument("--version", action="version", version=f"phaseveil {__version__}")
    root = p.add_subparsers(dest="group", required=True, metavar="COMMAND")

    # ============================== audio =================================
    audio = root.add_parser("audio", help="frequency-domain phase-spectrum steganography")
    a_sub = audio.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")

    ah = a_sub.add_parser("hide", help="embed a payload into a cover .wav file")
    ah.add_argument("--in", dest="input", required=True, help="cover .wav (16-bit PCM mono)")
    ah.add_argument("--out", dest="output", required=True, help="output stego .wav path")
    _add_payload_args(ah)
    ah.add_argument("--no-compress", action="store_true", help="disable zlib compression")
    ah.set_defaults(func=cmd_audio_hide)

    ae = a_sub.add_parser("extract", help="extract a payload from a stego .wav file")
    ae.add_argument("--in", dest="input", required=True, help="stego .wav path")
    ae.add_argument("--out", dest="out", help="write recovered bytes to a file")
    ae.add_argument("--passphrase", help="AES-256-GCM passphrase used at hide time")
    ae.set_defaults(func=cmd_audio_extract)

    # ============================== timing ================================
    timing = root.add_parser("timing", help="network inter-packet-delay covert channel")
    t_sub = timing.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")

    ts = t_sub.add_parser("send", help="transmit a payload over loopback IPD channel")
    ts.add_argument("--host", default="127.0.0.1", help="destination host (default 127.0.0.1)")
    ts.add_argument("--port", type=int, default=9999, help="destination port (default 9999)")
    _add_payload_args(ts)
    ts.add_argument("--delay-ms", dest="delay_ms", help="'BIT0,BIT1' ms delays (default 20,60)")
    ts.add_argument("--tcp", action="store_true", help="use TCP instead of UDP")
    ts.add_argument("--no-compress", action="store_true", help="disable zlib compression")
    ts.set_defaults(func=cmd_timing_send)

    tl = t_sub.add_parser("listen", help="listen for a loopback IPD transmission")
    tl.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    tl.add_argument("--port", type=int, default=9999, help="bind port (default 9999)")
    tl.add_argument("--out", help="write recovered bytes to a file")
    tl.add_argument("--save-trace", dest="save_trace", help="save packet timestamps to JSON")
    tl.add_argument("--passphrase", help="AES-256-GCM passphrase used at send time")
    tl.add_argument("--delay-ms", dest="delay_ms", help="'BIT0,BIT1' ms delays (default 20,60)")
    tl.add_argument("--tcp", action="store_true", help="use TCP instead of UDP")
    tl.set_defaults(func=cmd_timing_listen)

    # ============================= analyze ================================
    analyze = root.add_parser("analyze", help="deterministic statistical steganalysis")
    an_sub = analyze.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")

    aa = an_sub.add_parser("audio", help="phase-spectrum analysis of a .wav file")
    aa.add_argument("--in", dest="input", required=True, help="audio file to analyse")
    aa.add_argument("--baseline", help="optional clean cover for differential analysis")
    aa.set_defaults(func=cmd_analyze_audio)

    at = an_sub.add_parser("timing", help="IAT statistical analysis of a packet trace")
    at.add_argument("--trace", required=True, help="trace.json written by 'timing listen'")
    at.set_defaults(func=cmd_analyze_timing)

    # ============================= benchmark ==============================
    bm = root.add_parser("benchmark", help="synthetic capacity/SNR/jitter/detector benchmark")
    bm.add_argument("--json", help="also dump the raw benchmark data as JSON")
    bm.set_defaults(func=cmd_benchmark)

    return p


def _add_payload_args(sp: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive --text/--file payload source arguments."""
    grp = sp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--text", help="payload text (UTF-8)")
    grp.add_argument("--file", help="payload file path (read as bytes)")
    sp.add_argument("--passphrase", help="AES-256-GCM passphrase (PBKDF2-HMAC-SHA256)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())