# Contributing to PhaseVeil

Thanks for your interest. PhaseVeil is a research engine, so contributions that
keep it *deterministic, well-documented, and mathematically clean* are worth
far more than new features. Read the README first — it documents the scheme,
the parameters, and the intended architecture.

## Ground rules

1. **Determinism is a feature.** Every transform, encoder, and detector must
   produce byte-identical output for identical inputs. No unseeded randomness
   outside `os.urandom` in the crypto layer (salt/nonce), no dependence on
   wall-clock time, dict ordering, locale, or thread scheduling.
2. **No new dependencies.** The allow-list is exactly `numpy`, `scipy`,
   `cryptography`, `reedsolo` (+ `pytest` for tests) and the Python standard
   library. No AI/ML libraries, web frameworks, GUIs, or raw-packet
   (`scapy`-style) dependencies. Adding one requires a very good argument and a
   maintainer decision, not a PR.
3. **No naive LSB shortcuts.** Improvements to the audio layer must stay in the
   frequency-domain phase-coding family. If you want to compare against
   time-domain LSB as a baseline, do it in a benchmark or a test, not as the
   default path.
4. **Authorized lab use only.** The network channel stays on `127.0.0.1`
   loopback with unprivileged sockets by default. Do not add features that open
   non-loopback sockets, need elevated privileges, or craft raw packets.
5. **Every number must be reproducible.** When you change a parameter or a
   threshold, run `phaseveil benchmark` and report the before/after in the PR
   description. Threshold changes are expected to move TP/FP rates; say how.

## Local setup

No virtualenv is required (see the README for the full options):

```bash
pip install --user --break-system-packages -e .     # editable, no venv

# tests, straight from source with no install at all:
python3 -m pytest
```

Python 3.10 or newer.

## Definition of done

A pull request is ready when **all** of these hold:

- [ ] `python3 -m pytest` — **136 tests pass** (growing, please).
- [ ] `phaseveil benchmark` — `overall: PASS`, peak RSS under the 150 MB budget.
- [ ] New behaviour has new tests. Synthetic fixtures are generated **in
      memory** inside `tests/`; never commit `.wav` or trace artifacts.
- [ ] Full type hints on public functions; docstrings that state the *math*
      (transforms, estimators, decision rules), not just the intent.
- [ ] `pyproject.toml` dependency list unchanged (see ground rule 2).
- [ ] The README is updated if a parameter, threshold, metric, or limitation
      moved. Numbers in the README must match `phaseveil benchmark`.

## Style

- Python 3.10+, `from __future__ import annotations`, standard library
  `typing`/`dataclasses`. Type hints everywhere on public APIs.
- Keep modules single-purpose: `config.py` (parameters), `crypto.py`, `payload.py`,
  `audio_stego.py`, `timing_channel.py`, `detector.py`, `benchmark.py`, `cli.py`.
  New tunables go in `config.py`, never as a literal buried in logic.
- Prefer `numpy`/`scipy` vectorised code over Python loops in hot paths; process
  audio one chunk at a time so long files never materialise a full-file FFT.
- Errors: raise the project's own exception types
  (`AudioStegoError`, `CapacityError`, `ExtractionError`, `SyncError`,
  `TimingError`, `FramingError`, `CryptoError`, `PassphraseError`) and let the
  CLI render a one-line `phaseveil: error: ...` message. Never fall back
  silently to "best effort" data.
- Commit messages: short imperative subject, then a body explaining *why*.
  Reference the bug you fixed and how you verified the fix.

## Especially welcome

- Robustness work: new cover classes in the audio test matrix, heavier jitter
  profiles, lossy/tampered-input tests.
- Detector calibration: measuring TP/FP across more cover/traffic families and
  reporting the confusion matrix.
- Accuracy improvements to the metrics (`SNR`, `PSNR`, capacity accounting).
- Documentation and mathematical exposition — the README's math sections are
  part of the deliverable.

## Things that will be declined

- AI/ML-based detection or embedding, "let's add a neural net".
- Removing the AES-GCM/Reed-Solomon layers to make demos shorter.
- Making the network channel target non-loopback hosts by default.
- Trading determinism or the 150 MB RSS budget for speed.
- Committing binary fixtures or generated traces.

## Security reports

Please do **not** open a public issue for vulnerabilities — see
[SECURITY.md](SECURITY.md) for the private reporting path.