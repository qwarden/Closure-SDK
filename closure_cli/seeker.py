"""Closure Seeker — deep streaming classification via Enkidu.

Classifies every record as it arrives. Maintains unmatched pools,
tracks lifecycles, reclassifies missing → reorder when matches arrive.
Use this for critical streams where you need every incident in real time.

For lightweight monitoring of many streams, use 'closure observer' instead.
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from closure_sdk import Enkidu, IncidentReport


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "seeker",
        help="Watch two streams and classify incidents in real time.",
        description=(
            "Reads newline-delimited records from two sources, feeds them "
            "through Enkidu, and prints incidents as they happen. "
            "Use files, named pipes, process substitution, or '-' for stdin."
        ),
    )
    p.add_argument("source", help="Source stream: file path, named pipe, or '-' for stdin.")
    p.add_argument("target", help="Target stream: file path, named pipe, or '-' for stdin.")
    p.add_argument("--window", type=int, default=100,
                   help="Records between each cycle tick (default: 100).")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Save incident log to this JSON file on exit.")
    p.set_defaults(func=run)


@dataclass
class _Event:
    """One event in the observer's log. Tracks the lifecycle."""
    incident: IncidentReport
    event_type: str  # "held", "confirmed_missing", "reclassified_reorder"


@dataclass
class _ObserverState:
    enkidu: Enkidu
    window: int
    events: list[_Event] = field(default_factory=list)
    total: int = 0
    t0: float = field(default_factory=time.perf_counter)

    def ingest(self, payload: bytes, pos: int, side: str) -> None:
        inc = self.enkidu.ingest(payload, pos, side)
        if inc:
            # ingest() only returns when reclassifying: was missing → now reorder
            ev = _Event(inc, "reclassified_reorder")
            self.events.append(ev)
            _print_reclassified(inc)
        self.total += 1

    def cycle(self) -> None:
        for inc in self.enkidu.advance_cycle():
            ev = _Event(inc, "confirmed_missing")
            self.events.append(ev)
            _print_confirmed(inc)


def run(args: argparse.Namespace) -> int:
    if args.source == "-" and args.target == "-":
        print("  Error: both source and target cannot be stdin.", file=sys.stderr)
        return 1

    # Header
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Closure Seeker                                            │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Source:  {args.source}")
    print(f"  Target:  {args.target}")
    print(f"  Window:  {args.window} records")
    print()

    # Detect file vs stream
    is_stream = args.source == "-" or args.target == "-"
    if not is_stream:
        import os, stat
        for p in (args.source, args.target):
            try:
                if stat.S_ISFIFO(os.stat(p).st_mode):
                    is_stream = True
                    break
            except OSError:
                pass

    if is_stream:
        return _run_threaded(args)
    else:
        return _run_interleaved(args)


def _run_interleaved(args: argparse.Namespace) -> int:
    """Both inputs are regular files — interleave line by line."""
    state = _ObserverState(Enkidu(), args.window)

    print("  Reading...", flush=True)
    print()

    src_file = open(args.source, "rb")
    tgt_file = open(args.target, "rb")
    src_pos = 0
    tgt_pos = 0
    src_done = False
    tgt_done = False

    while not (src_done and tgt_done):
        if not src_done:
            line = src_file.readline()
            if not line:
                src_done = True
            else:
                payload = line.rstrip(b"\n").rstrip(b"\r")
                if payload:
                    state.ingest(payload, src_pos, "source")
                    src_pos += 1

        if not tgt_done:
            line = tgt_file.readline()
            if not line:
                tgt_done = True
            else:
                payload = line.rstrip(b"\n").rstrip(b"\r")
                if payload:
                    state.ingest(payload, tgt_pos, "target")
                    tgt_pos += 1

        if state.total > 0 and state.total % state.window == 0:
            state.cycle()

    src_file.close()
    tgt_file.close()
    state.cycle()  # Final flush

    _print_report(state, args.output)
    return 0 if not state.events else 2


def _run_threaded(args: argparse.Namespace) -> int:
    """At least one input is a pipe/stdin — use threads."""
    state = _ObserverState(Enkidu(), args.window)
    stop = threading.Event()

    def on_signal(sig, frame):
        stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    q: queue.Queue[tuple[str, int, bytes] | None] = queue.Queue(maxsize=4096)

    print("  Watching...", flush=True)
    print()

    src_thread = threading.Thread(
        target=_read_stream, args=(args.source, "source", q, stop), daemon=True)
    tgt_thread = threading.Thread(
        target=_read_stream, args=(args.target, "target", q, stop), daemon=True)
    src_thread.start()
    tgt_thread.start()

    active_readers = 2
    while active_readers > 0 and not stop.is_set():
        try:
            item = q.get(timeout=0.2)
        except queue.Empty:
            continue

        if item is None:
            active_readers -= 1
            continue

        side, pos, payload = item
        state.ingest(payload, pos, side)

        if state.total % state.window == 0:
            state.cycle()

    state.cycle()  # Final flush

    _print_report(state, args.output)
    return 0 if not state.events else 2


def _read_stream(
    path: str,
    side: str,
    q: queue.Queue,
    stop: threading.Event,
) -> None:
    """Read lines from a file/pipe/stdin and push to the shared queue."""
    f = sys.stdin.buffer if path == "-" else open(path, "rb")
    try:
        pos = 0
        for line in f:
            if stop.is_set():
                break
            payload = line.rstrip(b"\n").rstrip(b"\r")
            if payload:
                q.put((side, pos, payload))
                pos += 1
    finally:
        if f is not sys.stdin.buffer:
            f.close()
        q.put(None)


# ── Live output ──────────────────────────────────────────────

def _print_confirmed(inc: IncidentReport) -> None:
    """A record waited a full cycle with no match. Now confirmed missing."""
    src = _pos(inc.source_index)
    tgt = _pos(inc.target_index)
    payload = inc.record[:50].decode("utf-8", errors="replace")
    print(f"  MISSING          {src:<12} {tgt:<12} {payload}", flush=True)


def _print_reclassified(inc: IncidentReport) -> None:
    """A previously missing record just matched. Reclassified to reorder."""
    src = _pos(inc.source_index)
    tgt = _pos(inc.target_index)
    payload = inc.record[:50].decode("utf-8", errors="replace")
    print(f"  REORDER ← was missing  {src:<12} {tgt:<12} {payload}", flush=True)


def _pos(idx: int | None) -> str:
    return f"[{idx}]" if idx is not None else "[—]"


# ── Final report ─────────────────────────────────────────────

def _print_report(state: _ObserverState, output_path: str | None) -> None:
    elapsed = time.perf_counter() - state.t0

    # Final counts: only the last classification for each record matters
    confirmed_missing = [e for e in state.events if e.event_type == "confirmed_missing"]
    reclassified = [e for e in state.events if e.event_type == "reclassified_reorder"]

    # Reclassified reorders cancel out their original missing classification.
    # Final state: confirmed_missing that were NOT later reclassified, plus reclassified.
    reclassified_payloads = {e.incident.record for e in reclassified}
    final_missing = [e for e in confirmed_missing if e.incident.record not in reclassified_payloads]
    final_reorder = reclassified

    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Seeker Report                                             │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Records processed:  {state.total:,}")

    if elapsed < 1.0:
        print(f"  Duration:           {elapsed * 1000:.1f}ms")
    else:
        print(f"  Duration:           {elapsed:.2f}s")

    print(f"  Unresolved:         {state.enkidu.unresolved_source} source, {state.enkidu.unresolved_target} target")
    print()

    total_final = len(final_missing) + len(final_reorder)
    if total_final == 0:
        print("  Result:  Coherent")
        print("  All records matched across both streams.")
    else:
        print(f"  Result:  {total_final} incidents")
        if final_missing:
            print(f"    Missing:   {len(final_missing)} records never matched")
        if final_reorder:
            print(f"    Reorder:   {len(final_reorder)} records arrived out of order (initially looked missing)")
        print()

        # Final incident table
        all_final = (
            [(e.incident, "missing") for e in final_missing]
            + [(e.incident, "reorder") for e in final_reorder]
        )
        all_final.sort(key=lambda x: (x[0].source_index or 0, x[0].target_index or 0))

        print("  ─────┬──────────┬──────────┬──────────┬────────────────────────────────")
        print("   #   │ Type     │ Source   │ Target   │ Payload")
        print("  ─────┼──────────┼──────────┼──────────┼────────────────────────────────")
        for i, (inc, typ) in enumerate(all_final, 1):
            src = str(inc.source_index) if inc.source_index is not None else "—"
            tgt = str(inc.target_index) if inc.target_index is not None else "—"
            payload = inc.record[:60].decode("utf-8", errors="replace")
            print(f"  {i:>4} │ {typ:<8} │ {src:<8} │ {tgt:<8} │ {payload}")
        print("  ─────┴──────────┴──────────┴──────────┴────────────────────────────────")

    print()

    if output_path:
        records = []
        for inc, typ in (
            [(e.incident, "missing") for e in final_missing]
            + [(e.incident, "reorder") for e in final_reorder]
        ):
            records.append({
                "type": typ,
                "source_index": inc.source_index,
                "target_index": inc.target_index,
                "payload": inc.record.decode("utf-8", errors="replace"),
            })
        report = {
            "closure_seeker_report": {
                "source": state._source_name if hasattr(state, '_source_name') else "stream",
                "target": state._target_name if hasattr(state, '_target_name') else "stream",
                "records_processed": state.total,
                "duration_seconds": round(elapsed, 3),
            },
            "summary": {
                "total_incidents": total_final,
                "missing": len(final_missing),
                "reorder": len(final_reorder),
                "coherent": total_final == 0,
            },
            "incidents": records,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved: {output_path}")
        print()
