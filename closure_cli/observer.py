"""Closure Observer — lightweight streaming monitor via Seer.

Composes each record on S³ as it arrives, checks drift at every window tick.
When drift exceeds threshold, dumps the retention window into Gilgamesh
for exact incident classification.

This is the smoke detector: cheap, always-on, scales to many streams.
For deep real-time classification of every record, use 'closure seeker'.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from closure_sdk import (
    Seer, RetentionWindow, gilgamesh, IncidentReport,
    expose_incident,
)
from .formatter import format_report_json


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "observer",
        help="Monitor two streams for drift, escalate on detection.",
        description=(
            "Lightweight monitor: composes each record on S³, checks drift "
            "at every window tick. When drift is detected, auto-escalates "
            "to Gilgamesh for exact incident classification."
        ),
    )
    p.add_argument("source", help="Source stream: file path, named pipe, or '-' for stdin.")
    p.add_argument("target", help="Target stream: file path, named pipe, or '-' for stdin.")
    p.add_argument("--window", type=int, default=100,
                   help="Records per retention block (default: 100).")
    p.add_argument("--threshold", type=float, default=1e-10,
                   help="Drift threshold to trigger escalation (default: 1e-10).")
    p.add_argument("--retain", type=int, default=64,
                   help="Number of blocks to retain for escalation (default: 64).")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Save report to this JSON file on exit.")
    p.set_defaults(func=run)


@dataclass
class _ObserverState:
    src_seer: Seer
    tgt_seer: Seer
    src_window: RetentionWindow
    tgt_window: RetentionWindow
    window_size: int
    threshold: float
    src_block: list[bytes] = field(default_factory=list)
    tgt_block: list[bytes] = field(default_factory=list)
    src_pos: int = 0
    tgt_pos: int = 0
    total: int = 0
    checks: int = 0
    escalations: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def ingest_source(self, payload: bytes) -> None:
        self.src_seer.ingest(payload)
        self.src_block.append(payload)
        self.src_pos += 1
        self.total += 1

    def ingest_target(self, payload: bytes) -> None:
        self.tgt_seer.ingest(payload)
        self.tgt_block.append(payload)
        self.tgt_pos += 1
        self.total += 1

    def tick(self, block_num: int) -> None:
        """Flush current blocks into retention, check drift."""
        if self.src_block:
            self.src_window.append(block_num, list(self.src_block))
            self.src_block.clear()
        if self.tgt_block:
            self.tgt_window.append(block_num, list(self.tgt_block))
            self.tgt_block.clear()

        result = self.src_seer.compare(self.tgt_seer, threshold=self.threshold)
        self.checks += 1

        if result.coherent:
            _print_check(block_num, result.drift, coherent=True)
        else:
            _print_check(block_num, result.drift, coherent=False)
            self._escalate(block_num, result.drift)

    def _escalate(self, block_num: int, drift: float) -> None:
        """Drift detected. Dump retention windows into Gilgamesh."""
        _print_escalation()

        src_records = self.src_window.flatten()
        tgt_records = self.tgt_window.flatten()

        t0 = time.perf_counter()
        incidents = gilgamesh(src_records, tgt_records)
        elapsed = time.perf_counter() - t0

        self.escalations.append({
            "block": block_num,
            "drift": drift,
            "incidents": incidents,
            "src_records": len(src_records),
            "tgt_records": len(tgt_records),
            "elapsed": elapsed,
        })

        _print_incidents(incidents, elapsed)


def run(args: argparse.Namespace) -> int:
    if args.source == "-" and args.target == "-":
        print("  Error: both source and target cannot be stdin.", file=sys.stderr)
        return 1

    # Header
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Closure Observer                                          │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Source:     {args.source}")
    print(f"  Target:     {args.target}")
    print(f"  Window:     {args.window} records")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Retention:  {args.retain} blocks ({args.retain * args.window:,} records max)")
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


def _make_state(args: argparse.Namespace) -> _ObserverState:
    return _ObserverState(
        src_seer=Seer(),
        tgt_seer=Seer(),
        src_window=RetentionWindow(maxlen=args.retain),
        tgt_window=RetentionWindow(maxlen=args.retain),
        window_size=args.window,
        threshold=args.threshold,
    )


def _run_interleaved(args: argparse.Namespace) -> int:
    """Both inputs are regular files — interleave line by line."""
    state = _make_state(args)
    block_num = 0

    print("  Monitoring...", flush=True)
    print()

    src_file = open(args.source, "rb")
    tgt_file = open(args.target, "rb")
    src_done = False
    tgt_done = False
    records_in_block = 0

    while not (src_done and tgt_done):
        if not src_done:
            line = src_file.readline()
            if not line:
                src_done = True
            else:
                payload = line.rstrip(b"\n").rstrip(b"\r")
                if payload:
                    state.ingest_source(payload)
                    records_in_block += 1

        if not tgt_done:
            line = tgt_file.readline()
            if not line:
                tgt_done = True
            else:
                payload = line.rstrip(b"\n").rstrip(b"\r")
                if payload:
                    state.ingest_target(payload)
                    records_in_block += 1

        if records_in_block >= state.window_size * 2:
            state.tick(block_num)
            block_num += 1
            records_in_block = 0

    src_file.close()
    tgt_file.close()

    # Final tick
    if records_in_block > 0:
        state.tick(block_num)

    _print_report(state, args)
    return 0 if not state.escalations else 2


def _run_threaded(args: argparse.Namespace) -> int:
    """At least one input is a pipe/stdin — use threads."""
    state = _make_state(args)
    block_num = 0
    stop = threading.Event()

    def on_signal(sig, frame):
        stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    q: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=4096)

    print("  Monitoring...", flush=True)
    print()

    src_thread = threading.Thread(
        target=_read_stream, args=(args.source, "source", q, stop), daemon=True)
    tgt_thread = threading.Thread(
        target=_read_stream, args=(args.target, "target", q, stop), daemon=True)
    src_thread.start()
    tgt_thread.start()

    active_readers = 2
    records_in_block = 0

    while active_readers > 0 and not stop.is_set():
        try:
            item = q.get(timeout=0.2)
        except queue.Empty:
            continue

        if item is None:
            active_readers -= 1
            continue

        side, payload = item
        if side == "source":
            state.ingest_source(payload)
        else:
            state.ingest_target(payload)
        records_in_block += 1

        if records_in_block >= state.window_size * 2:
            state.tick(block_num)
            block_num += 1
            records_in_block = 0

    # Final tick
    if records_in_block > 0:
        state.tick(block_num)

    _print_report(state, args)
    return 0 if not state.escalations else 2


def _read_stream(
    path: str,
    side: str,
    q: queue.Queue,
    stop: threading.Event,
) -> None:
    f = sys.stdin.buffer if path == "-" else open(path, "rb")
    try:
        for line in f:
            if stop.is_set():
                break
            payload = line.rstrip(b"\n").rstrip(b"\r")
            if payload:
                q.put((side, payload))
    finally:
        if f is not sys.stdin.buffer:
            f.close()
        q.put(None)


# ── Live output ──────────────────────────────────────────────

def _print_check(block: int, drift: float, coherent: bool) -> None:
    status = "coherent" if coherent else f"DRIFT DETECTED"
    sigma = f"σ={drift:.6f}"
    if coherent:
        print(f"  [{block:>4}]  {sigma}  {status}", flush=True)
    else:
        print(f"  [{block:>4}]  {sigma}  *** {status} ***", flush=True)


def _print_escalation() -> None:
    print()
    print("         Escalating → Gilgamesh...", flush=True)


def _print_incidents(incidents: list[IncidentReport], elapsed: float) -> None:
    if not incidents:
        print("         No incidents found in retention window.", flush=True)
        print()
        return

    missing = sum(1 for i in incidents if i.incident_type == "missing")
    reorders = sum(1 for i in incidents if i.incident_type == "reorder")
    print(f"         {len(incidents)} incidents ({missing} missing, {reorders} reorder) in {elapsed*1000:.1f}ms", flush=True)
    print()

    for i, inc in enumerate(incidents[:10], 1):
        src = str(inc.source_index) if inc.source_index is not None else "—"
        tgt = str(inc.target_index) if inc.target_index is not None else "—"
        payload = inc.record[:50].decode("utf-8", errors="replace")
        typ = inc.incident_type
        print(f"         {i:>3}. {typ:<8}  src[{src}] tgt[{tgt}]  {payload}", flush=True)

    if len(incidents) > 10:
        print(f"         ... and {len(incidents) - 10} more", flush=True)
    print()


# ── Final report ─────────────────────────────────────────────

def _print_report(state: _ObserverState, args: argparse.Namespace) -> None:
    elapsed = time.perf_counter() - state.t0

    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Observer Report                                           │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Records processed:  {state.total:,}")
    print(f"  Drift checks:       {state.checks}")
    print(f"  Escalations:        {len(state.escalations)}")

    if elapsed < 1.0:
        print(f"  Duration:           {elapsed * 1000:.1f}ms")
    else:
        print(f"  Duration:           {elapsed:.2f}s")

    if not state.escalations:
        print()
        print("  Result:  Coherent")
        print("  All drift checks passed. Streams are identical.")
    else:
        total_incidents = sum(len(e["incidents"]) for e in state.escalations)
        print()
        print(f"  Result:  {total_incidents} incidents across {len(state.escalations)} escalation(s)")
        print()

        for esc in state.escalations:
            incidents = esc["incidents"]
            missing = sum(1 for i in incidents if i.incident_type == "missing")
            reorders = sum(1 for i in incidents if i.incident_type == "reorder")

            print(f"  Escalation at block {esc['block']}  (σ={esc['drift']:.6f})")
            print(f"    Window: {esc['src_records']:,} src × {esc['tgt_records']:,} tgt")
            print(f"    Found:  {len(incidents)} incidents ({missing} missing, {reorders} reorder)")
            print()

            # Table
            print("    ─────┬──────────┬──────────┬──────────┬────────────────────────────")
            print("     #   │ Type     │ Source   │ Target   │ Payload")
            print("    ─────┼──────────┼──────────┼──────────┼────────────────────────────")
            for i, inc in enumerate(incidents, 1):
                src = str(inc.source_index) if inc.source_index is not None else "—"
                tgt = str(inc.target_index) if inc.target_index is not None else "—"
                payload = inc.record[:50].decode("utf-8", errors="replace")
                print(f"    {i:>4} │ {inc.incident_type:<8} │ {src:<8} │ {tgt:<8} │ {payload}")
            print("    ─────┴──────────┴──────────┴──────────┴────────────────────────────")
            print()

    print()

    if args.output:
        _save_report(state, args)


def _save_report(state: _ObserverState, args: argparse.Namespace) -> None:
    import json

    elapsed = time.perf_counter() - state.t0
    escalation_records = []
    for esc in state.escalations:
        esc_data = {
            "block": esc["block"],
            "drift": round(esc["drift"], 6),
            "window_src_records": esc["src_records"],
            "window_tgt_records": esc["tgt_records"],
            "incidents": [],
        }
        for inc in esc["incidents"]:
            esc_data["incidents"].append({
                "type": inc.incident_type,
                "source_index": inc.source_index,
                "target_index": inc.target_index,
                "payload": inc.record.decode("utf-8", errors="replace"),
            })
        escalation_records.append(esc_data)

    report = {
        "closure_observer_report": {
            "source": args.source,
            "target": args.target,
            "records_processed": state.total,
            "drift_checks": state.checks,
            "duration_seconds": round(elapsed, 3),
        },
        "summary": {
            "escalations": len(state.escalations),
            "total_incidents": sum(len(e["incidents"]) for e in state.escalations),
            "coherent": len(state.escalations) == 0,
        },
        "escalations": escalation_records,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {args.output}")
    print()
