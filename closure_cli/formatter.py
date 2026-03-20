"""Format incident reports for humans and machines."""

from __future__ import annotations

import json
from closure_sdk import IncidentReport, IncidentValence


def format_stdout(
    incidents: list[IncidentReport],
    source_name: str,
    target_name: str,
    source_count: int,
    target_count: int,
    elapsed: float,
    report_path: str | None = None,
) -> str:
    """The full human-readable output for the terminal."""
    lines = [
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  Closure Identity                                          │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        f"  Source:  {source_name} ({source_count:,} records)",
        f"  Target:  {target_name} ({target_count:,} records)",
        "",
    ]

    if not incidents:
        lines += [
            "  Result:  Coherent",
            "  The two files are identical. No missing records, no reorders.",
            "",
            f"  Verified in {_time_str(elapsed)}.",
            "",
        ]
        return "\n".join(lines)

    missing = [i for i in incidents if i.incident_type == "missing"]
    reorders = [i for i in incidents if i.incident_type == "reorder"]

    lines += [
        f"  Result:  {len(incidents)} incidents found",
        "",
    ]

    # Breakdown
    if missing:
        only_src = sum(1 for i in missing if i.source_index is not None and i.target_index is None)
        only_tgt = sum(1 for i in missing if i.target_index is not None and i.source_index is None)
        lines.append(f"    Missing:   {len(missing)} records")
        if only_src:
            lines.append(f"               {only_src} in source only (absent from target)")
        if only_tgt:
            lines.append(f"               {only_tgt} in target only (absent from source)")

    if reorders:
        lines.append(f"    Reorder:   {len(reorders)} records found in both files at different positions")

    lines.append("")

    # Table
    lines += [
        "  ─────┬──────────┬──────────┬──────────┬────────────────────────────────",
        "   #   │ Type     │ Source   │ Target   │ Payload",
        "  ─────┼──────────┼──────────┼──────────┼────────────────────────────────",
    ]

    for i, inc in enumerate(incidents, 1):
        src = str(inc.source_index) if inc.source_index is not None else "—"
        tgt = str(inc.target_index) if inc.target_index is not None else "—"
        payload = inc.record[:60].decode("utf-8", errors="replace")
        lines.append(f"  {i:>4} │ {inc.incident_type:<8} │ {src:<8} │ {tgt:<8} │ {payload}")

    lines.append("  ─────┴──────────┴──────────┴──────────┴────────────────────────────────")
    lines.append("")
    lines.append(f"  Verified in {_time_str(elapsed)}.")

    if report_path:
        lines.append(f"  Full report (with color channels): {report_path}")

    lines.append("")
    return "\n".join(lines)


def format_report_json(
    incidents: list[IncidentReport],
    valences: list[IncidentValence],
    source_name: str,
    target_name: str,
    source_count: int,
    target_count: int,
    elapsed: float,
) -> str:
    """Full detailed report as JSON — for saving to file or piping."""
    missing_count = sum(1 for i in incidents if i.incident_type == "missing")
    reorder_count = sum(1 for i in incidents if i.incident_type == "reorder")

    records = []
    for inc, v in zip(incidents, valences):
        records.append({
            "type": inc.incident_type,
            "source_index": inc.source_index,
            "target_index": inc.target_index,
            "payload": inc.record.decode("utf-8", errors="replace"),
            "checks": inc.checks,
            "sigma": round(v.sigma, 6),
            "channels": {
                "R": round(v.base[0], 6),
                "G": round(v.base[1], 6),
                "B": round(v.base[2], 6),
                "W": round(v.phase, 6),
            },
            "axis": v.axis,
        })

    report = {
        "closure_identity_report": {
            "source": source_name,
            "target": target_name,
            "source_records": source_count,
            "target_records": target_count,
            "verified_in_seconds": round(elapsed, 3),
        },
        "summary": {
            "total_incidents": len(incidents),
            "missing": missing_count,
            "reorder": reorder_count,
            "coherent": len(incidents) == 0,
        },
        "incidents": records,
    }
    return json.dumps(report, indent=2)


def _time_str(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}μs"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"
