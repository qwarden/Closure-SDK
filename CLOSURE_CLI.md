# Closure CLI

Three commands, one tool. Each one answers a different version of the same question: are these two data streams the same?

---

## Install

```bash
pip install -e '.[dev]'
```

This gives you the `closure` command and all three modes. You can also run it as a module:

```bash
python -m closure_cli identity source.jsonl target.jsonl
```

---

## The three commands

| Command | What it does | When to use it |
|---|---|---|
| `closure identity` | Takes two files, compares every record, reports every discrepancy | You have two complete files and want the full picture |
| `closure observer` | Watches two streams cheaply, escalates the moment something drifts | Production monitoring — you want to know the instant it breaks |
| `closure seeker` | Classifies every single record as it arrives | You need the play-by-play, not just the verdict |

They sit on different layers of the SDK. Identity uses Gilgamesh (the batch comparator). Observer uses the Seer (the lightweight sensor) and escalates to Gilgamesh when it detects drift. Seeker uses Enkidu (the streaming classifier). Each one trades off cost against detail — pick the one that matches your situation.

---

## `closure identity`

The forensic tool. You have two files, you want to know exactly what's different between them — which records are missing, which ones moved, where in each file, and the actual payload. One command, full diagnosis.

### Usage

```bash
closure identity source.jsonl target.jsonl
```

### What it does

1. Reads both files into memory
2. Feeds them to Gilgamesh, which composes both sequences on S³ and walks them side by side
3. Every discrepancy gets classified as either **missing** (a record exists in one file but not the other) or **reorder** (a record exists in both but at different positions)
4. Prints a clean table to the terminal
5. Saves a full JSON report with color channels (σ, RGB, W) for programmatic use

### Terminal output

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Closure Identity                                          │
  └─────────────────────────────────────────────────────────────┘

  Source:  source.jsonl (100 records)
  Target:  target.jsonl (100 records)

  Result:  3 incidents found

    Missing:   2 records
               1 in source only (absent from target)
               1 in target only (absent from source)
    Reorder:   1 records found in both files at different positions

  ─────┬──────────┬──────────┬──────────┬────────────────────────────────
   #   │ Type     │ Source   │ Target   │ Payload
  ─────┼──────────┼──────────┼──────────┼────────────────────────────────
     1 │ missing  │ 42       │ —        │ {"id": "rec_042", "value": ...}
     2 │ missing  │ —        │ 67       │ {"id": "rec_067", "value": ...}
     3 │ reorder  │ 10       │ 95       │ {"id": "rec_010", "value": ...}
  ─────┴──────────┴──────────┴──────────┴────────────────────────────────

  Verified in 4.2ms.
  Full report (with color channels): source_identity.json
```

When the files are identical:

```
  Result:  Coherent
  The two files are identical. No missing records, no reorders.

  Verified in 3.1ms.
```

### JSON report

The JSON report includes everything the terminal shows plus the color channel decomposition for each incident. Saved automatically when incidents are found, or to a path you specify with `--output`.

```json
{
  "closure_identity_report": {
    "source": "source.jsonl",
    "target": "target.jsonl",
    "source_records": 100,
    "target_records": 100,
    "verified_in_seconds": 0.004
  },
  "summary": {
    "total_incidents": 3,
    "missing": 2,
    "reorder": 1,
    "coherent": false
  },
  "incidents": [
    {
      "type": "missing",
      "source_index": 42,
      "target_index": null,
      "payload": "{\"id\": \"rec_042\", ...}",
      "checks": 7,
      "sigma": 0.768921,
      "channels": {
        "R": 0.234567,
        "G": -0.123456,
        "B": 0.345678,
        "W": 0.567890
      },
      "axis": "existence"
    }
  ]
}
```

The `channels` break down each incident into its geometric components. W (the phase) responds to whether a record exists or doesn't. R, G, B (the base) respond to positional displacement. `axis` tells you which one dominated: `"existence"` for missing records, `"position"` for reorders.

### Options

| Flag | Default | What it does |
|---|---|---|
| `--format` | auto-detect | Force input format: `jsonl`, `csv`, or `text` |
| `--header` | off | Skip the first row of each file (useful for CSV headers) |
| `--columns` | all | CSV only: comma-separated column indices (0-based). `--columns 0,2` uses the first and third columns |
| `--delimiter` | `,` | CSV delimiter |
| `--max-faults` | 1000 | Stop after this many incidents |
| `--output`, `-o` | auto | Path for the JSON report. If omitted and incidents are found, saves as `<source>_identity.json` |

---

## `closure observer`

The smoke detector. Observer watches two streams for almost nothing — 32 bytes per stream, one comparison per window tick. The moment drift is detected, it auto-escalates: dumps the recent records into Gilgamesh and gives you exact incident classification. You get cheap monitoring most of the time, and precise diagnosis exactly when you need it.

### Usage

```bash
# Two files (interleaved reading)
closure observer source.jsonl target.jsonl

# With tuning
closure observer --window 1000 --retain 128 source.jsonl target.jsonl

# One file, one pipe
tail -f /var/log/stream_a | closure observer - target.jsonl

# Two pipes
mkfifo /tmp/src /tmp/tgt
closure observer /tmp/src /tmp/tgt
```

### What it does

1. Creates two Seers (one per stream) and two retention windows
2. Ingests records from both streams, composing each one on S³
3. Every `--window` records, compares the two Seers
4. If drift is below threshold: prints `coherent` and keeps going
5. If drift exceeds threshold: **escalation** — dumps the retention window contents into Gilgamesh, which classifies every incident in the buffered records
6. On exit, prints a summary of all escalations

The key insight: the Seer costs nearly nothing per record (one SHA-256 hash plus one quaternion multiplication — about 5% overhead on top of the hash alone). You can watch thousands of streams simultaneously. When something goes wrong, Gilgamesh runs only on the buffered window, not the entire history.

### Terminal output

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Closure Observer                                          │
  └─────────────────────────────────────────────────────────────┘

  Source:     source.jsonl
  Target:     target.jsonl
  Window:     100 records
  Threshold:  1e-10
  Retention:  64 blocks (6,400 records max)

  Monitoring...

  [   0]  σ=0.000000  coherent
  [   1]  σ=0.000000  coherent
  [   2]  σ=0.768921  *** DRIFT DETECTED ***

         Escalating → Gilgamesh...
         3 incidents (2 missing, 1 reorder) in 1.2ms

     1. missing   src[42] tgt[—]  {"id": "rec_042", ...}
     2. missing   src[—]  tgt[67]  {"id": "rec_067", ...}
     3. reorder   src[10] tgt[95]  {"id": "rec_010", ...}

  ┌─────────────────────────────────────────────────────────────┐
  │  Observer Report                                           │
  └─────────────────────────────────────────────────────────────┘

  Records processed:  600
  Drift checks:       3
  Escalations:        1
  Duration:           12.4ms

  Result:  3 incidents across 1 escalation(s)

  Escalation at block 2  (σ=0.768921)
    Window: 300 src × 300 tgt
    Found:  3 incidents (2 missing, 1 reorder)
```

When streams are identical:

```
  Result:  Coherent
  All drift checks passed. Streams are identical.
  Escalations:        0
```

### How escalation works

Observer keeps a rolling buffer of recent records (the retention window). When drift is detected, it doesn't scan the whole history — it dumps just the buffered records into Gilgamesh for classification. The buffer holds `--retain` blocks of `--window` records each. With defaults (64 blocks × 100 records), that's the most recent 6,400 records from each side.

After escalation, monitoring continues. If drift persists in the next window, it escalates again. Each escalation is independent.

### Input modes

Observer auto-detects whether its inputs are regular files or streams:

- **Two files**: reads them interleaved, one line from each side in turn. This prevents one side from racing ahead of the other.
- **One or both pipes/stdin**: uses threads to read both sides concurrently. Records are queued and processed as they arrive.

Use `-` for stdin. Named pipes (`mkfifo`) work. Process substitution works where your shell supports it.

### Options

| Flag | Default | What it does |
|---|---|---|
| `--window` | 100 | Records per retention block. Also controls how often drift is checked |
| `--threshold` | 1e-10 | Drift below this is treated as coherent |
| `--retain` | 64 | Number of blocks to keep in the retention window |
| `--output`, `-o` | none | Save the full report as JSON on exit |

### JSON report

```json
{
  "closure_observer_report": {
    "source": "source.jsonl",
    "target": "target.jsonl",
    "records_processed": 600,
    "drift_checks": 3,
    "duration_seconds": 0.012
  },
  "summary": {
    "escalations": 1,
    "total_incidents": 3,
    "coherent": false
  },
  "escalations": [
    {
      "block": 2,
      "drift": 0.768921,
      "window_src_records": 300,
      "window_tgt_records": 300,
      "incidents": [
        {
          "type": "missing",
          "source_index": 42,
          "target_index": null,
          "payload": "{\"id\": \"rec_042\", ...}"
        }
      ]
    }
  ]
}
```

---

## `closure seeker`

The deep mode. Seeker classifies every record the moment it arrives — not at window boundaries, not on escalation, but as each record is ingested. It uses Enkidu, the streaming classifier, which tracks unmatched records across both sides and reclassifies them as more data arrives.

Use seeker when you need the full play-by-play. Use observer when you need cheap monitoring with on-demand detail.

### Usage

```bash
# Two files
closure seeker source.jsonl target.jsonl

# With a wider cycle window
closure seeker --window 500 source.jsonl target.jsonl

# One pipe
tail -f /var/log/events | closure seeker - reference.jsonl
```

### What it does

1. Creates an Enkidu instance
2. Ingests records from both streams with their positions
3. For each record, Enkidu checks whether a matching record already arrived from the other side
4. If no match exists: the record is **held** for one cycle (a grace period)
5. If a match arrives during the grace period: reclassified as a **reorder** (it was in both streams, just at different positions)
6. If the cycle ends with no match: promoted to **missing** (confirmed absent from the other side)
7. On exit, prints the final classification of every incident

This two-phase approach solves the oracle problem: when a record shows up on one side but not the other, you can't immediately tell if it's truly missing or just late. Enkidu holds its judgment for one cycle, then commits.

### Terminal output

Live output as records are classified:

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Closure Seeker                                            │
  └─────────────────────────────────────────────────────────────┘

  Source:  source.jsonl
  Target:  target.jsonl
  Window:  100 records

  Reading...

  MISSING          [42]         [—]          {"id": "rec_042", ...}
  REORDER ← was missing  [10]         [95]         {"id": "rec_010", ...}
  MISSING          [—]          [67]         {"id": "rec_067", ...}

  ┌─────────────────────────────────────────────────────────────┐
  │  Seeker Report                                             │
  └─────────────────────────────────────────────────────────────┘

  Records processed:  200
  Duration:           5.3ms
  Unresolved:         0 source, 0 target

  Result:  3 incidents
    Missing:   2 records never matched
    Reorder:   1 records arrived out of order (initially looked missing)

  ─────┬──────────┬──────────┬──────────┬────────────────────────────────
   #   │ Type     │ Source   │ Target   │ Payload
  ─────┼──────────┼──────────┼──────────┼────────────────────────────────
     1 │ reorder  │ 10       │ 95       │ {"id": "rec_010", ...}
     2 │ missing  │ 42       │ —        │ {"id": "rec_042", ...}
     3 │ missing  │ —        │ 67       │ {"id": "rec_067", ...}
  ─────┴──────────┴──────────┴──────────┴────────────────────────────────
```

The live lines tell you what's happening as it happens. `MISSING` means a record was held for a full cycle and no match arrived. `REORDER ← was missing` means a record that initially looked missing just found its match — it was in both streams, just displaced.

When streams are identical:

```
  Result:  Coherent
  All records matched across both streams.
```

### Input modes

Same as observer: auto-detects files vs streams.

- **Two files**: interleaved reading, one line from each side per iteration. Prevents race conditions where one file is consumed entirely before the other starts.
- **Pipes/stdin**: threaded reading with a shared queue.

### Options

| Flag | Default | What it does |
|---|---|---|
| `--window` | 100 | Records between each cycle tick. Controls how long unmatched records are held before being promoted to missing |
| `--output`, `-o` | none | Save the incident log as JSON on exit |

### JSON report

```json
{
  "closure_seeker_report": {
    "source": "stream",
    "target": "stream",
    "records_processed": 200,
    "duration_seconds": 0.005
  },
  "summary": {
    "total_incidents": 3,
    "missing": 2,
    "reorder": 1,
    "coherent": false
  },
  "incidents": [
    {
      "type": "reorder",
      "source_index": 10,
      "target_index": 95,
      "payload": "{\"id\": \"rec_010\", ...}"
    },
    {
      "type": "missing",
      "source_index": 42,
      "target_index": null,
      "payload": "{\"id\": \"rec_042\", ...}"
    }
  ]
}
```

---

## Input formats

All three commands accept the same file formats. The reader auto-detects from the file extension.

| Extension | Format | How records are split |
|---|---|---|
| `.jsonl` | JSON Lines | One JSON object per line |
| `.csv` | CSV | One row per record (optionally skip header, select columns) |
| anything else | Plain text | One line per record |

Each line or row becomes one byte record. The SDK doesn't know or care about the structure — it hashes the raw bytes. This means two files with the same content but different formatting (extra whitespace, different column order) will not match, which is by design: if the bytes differ, the records differ.

### CSV options

For CSV files, identity mode supports column selection and custom delimiters:

```bash
# Skip header row
closure identity --header data_a.csv data_b.csv

# Use only columns 0 and 3
closure identity --header --columns 0,3 data_a.csv data_b.csv

# Tab-delimited
closure identity --delimiter $'\t' data_a.tsv data_b.tsv
```

---

## Exit codes

All three commands use the same convention:

| Code | Meaning |
|---|---|
| 0 | Coherent — no incidents found |
| 1 | Error — file not found, invalid arguments, both inputs are stdin |
| 2 | Incidents found — at least one missing or reordered record |

This makes it straightforward to use in scripts:

```bash
closure identity source.jsonl target.jsonl
if [ $? -eq 2 ]; then
    echo "Discrepancies found — check the report."
fi
```

---

## Which command to use

**You have two files and want a full diagnosis.**
Use `closure identity`. It reads both files, runs Gilgamesh, and gives you every incident in one pass. This is the forensic tool — use it after the fact, when you already have both datasets.

**You're monitoring live streams and want to know the moment they diverge.**
Use `closure observer`. It costs almost nothing during normal operation (one Seer per stream, 32 bytes of state). The moment drift is detected, it auto-escalates to Gilgamesh and gives you exact classification. This is the production tool — leave it running, check the output when it pages you.

**You need every record classified as it arrives, in real time.**
Use `closure seeker`. It uses Enkidu to track every record across both streams, holding unresolved ones for a grace period before committing a classification. This is the deep tool — more expensive than observer, but you get the complete lifecycle of every record.

**Rule of thumb**: start with observer. If you need more detail on a specific incident window, rerun that window through identity. If you need the full play-by-play on every record, switch to seeker.

---

## Architecture

```
closure identity  ──→  reader  ──→  Gilgamesh        ──→  formatter  ──→  stdout + JSON
closure observer  ──→  reader  ──→  Seer → Gilgamesh  ──→  stdout + JSON
closure seeker    ──→  reader  ──→  Enkidu            ──→  stdout + JSON
```

The CLI is a thin layer over the SDK. Each command reads input, feeds it to the appropriate SDK component, and formats the output. The math, the sphere, the classification — all of that lives in `closure_sdk/`. The CLI just wires it to files, pipes, and terminals.

| CLI module | SDK components used |
|---|---|
| `identity.py` | `gilgamesh()`, `Seer`, `expose_incident()` |
| `observer.py` | `Seer`, `RetentionWindow`, `gilgamesh()`, `expose_incident()` |
| `seeker.py` | `Enkidu` |
| `reader.py` | (pure I/O — no SDK dependency) |
| `formatter.py` | `IncidentReport`, `IncidentValence` |
