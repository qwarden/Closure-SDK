"""Tests for the Closure CLI — identity, observer, seeker."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

DATA = Path("data")
CLI = [sys.executable, "-m", "closure_cli"]


# ── Identity ─────────────────────────────────────────────────

class TestIdentity:
    def _run(self, src, tgt, *extra):
        result = subprocess.run(
            [*CLI, "identity", str(src), str(tgt), *extra],
            capture_output=True, text=True, timeout=30,
        )
        return result

    def test_coherent(self):
        r = self._run(DATA / "test_a.jsonl", DATA / "test_a.jsonl")
        assert r.returncode == 0
        assert "Coherent" in (r.stdout + r.stderr)

    def test_test_pair(self):
        r = self._run(DATA / "test_a.jsonl", DATA / "test_b.jsonl")
        assert r.returncode == 2
        out = r.stdout + r.stderr
        assert "3 incidents" in out
        assert "Missing:" in out
        assert "Reorder:" in out

    def test_walmart_100(self):
        r = self._run(DATA / "walmart_a_100.jsonl", DATA / "walmart_b_100.jsonl")
        assert r.returncode == 2
        out = r.stdout + r.stderr
        assert "3 incidents" in out

    def test_stress(self):
        r = self._run(DATA / "test_stress_a.jsonl", DATA / "test_stress_b.jsonl")
        assert r.returncode == 2
        out = r.stdout + r.stderr
        assert "28 incidents" in out
        assert "Missing:" in out
        assert "Reorder:" in out

    def test_json_output(self, tmp_path):
        out_file = tmp_path / "report.json"
        r = self._run(
            DATA / "test_a.jsonl", DATA / "test_b.jsonl",
            "--output", str(out_file),
        )
        assert r.returncode == 2
        report = json.loads(out_file.read_text())
        assert report["summary"]["total_incidents"] == 3
        assert report["summary"]["missing"] == 2
        assert report["summary"]["reorder"] == 1
        assert len(report["incidents"]) == 3

    def test_missing_file(self):
        r = self._run(DATA / "nonexistent.jsonl", DATA / "test_b.jsonl")
        assert r.returncode == 1


# ── Observer ─────────────────────────────────────────────────

class TestObserver:
    def _run(self, src, tgt, *extra):
        result = subprocess.run(
            [*CLI, "observer", str(src), str(tgt), *extra],
            capture_output=True, text=True, timeout=30,
        )
        return result

    def test_coherent(self):
        r = self._run(DATA / "test_a.jsonl", DATA / "test_a.jsonl")
        assert r.returncode == 0
        assert "Coherent" in (r.stdout + r.stderr)
        assert "Escalations:        0" in (r.stdout + r.stderr)

    def test_drift_detected(self):
        r = self._run(DATA / "test_a.jsonl", DATA / "test_b.jsonl")
        assert r.returncode == 2
        out = r.stdout + r.stderr
        assert "DRIFT DETECTED" in out
        assert "Escalating" in out
        assert "3 incidents" in out

    def test_json_output(self, tmp_path):
        out_file = tmp_path / "report.json"
        r = self._run(
            DATA / "test_a.jsonl", DATA / "test_b.jsonl",
            "--output", str(out_file),
        )
        assert r.returncode == 2
        report = json.loads(out_file.read_text())
        assert report["summary"]["coherent"] is False
        assert report["summary"]["total_incidents"] == 3


# ── Seeker ───────────────────────────────────────────────────

class TestSeeker:
    def _run(self, src, tgt, *extra):
        result = subprocess.run(
            [*CLI, "seeker", str(src), str(tgt), *extra],
            capture_output=True, text=True, timeout=30,
        )
        return result

    def test_coherent(self):
        r = self._run(DATA / "test_a.jsonl", DATA / "test_a.jsonl")
        assert r.returncode == 0
        assert "Coherent" in (r.stdout + r.stderr)

    def test_finds_incidents(self):
        r = self._run(DATA / "test_a.jsonl", DATA / "test_b.jsonl")
        assert r.returncode == 2
        out = r.stdout + r.stderr
        # Seeker should find the 2 true missing records at minimum
        assert "missing" in out.lower()


# ── Reader ───────────────────────────────────────────────────

class TestReader:
    def test_read_jsonl(self, tmp_path):
        from closure_cli.reader import read_file
        f = tmp_path / "test.jsonl"
        f.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
        records = read_file(f)
        assert len(records) == 3
        assert records[0] == b'{"a":1}'

    def test_read_csv(self, tmp_path):
        from closure_cli.reader import read_file
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        records = read_file(f, header=True)
        assert len(records) == 2
        assert records[0] == b"1,2,3"

    def test_read_csv_columns(self, tmp_path):
        from closure_cli.reader import read_file
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        records = read_file(f, header=True, columns=[0, 2])
        assert len(records) == 2
        assert records[0] == b"1,3"

    def test_read_text(self, tmp_path):
        from closure_cli.reader import read_file
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld\n")
        records = read_file(f)
        assert len(records) == 2
        assert records[0] == b"hello"
