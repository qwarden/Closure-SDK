# Test Results — Closure SDK v1.0

**Date**: 2026-03-19
**Python**: 3.13.7
**Platform**: Linux 6.17.0-19-generic
**Duration**: 0.76s

## Summary

| Suite | Tests | Status |
|---|---|---|
| test_sdk.py | 16 | PASS |
| test_retention.py | 8 | PASS |
| test_stream_classifier.py | 13 | PASS |
| test_theorems.py | 32 | PASS |
| test_groups.py | 37 | PASS |
| test_path.py | 14 | PASS |
| test_hierarchy.py | 10 | PASS |
| test_fast_path.py | 10 | PASS |
| test_stream_monitor.py | 9 | PASS |
| **Total** | **151** | **ALL PASS** |

## Coverage

- **Algebra**: group axioms (identity, inverse, associativity, distance) across Circle, Sphere, Torus, Hybrid
- **Convergence**: Theorem 1 (convergence) and Theorem 2 (uniformity) across groups and corruption rates
- **Paths**: recovery, closure property, range checks, localization, input validation
- **Hierarchy**: clean/corrupted sequences, logarithmic check count, length mismatch
- **Fast path**: raw-bytes and element-based construction, validation
- **Stream monitor**: batch/stream equivalence, sigma, reset, group mismatch
- **SDK surface**: embed, compose, invert, sigma, diff, compare, Seer, Oracle, Witness, expose, expose_incident
- **Retention**: window append/flatten, block map, bounds, localize_all (missing, reorder, coherent)
- **Stream classifier**: coherent matching, missing promotion, grace period, late reclassification, chained errors, exchange symmetry
