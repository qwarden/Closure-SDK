# Data Catalog

## Test Pairs

### test_a / test_b (100 records each)
Self-documenting tags. 3 isolated incidents.

| # | Type    | Source | Target | Payload tag     |
|---|---------|--------|--------|-----------------|
| 1 | reorder | [10]   | [95]   | reorder         |
| 2 | missing | [50]   | —      | only_in_a       |
| 3 | missing | —      | [75]   | only_in_b       |

### walmart_a_100 / walmart_b_100 (100 records each)
Same fault pattern as test_a/b, real Walmart schema.

| # | Type    | Source | Target | Description              |
|---|---------|--------|--------|--------------------------|
| 1 | reorder | [10]   | [95]   | Record moved             |
| 2 | missing | [50]   | —      | Great Value Whole Milk   |
| 3 | missing | —      | [75]   | Equate Ibuprofen         |

### walmart_pos / walmart_wms (50,000 records each)
POS vs Warehouse Management. 2 isolated field-level mutations.

| # | Type    | Position | Description                                      |
|---|---------|----------|--------------------------------------------------|
| 1 | missing | [31247]  | Completely different records at same position     |
| 2 | missing | [41803]  | Subtle amount mutation ($469.98 vs $234.99)       |

### walmart_a_500k / walmart_b_500k (500,000 vs 500,015)
Heavy consecutive fault injection. 6 contiguous blocks in both + 15 extra in B.

| Block | Positions       | Count | Pattern              |
|-------|-----------------|-------|----------------------|
| 1     | 80000–80014     | 15    | Replaced in both     |
| 2     | 160000–160009   | 10    | Replaced in both     |
| 3     | 250000–250019   | 20    | Replaced in both     |
| 4     | 310000–310007   | 8     | Replaced in both     |
| 5     | 420000–420011   | 12    | Replaced in both     |
| 6     | 475000–475009   | 10    | Replaced in both     |
| 7     | 500000–500014   | 15    | Extra records in B   |

Total: 165 incidents (all missing, no reorders).

### walmart_primary_500k / walmart_replica_500k (500,000 each)
Primary/replica divergence. 3 contiguous blocks of field-level mutations.

| Block | Positions       | Count |
|-------|-----------------|-------|
| 1     | 120000–120049   | 50    |
| 2     | 310000–310019   | 20    |
| 3     | 450000–450009   | 10    |

Total: 160 incidents (80 pairs, each position has a different record in each file).

### test_stress_a / test_stress_b (200 records each)
Stress test for consecutive and mixed incident patterns.

| Pattern              | Positions   | Count | Description                       |
|----------------------|-------------|-------|-----------------------------------|
| Isolated missing     | [30]        | 1     | Single record absent from B       |
| Isolated reorder     | [50]→[180]  | 1     | Single record moved               |
| Consecutive missing  | [80–84]     | 5+5   | Block replaced in both directions |
| Adjacent mixed       | [150–152]   | 2+1   | missing, reorder, missing in a row|
| Back-to-back reorder | [160–161]   | 2     | Two reorders at adjacent positions|

Total: 28 incidents (24 missing, 4 reorder).

## Reference Files (no pair)

| File | Records | Description |
|------|---------|-------------|
| gharchive_2025-01-01_10k.jsonl | 10,000 | GitHub Archive events |
| wikipedia_recentchange_window.jsonl | 5,000 | Wikipedia edit stream |
| torus_finalized_window.jsonl | 100 | Torus blockchain blocks |
| torus_chain_stakes_snapshot.json | 203 | Torus stakes snapshot |
