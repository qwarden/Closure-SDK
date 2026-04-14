# Closure DNA

Product reference for the Closure DNA module.

Status:
- frozen as the standalone database/memory layer for this refactor cycle
- treat this document as an internal product reference for `closure_dna/`
- only maintenance, cleanup, and verification work should happen here until the higher `closure_ea` layer is finished

Closure DNA is the standalone persistent memory/database module,
built on the shared geometric Rust core.

The important boundary is:
- shared Rust core: `closure_rs`
- memory/database layer: `closure_dna`

## Architecture

```text
SQL / CLI / Web
        |
        v
closure_dna Python surface
        |
        v
closure_rs Rust extension
        |
        v
typed columnar geometric engine
```

The Rust core lives in the monorepo `rust/` crate.
That is shared implementation, not product ownership confusion.

`closure_dna` owns:
- the database surface
- SQL execution
- table versioning and restore
- workbench
- demos
- database semantics

`closure_rs` owns:
- embedding
- typed table engine
- persistence primitives
- identity
- resonance
- repairable geometric state

## Storage Model

A Closure DNA database is a directory-backed `.cdb`.

Example:

```text
shop.cdb/
├── people.cdna/
├── orders.cdna/
└── ...
```

Each table is persisted as its own typed storage directory.

The engine is:
- typed
- columnar
- local
- file-backed

## Data Types

Closure DNA currently exposes:
- `i64`
- `f64`
- `bytes`

These are the real engine types.

SQL also maps standard names onto them where appropriate:
- `INTEGER` -> `i64`
- `REAL` -> `f64`
- `TEXT` -> `bytes`
- `BLOB` -> `bytes`

## Python API

Main public objects:
- `Database`
- `Transaction`
- `Table`
- `ResonanceHit`
- `SQLResult`
- `execute`

### Database

Main methods:
- `Database.create(path)`
- `Database.open(path)`
- `db.create_table(name, schema)`
- `db.drop_table(name)`
- `db.tables()`
- `db.schema(name)`
- `db.table(name)`
- `db.select(...)`
- `db.join(...)`
- `db.group_by(...)`
- `db.subquery(...)`
- `db.update_where(...)`
- `db.delete_where(...)`
- `db.add_column(...)`
- `db.compact(...)`
- `db.audit(...)`
- `db.repair(...)`
- `db.info(...)`
- `db.import_table(...)`
- `db.export_table(...)`
- `db.execute(sql)`
- `db.transaction()`
- `db.read_transaction()`

### Table

Main methods:
- `Table.create(path, schema)`
- `Table.open(path)`
- `table.insert(values)`
- `table.insert_many(rows)`
- `table.insert_columns(columns)`
- `table.get_row(row_id)`
- `table.get_f64(row_id, column)`
- `table.get_i64(row_id, column)`
- `table.get_bytes(row_id, column)`
- `table.filter_equals(column, value)`
- `table.filter_cmp(column, op, value)`
- `table.sum(column)`
- `table.avg(column)`
- `table.argsort(column, descending=False)`
- `table.search(values, k=5)`
- `table.snapshot(name=None)`
- `table.history(limit=None)`
- `table.snapshots()`
- `table.restore_snapshot(name)`
- `table.identity()`
- `table.check()`
- `table.check_hopf()`
- `table.inspect_row(row_id)`
- `table.count()`
- `table.save()`

## SQL Surface

Standard SQL supported today:

- `CREATE TABLE`
- `DROP TABLE`
- `ALTER TABLE ... ADD COLUMN`
- `SELECT`
- `INSERT`
- `UPDATE`
- `DELETE`
- `JOIN`
- `LEFT JOIN`
- `RIGHT JOIN`
- `FULL OUTER JOIN`
- `GROUP BY`
- `HAVING`
- `DISTINCT`
- `LIKE`
- `BETWEEN`
- `EXISTS`
- `UNION`
- nested subqueries
- multi-statement scripts
- `BEGIN`
- `COMMIT`
- `ROLLBACK`

### DNA-specific SQL

- `SELECT IDENTITY() FROM table`
- `SELECT DRIFT() FROM table`
- `SELECT DECOMPOSE_DRIFT() FROM table`
- `AUDIT table`
- `COMPACT table`
- `INSPECT ROW n FROM table`
- `SELECT * FROM table RESONATE NEAR (...) LIMIT k`

These are product features, not standard SQL.
They intentionally expose the geometric capabilities of the engine.

## Web Workbench

The local web UI supports:
- opening built-in demo databases
- browsing tables
- paging rows
- editing rows
- deleting rows
- creating tables
- adding columns
- running SQL
- viewing schema details
- viewing schema relationships
- audit / repair / compact actions

Run it with:

```bash
closure-dna web
```

## CLI

Main commands include:
- `create-db`
- `create-table`
- `add-column`
- `tables`
- `schema`
- `count`
- `check`
- `audit`
- `repair`
- `info`
- `get`
- `insert`
- `update`
- `delete`
- `update-where`
- `delete-where`
- `group-by`
- `compact`
- `filter`
- `select`
- `join`
- `sum`
- `avg`
- `sort`
- `search`
- `export`
- `import`
- `sql`
- `repl`
- `web`
- `demo-databases`
- `build-demo-db`
- `web-demo`

## Demo Databases

Built demo databases:
- `browser_profile`
- `chat_app`
- `music_streaming`

They live in:

```text
closure_dna/demo/databases/
```

Source datasets live in:

```text
closure_dna/demo_data/
```

## Geometric Capabilities

What makes Closure DNA different from a normal local database:

- table identity in 32 bytes
- drift as an integrity measure
- drift decomposition
- repair from persisted column data
- resonance search over stored rows
- weighted composite resonance search
- persisted composite-key acceleration sidecars
- append-only table history
- named snapshots and restore

The product-level story is:
- local embedded database
- built-in integrity identity
- built-in resonance search
- built-in git-like table versioning
- built-in local web workbench

That combination is the point of the product surface.

## Versioned Tables

Closure DNA tables are versioned directly at the table layer.

Native table operations:
- `snapshot(name)` — save a named table state
- `history(limit)` — inspect append-only history entries
- `snapshots()` — list available saved states
- `restore_snapshot(name)` — restore the table to that exact state

This is not a separate backup product bolted on from the outside.
The versioning surface lives with the table.

This is why "git-like" is the right mental model:
- snapshots are named states
- history is append-only
- restore returns to an exact prior state

This is why the Rust core is geometric instead of B-tree based.

## Packaging Notes

Closure DNA now lives as its own standalone memory/database module.

The correct architecture story is:
- `closure_dna` owns the database surface
- `closure_ea/vm` remains a separate execution layer elsewhere in the repo
- `closure_rs` remains the shared Rust foundation

This keeps the runtime stack honest: memory and execution are distinct layers
of one computer, not unrelated sibling products.

## Benchmark Story

The benchmark story for Closure DNA should cover two layers:

1. Core engine comparisons
- insert
- get
- integrity check
- indexed filter
- aggregate
- sort
- update/delete
- cold start

2. Product-native capabilities
- resonance search
- table identity
- snapshot creation
- history reads
- snapshot listing
- restore

SQLite is still the right comparison for the first layer.
For the second layer, many operations are product-exclusive because SQLite
does not provide those capabilities natively as part of the same local table API.
