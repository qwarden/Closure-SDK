# Closure DNA

Closure DNA is the standalone persistent memory/database module in the Closure repo.

Status:
- frozen as a standalone product surface for now
- only stability, bug fixes, and documentation alignment should land here
- deeper feature work resumes after the higher `closure_ea` layer settles

It gives you:
- typed tables
- transactions
- SQL execution
- a local web workbench
- git-like table snapshots, history, and restore
- built-in integrity identity
- resonance search over stored rows

It stands on its own and can be demoed directly.

Its low-level engine lives in the shared monorepo Rust core in `rust/`, exposed to Python as `closure_rs`, and Closure DNA builds its database surface on top of that shared core.

## What It Is

Closure DNA is a local embedded database for structured data.

The storage engine is:
- columnar
- typed
- persisted on disk as a `.cdb` directory

The geometric layer gives each row and table a compositional identity on `S^3`.
That identity powers:
- fast integrity checks
- drift decomposition
- repair
- resonance search
- append-only version history
- named snapshots and exact restore

The product story is not just "SQLite with geometry."
It is:
- a local embedded database
- with built-in integrity
- with built-in similarity search
- with built-in git-like table versioning
- with a local web workbench for demos and inspection

## What It Supports

### Types
- `i64`
- `f64`
- `bytes`

### Core database operations
- create/open/drop database tables
- insert/update/delete
- add column with default backfill
- transactions
- read snapshots
- append-only table history
- named snapshots and restore
- compact / audit / repair
- import / export
- joins
- group by + aggregates
- null support

### SQL
Closure DNA now speaks real SQL through the parser layer.

Supported SQL includes:
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
- multi-statement SQL scripts
- `BEGIN / COMMIT / ROLLBACK`

### DNA-specific SQL
- `SELECT IDENTITY() FROM table`
- `SELECT DRIFT() FROM table`
- `SELECT DECOMPOSE_DRIFT() FROM table`
- `AUDIT table`
- `COMPACT table`
- `INSPECT ROW n FROM table`
- `SELECT * FROM table RESONATE NEAR (...) LIMIT k`

### Git-like table versioning
Every table also supports:
- `snapshot(name)` — capture the current table state by name
- `history(limit)` — read the append-only operation history
- `snapshots()` — list named saved states
- `restore_snapshot(name)` — restore exactly to a prior named state

This is native product surface, not an external backup tool.

## Quick Start

### Python API

```python
from closure_dna import Database

db = Database.create("shop.cdb")

db.create_table(
    "people",
    [
        {"name": "id", "type": "i64", "primary": True},
        {"name": "name", "type": "bytes"},
        {"name": "city", "type": "bytes", "indexed": True},
        {"name": "age", "type": "f64"},
    ],
)

db.table("people").insert([1, b"Alice", b"Tokyo", 31.0])
db.table("people").insert([2, b"Bob", b"Paris", 22.0])

rows = db.execute("SELECT name FROM people WHERE city = 'Tokyo'").rows
```

### CLI

```bash
closure-dna create-db shop.cdb
closure-dna create-table shop.cdb people '[{"name":"id","type":"i64","primary":true},{"name":"name","type":"bytes"}]'
closure-dna insert shop.cdb people '[1,"Alice"]'
closure-dna sql shop.cdb "SELECT * FROM people"
closure-dna web
```

### Web workbench

Run:

```bash
closure-dna web
```

The web UI includes:
- built-in demo database browser
- table browser
- row editing
- row deletion
- SQL workbench
- schema view
- schema relationship view

The workbench is for demos and product inspection:
- open a built demo database directly
- browse tables and rows
- run SQL
- inspect integrity and repair paths

## Built-in Demo Databases

Built demo databases live under:

```text
closure_dna/demo/databases/
```

Current demos:
- `browser_profile.cdb`
- `chat_app.cdb`
- `music_streaming.cdb`

The web UI can open these directly from the left-hand database list.

## Package Layout

```text
closure_dna/
├── table.py        # low-level typed table wrapper over closure_rs.Table
├── database.py     # multi-table database surface
├── sql.py          # SQL parser/executor layer
├── web.py          # local web workbench
├── cli.py          # command line entrypoint
├── repl.py         # interactive REPL
├── demo/           # built demo databases + registry
├── demo_data/      # source datasets for demos
└── tests/          # module test suite
```

Shared Rust core:

```text
rust/
└── src/
    ├── lib.rs
    └── groups/
        └── sphere.rs
closure_dna/
└── rust/
    └── src/
        ├── table.rs
        └── composition_tree.rs
```

## Relationship To The Rust Core

Closure DNA now lives as its own standalone module.

It owns its own database surface, semantics, SQL layer, demos, and workbench.
The correct architecture is:
- `closure_dna` is the memory/database layer of the computer stack
- `closure_ea/vm` remains the execution layer in the umbrella repo
- both share the common Rust foundation in `closure_rs`

## What The Geometry Adds

Traditional embedded databases give you storage and query execution.

Closure DNA also gives you:
- table identity in 32 bytes
- integrity drift checks
- drift decomposition
- repair from persisted columns
- resonance search over stored rows
- weighted composite resonance search
- persisted composite-key acceleration sidecars
- append-only table history
- named snapshots and exact restore

Those are not wrappers around another database.
They come from the engine itself.

## Current Scope

Closure DNA is:
- a local database
- SQL-capable
- parser-backed
- versioned at the table layer
- geometric at the engine level
- demoable through a built-in local web workbench

Closure DNA is not trying to be:
- a client/server database
- a distributed database
- a drop-in clone of SQLite internals

It is a different database architecture with a familiar SQL surface.

## Documentation

For the fuller product reference, see:
- [CLOSURE_DNA.md](CLOSURE_DNA.md)
