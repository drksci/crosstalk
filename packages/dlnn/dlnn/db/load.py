"""Load an `out/<run_id>/` directory into a results database.

* SQLite (default): stdlib `sqlite3`, file `out/<run_id>/dlnn.sqlite`.
* PostgreSQL: `psycopg` (v3), DSN from `--pg-dsn` or the env var `DLNN_PG_DSN`. Same DDL (`schema.sql`).
  Verified against a live local PostgreSQL 16 server (see docs/build-log.md, section M4a).

JSON files under the run directory stay canonical; the database is a queryable copy of them.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .. import contract

SCHEMA = Path(__file__).with_name("schema.sql")
PG_DSN_ENV = "DLNN_PG_DSN"


class DbUnavailable(RuntimeError):
    """The requested database driver or server cannot be used."""


def connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA.read_text())
    return con


def connect_pg(dsn: str | None = None):
    """Open a psycopg connection (autocommit off) and create the tables if needed. Raises DbUnavailable."""
    dsn = dsn or os.environ.get(PG_DSN_ENV)
    if not dsn:
        raise DbUnavailable(f"--db postgres needs --pg-dsn or ${PG_DSN_ENV} (e.g. 'dbname=dlnn host=/var/run/postgresql')")
    try:
        import psycopg
    except Exception as e:  # noqa: BLE001
        raise DbUnavailable(f"psycopg is not installed ({e}); pip install 'psycopg[binary]' (extra: pg)") from e
    try:
        con = psycopg.connect(dsn)
    except Exception as e:  # noqa: BLE001
        raise DbUnavailable(f"cannot connect to PostgreSQL with the given DSN: {type(e).__name__}: {e}") from e
    with con.cursor() as cur:
        ddl = "\n".join(l for l in SCHEMA.read_text().splitlines() if not l.lstrip().startswith("--"))
        for stmt in [q for q in ddl.split(";") if q.strip()]:
            cur.execute(stmt)
    con.commit()
    return con


class _Db:
    """Tiny adapter: '?' placeholders for both drivers."""

    def __init__(self, con, kind: str):
        self.con, self.kind = con, kind
        self.ph = "?" if kind == "sqlite" else "%s"

    def q(self, sql: str) -> str:
        return sql.replace("?", self.ph)

    def execute(self, sql: str, params: tuple = ()):
        cur = self.con.cursor()
        cur.execute(self.q(sql), params)
        return cur

    def executemany(self, sql: str, rows: list) -> None:
        if rows:
            cur = self.con.cursor()
            cur.executemany(self.q(sql), rows)


def _insert_run(db: _Db, run_dir: Path, args: dict | None) -> str:
    cmp = json.loads((run_dir / "compare.json").read_text())
    rid = cmp["run_id"]
    for tbl in ("llm_outputs", "predictions", "sweeps", "tier_results", "runs"):
        db.execute(f"DELETE FROM {tbl} WHERE run_id = ?", (rid,))
    created = (run_dir / "created_at.txt").read_text().strip() if (run_dir / "created_at.txt").exists() else ""
    db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
               (rid, cmp["task"], created, cmp["model_sha256"], cmp["dataset_sha256"],
                json.dumps(args or {}), json.dumps(cmp)))
    subsets = _subsets(run_dir)
    for t in cmp["tiers"]:
        r = contract.load(run_dir / "tiers" / f"{t}.json")
        db.execute("INSERT INTO tier_results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (rid, t, r.status, r.tool["name"], r.tool["version"], r.n_examples, r.runtime_s,
                    json.dumps(r.metrics), json.dumps(r.fidelity), json.dumps(r.inputs_from),
                    json.dumps(r.warnings)))
        db.executemany("INSERT INTO sweeps VALUES (?,?,?,?,?,?,?)",
                       [(rid, t, sw["axis"], x, *[sw[k][i] if k in sw else None for k in ("accuracy", "mean_kl", "nmse")])
                        for sw in r.sweeps for i, x in enumerate(sw["values"])])
        a = r.arrays
        if "pred" in a:
            lab = _labels(run_dir, a, subsets.get(t))
            ids = subsets.get(t) or list(range(len(a["pred"])))   # example_id = index in the FULL test set
            conf = a["probs"].max(axis=1)
            rows = [(rid, t, int(ids[i]), lab[i], int(a["pred"][i]), float(conf[i]), None, None)
                    for i in range(len(a["pred"]))]
        elif "y_pred" in a:
            yt, yp = a["y_true"][0], a["y_pred"][0]
            rows = [(rid, t, i, None, None, None, float(yt[i]), float(yp[i])) for i in range(len(yp))]
        else:
            rows = []
        db.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)", rows)
    cp = run_dir / "llm" / "commentary.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        db.execute("INSERT INTO llm_outputs VALUES (?,?,?,?,?)", (rid, c["source"], c["backend"], json.dumps(c), created))
    return rid


def load_run(run_dir: str | Path, db_path: str | Path, args: dict | None = None) -> str:
    """Insert (replace) one run into SQLite. Returns the run_id."""
    con = connect(db_path)
    try:
        with con:
            return _insert_run(_Db(con, "sqlite"), Path(run_dir), args)
    finally:
        con.close()


def load_run_pg(run_dir: str | Path, dsn: str | None = None, args: dict | None = None) -> str:
    """Insert (replace) one run into PostgreSQL (psycopg). Returns the run_id. Raises DbUnavailable."""
    con = connect_pg(dsn)
    try:
        rid = _insert_run(_Db(con, "postgres"), Path(run_dir), args)
        con.commit()
        return rid
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def pg_summary(run_id: str, dsn: str | None = None) -> dict[str, Any]:
    """SELECT verification: row counts per table and the tier statuses of one run, read back from PostgreSQL."""
    con = connect_pg(dsn)
    try:
        out: dict[str, Any] = {}
        with con.cursor() as cur:
            for tbl in ("runs", "tier_results", "sweeps", "predictions", "llm_outputs"):
                cur.execute(f"SELECT count(*) FROM {tbl} WHERE run_id = %s", (run_id,))
                out[tbl] = cur.fetchone()[0]
            cur.execute("SELECT tier, status, n_examples, runtime_s FROM tier_results WHERE run_id = %s ORDER BY tier",
                        (run_id,))
            out["tiers"] = [list(r) for r in cur.fetchall()]
        return out
    finally:
        con.close()


def _subsets(run_dir: Path) -> dict[str, list[int]]:
    """{tier: indices into the full test set} for tiers that ran on a subset (written by the pipeline)."""
    p = run_dir / "subsets.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {t: v["indices"] for t, v in d.get("tiers", {}).items()}


def _labels(run_dir: Path, a: dict, idx: list[int] | None = None) -> list:
    p = run_dir / "labels.npy"
    if p.exists():
        import numpy as np
        y = np.load(p)
        return [int(y[i]) for i in (idx if idx is not None else range(len(a["pred"])))]
    return [None] * len(a["pred"])
