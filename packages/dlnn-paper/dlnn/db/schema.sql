-- DLNN results database. Portable DDL: runs on SQLite (default) and PostgreSQL.
-- JSON files under out/<run_id>/ remain canonical; the DB is a queryable copy.
-- JSON-valued columns are TEXT (cast to jsonb in Postgres if desired).

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    task          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    model_sha256  TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    args_json     TEXT NOT NULL,
    compare_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tier_results (
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    tier          TEXT NOT NULL,
    status        TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    tool_version  TEXT NOT NULL,
    n_examples    INTEGER NOT NULL,
    runtime_s     DOUBLE PRECISION NOT NULL,
    metrics_json  TEXT NOT NULL,
    fidelity_json TEXT NOT NULL,
    inputs_from_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    PRIMARY KEY (run_id, tier)
);

CREATE TABLE IF NOT EXISTS sweeps (
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    tier      TEXT NOT NULL,
    axis      TEXT NOT NULL,
    x         DOUBLE PRECISION NOT NULL,
    accuracy  DOUBLE PRECISION,
    mean_kl   DOUBLE PRECISION,
    nmse      DOUBLE PRECISION,
    PRIMARY KEY (run_id, tier, axis, x)
);

-- classification: one row per test example; regression: one row per test time step.
CREATE TABLE IF NOT EXISTS predictions (
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    tier       TEXT NOT NULL,
    example_id INTEGER NOT NULL,
    label      INTEGER,
    pred       INTEGER,
    confidence DOUBLE PRECISION,
    y_true     DOUBLE PRECISION,
    y_pred     DOUBLE PRECISION,
    PRIMARY KEY (run_id, tier, example_id)
);

CREATE TABLE IF NOT EXISTS llm_outputs (
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    source     TEXT NOT NULL,      -- 'template' | 'model'
    backend    TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, backend)
);
