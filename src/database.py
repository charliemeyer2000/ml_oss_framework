"""SQLite database for experiment tracking."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

ALLOWED_METRICS = frozenset(
    {
        "server_f1_score",
        "best_val_f1",
        "best_val_accuracy",
        "best_val_loss",
        "training_duration_seconds",
        "model_size_mb",
        "num_parameters",
        "server_rank",
    }
)


class ExperimentDatabase:
    """SQLite database for tracking ML experiments."""

    def __init__(self, db_path: str = "experiments.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                -- Model config
                model_name TEXT,
                num_classes INTEGER,
                dropout REAL,
                -- Training config
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                weight_decay REAL,
                scheduler TEXT,
                -- Data config
                data_root TEXT,
                train_split REAL,
                img_size INTEGER,
                -- Training results
                best_epoch INTEGER,
                best_val_f1 REAL,
                best_val_accuracy REAL,
                best_val_loss REAL,
                training_duration_seconds REAL,
                -- Model info
                num_parameters INTEGER,
                model_size_mb REAL,
                model_path TEXT,
                -- Device info
                device TEXT,
                -- Server submission
                server_submission_id INTEGER,
                server_status TEXT,
                server_submitted_at TEXT,
                server_f1_score REAL,
                server_model_size_mb REAL,
                server_rank INTEGER,
                -- Notes
                notes TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                val_loss REAL,
                val_f1 REAL,
                val_accuracy REAL,
                learning_rate REAL,
                timestamp TEXT,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id),
                UNIQUE(run_id, epoch)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rank INTEGER,
                team_name TEXT,
                f1_score REAL,
                model_size_mb REAL
            )
        """)

        self.conn.commit()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def create_experiment(self, run_name: str, config: dict[str, Any]) -> int:
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiments (
                    run_name, timestamp,
                    model_name, num_classes, dropout,
                    epochs, batch_size, learning_rate, weight_decay, scheduler,
                    data_root, train_split, img_size,
                    device, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_name,
                    datetime.now().isoformat(),
                    config.get("model_name"),
                    config.get("num_classes"),
                    config.get("dropout"),
                    config.get("epochs"),
                    config.get("batch_size"),
                    config.get("learning_rate"),
                    config.get("weight_decay"),
                    config.get("scheduler"),
                    config.get("data_root"),
                    config.get("train_split"),
                    config.get("img_size"),
                    config.get("device"),
                    config.get("notes"),
                ),
            )
            return cursor.lastrowid or 0

    def update_experiment(self, run_name: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        with self.transaction() as conn:
            cursor = conn.cursor()
            set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
            values = list(kwargs.values()) + [run_name]
            cursor.execute(f"UPDATE experiments SET {set_clause} WHERE run_name = ?", values)

    def add_epoch_history(self, run_name: str, epoch: int, metrics: dict[str, float]) -> None:
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT run_id FROM experiments WHERE run_name = ?", (run_name,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Experiment {run_name} not found")
            run_id = result[0]

            cursor.execute(
                """
                INSERT OR REPLACE INTO training_history (
                    run_id, epoch, train_loss, val_loss, val_f1, val_accuracy, learning_rate, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    epoch,
                    metrics.get("train_loss"),
                    metrics.get("val_loss"),
                    metrics.get("val_f1"),
                    metrics.get("val_accuracy"),
                    metrics.get("learning_rate"),
                    datetime.now().isoformat(),
                ),
            )

    def get_experiment(self, run_name: str) -> dict[str, Any] | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE run_name = ?", (run_name,))
        result = cursor.fetchone()
        return dict(result) if result else None

    def get_recent_experiments(self, limit: int = 10) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_best_experiments(
        self, metric: str = "server_f1_score", limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get best experiments sorted by a metric.

        Args:
            metric: Column to sort by. Must be one of the allowed metrics to prevent SQL injection.
            limit: Maximum number of results to return.

        Raises:
            ValueError: If metric is not in the allowed whitelist.
        """
        if metric not in ALLOWED_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. Allowed metrics: {sorted(ALLOWED_METRICS)}"
            )
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT * FROM experiments WHERE {metric} IS NOT NULL ORDER BY {metric} DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_training_history(self, run_name: str) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT h.* FROM training_history h
            JOIN experiments e ON h.run_id = e.run_id
            WHERE e.run_name = ?
            ORDER BY h.epoch
        """,
            (run_name,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def save_leaderboard_snapshot(self, leaderboard_data: list[dict[str, Any]]) -> None:
        with self.transaction() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            for entry in leaderboard_data:
                cursor.execute(
                    """
                    INSERT INTO leaderboard_snapshots (timestamp, rank, team_name, f1_score, model_size_mb)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        entry.get("rank"),
                        entry.get("team_name"),
                        entry.get("f1_score"),
                        entry.get("model_size_mb"),
                    ),
                )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ExperimentDatabase":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
