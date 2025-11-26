"""Command-line interface for server operations.

Reads server configuration from configs/config.yaml

Usage:
    uv run python server_cli.py submit model.pt
    uv run python server_cli.py submit model.pt --wait
    uv run python server_cli.py status
    uv run python server_cli.py leaderboard
    uv run python server_cli.py rank
    uv run python server_cli.py wait-and-sync
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from src.database import ExperimentDatabase
from src.server import ServerAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"
DEFAULT_DB = "experiments.db"


def load_server_config() -> dict[str, Any]:
    """Load server configuration from configs/config.yaml."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file not found: {CONFIG_PATH}")
        return {}

    cfg = OmegaConf.load(CONFIG_PATH)
    server_section = cfg["server"] if "server" in cfg else {}  # type: ignore[index]
    server_cfg = OmegaConf.to_container(server_section, resolve=True)
    return cast(dict[str, Any], server_cfg) if isinstance(server_cfg, dict) else {}


class ServerCLI:
    """CLI for server operations: submit, status, leaderboard, rank, sync."""

    def __init__(
        self,
        token: str,
        username: str,
        server_url: str,
        leaderboard_path: str,
        db_path: str = DEFAULT_DB,
        max_retries: int = 3,
    ) -> None:
        self.token = token
        self.username = username
        self.server_url = server_url
        self.leaderboard_path = leaderboard_path
        self.db_path = db_path
        self.max_retries = max_retries
        self.server = ServerAPI(token, username, server_url)

    def submit(
        self,
        model_path: str,
        run_name: str | None = None,
        wait: bool = False,
        timeout: int = 1800,
    ) -> bool:
        """Submit a model to the server."""
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        size_mb = Path(model_path).stat().st_size / (1024**2)
        logger.info(f"Submitting model: {model_path} ({size_mb:.2f} MB)")

        result = self.server.submit_model(model_path, max_retries=self.max_retries)

        if not result or not result.get("success"):
            error = result.get("error") if result else "Unknown error"
            logger.error(f"Submission failed: {error}")
            return False

        logger.info("Submission successful!")

        if run_name:
            db = ExperimentDatabase(self.db_path)
            db.update_experiment(
                run_name,
                server_submission_id=result.get("attempt"),
                server_status="pending",
                model_path=model_path,
            )
            logger.info(f"Updated database for run: {run_name}")
            db.close()

        if wait:
            return self._wait_and_sync(run_name, timeout)

        return True

    def status(self, run_name: str | None = None) -> None:
        """Check submission status."""
        attempts = self.server.check_status(max_retries=self.max_retries)

        if not attempts:
            logger.error("Failed to retrieve status")
            return

        logger.info("=" * 60)
        logger.info(f"Submission Status for '{self.username}'")
        logger.info("=" * 60)

        for attempt in attempts:
            logger.info(f"\nAttempt #{attempt['attempt']}:")
            logger.info(f"  Status: {attempt['status']}")
            logger.info(f"  Submitted: {attempt.get('submitted_at', 'N/A')}")

            if isinstance(attempt.get("model_size"), int | float):
                logger.info(f"  Model Size: {attempt['model_size']:.2f} MB")
            if isinstance(attempt.get("score"), int | float):
                logger.info(f"  F1 Score: {attempt['score']:.4f}")

        if run_name and attempts:
            latest = attempts[-1]
            db = ExperimentDatabase(self.db_path)
            update: dict[str, Any] = {"server_status": latest["status"]}

            if isinstance(latest.get("model_size"), int | float):
                update["server_model_size_mb"] = latest["model_size"]
            if isinstance(latest.get("score"), int | float):
                update["server_f1_score"] = latest["score"]

            db.update_experiment(run_name, **update)
            logger.info(f"\nUpdated database for run: {run_name}")
            db.close()

    def leaderboard(self, top_n: int = 20, save_snapshot: bool = False) -> None:
        """View the leaderboard."""
        entries = self.server.scrape_leaderboard(self.leaderboard_path, self.max_retries)

        if not entries:
            logger.error("Failed to retrieve leaderboard")
            return

        logger.info("=" * 70)
        logger.info(f"{'Public Leaderboard':^70}")
        logger.info("=" * 70)
        logger.info(f"{'Rank':<6} {'Team':<35} {'F1 Score':<12} {'Size (MB)':<10}")
        logger.info("-" * 70)

        for entry in entries[:top_n]:
            marker = ">>>" if entry["team_name"] == self.username else "   "
            logger.info(
                f"{marker} {entry['rank']:<3} {entry['team_name']:<35} "
                f"{entry['f1_score']:<12.4f} {entry['model_size_mb']:<10.2f}"
            )

        logger.info("=" * 70)

        if save_snapshot:
            db = ExperimentDatabase(self.db_path)
            db.save_leaderboard_snapshot(entries)
            logger.info("Saved leaderboard snapshot to database")
            db.close()

    def rank(self) -> None:
        """Get our current rank."""
        entry = self.server.get_our_rank(self.leaderboard_path)

        if not entry:
            logger.info(f"Team '{self.username}' not found on leaderboard")
            return

        logger.info("=" * 50)
        logger.info(f"Team: {self.username}")
        logger.info("=" * 50)
        logger.info(f"Rank: #{entry['rank']}")
        logger.info(f"F1 Score: {entry['f1_score']:.4f}")
        logger.info(f"Model Size: {entry['model_size_mb']:.2f} MB")
        logger.info("=" * 50)

    def wait_and_sync(self, run_name: str | None = None, timeout: int = 1800) -> bool:
        """Wait for pending submission and sync results to database."""
        if not run_name:
            db = ExperimentDatabase(self.db_path)
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT run_name FROM experiments
                WHERE server_status IS NULL OR server_status = 'pending'
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()
            db.close()

            if row:
                run_name = row[0]
                logger.info(f"Auto-detected run to sync: {run_name}")
            else:
                logger.warning("No pending run found. Use --run-name to specify.")

        return self._wait_and_sync(run_name, timeout)

    def _wait_and_sync(self, run_name: str | None, timeout: int) -> bool:
        """Internal: wait for evaluation and sync metrics."""
        logger.info("Checking submission status...")
        attempts = self.server.check_status(max_retries=self.max_retries)

        if not attempts:
            logger.error("Failed to retrieve status")
            return False

        latest = attempts[-1]
        logger.info(f"Latest submission (Attempt #{latest['attempt']}): {latest['status']}")

        if latest["status"] == "pending":
            logger.info(f"Waiting for evaluation (timeout: {timeout}s)...")

            if not self.server.wait_for_evaluation(timeout=timeout, check_interval=30):
                logger.error("Evaluation timed out or failed")
                return False

            logger.info("Evaluation complete!")

        elif latest["status"] != "successful":
            logger.error(f"Submission status: {latest['status']}")
            return False

        time.sleep(3)
        logger.info("Fetching metrics from leaderboard...")
        metrics = self.server.get_metrics_from_leaderboard(self.leaderboard_path)

        if not metrics:
            logger.error("Failed to fetch metrics from leaderboard")
            return False

        logger.info(f"Rank: #{metrics['server_rank']}")
        logger.info(f"F1 Score: {metrics['server_f1_score']:.4f}")
        logger.info(f"Model Size: {metrics['server_model_size_mb']:.2f} MB")

        if run_name:
            db = ExperimentDatabase(self.db_path)
            db.update_experiment(run_name, server_status="successful", **metrics)
            logger.info(f"Updated database for run: {run_name}")
            db.close()

        logger.info("Sync complete!")
        return True


def main() -> None:
    # Load defaults from config.yaml
    server_cfg = load_server_config()
    default_token = server_cfg.get("token", "")
    default_username = server_cfg.get("username", "")
    default_url = server_cfg.get("url", "http://hadi.cs.virginia.edu:8000")
    default_leaderboard = server_cfg.get("leaderboard_path", "/leaderboard3")
    default_timeout = server_cfg.get("timeout", 1800)

    # Check if config has valid credentials
    has_config = default_token and default_token != "YOUR_TOKEN_HERE"

    parser = argparse.ArgumentParser(
        description="CLI for model submission and tracking. "
        "Reads defaults from configs/config.yaml."
    )

    # Make token/username optional if config has them
    parser.add_argument(
        "--token",
        default=default_token if has_config else None,
        required=not has_config,
        help="Auth token (reads from config.yaml if set)",
    )
    parser.add_argument(
        "--username",
        default=default_username if has_config else None,
        required=not has_config,
        help="Team/username (reads from config.yaml if set)",
    )
    parser.add_argument("--server-url", default=default_url, help="Server URL")
    parser.add_argument("--leaderboard-path", default=default_leaderboard, help="Leaderboard path")
    parser.add_argument("--db-path", default=DEFAULT_DB, help="Database path")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    submit_p = subparsers.add_parser("submit", help="Submit a model")
    submit_p.add_argument("model_path", help="Path to TorchScript model (.pt)")
    submit_p.add_argument("--run-name", help="Run name for database tracking")
    submit_p.add_argument("--wait", action="store_true", help="Wait for evaluation")
    submit_p.add_argument("--timeout", type=int, default=default_timeout, help="Timeout (seconds)")

    status_p = subparsers.add_parser("status", help="Check submission status")
    status_p.add_argument("--run-name", help="Run name for database update")

    lb_p = subparsers.add_parser("leaderboard", help="View leaderboard")
    lb_p.add_argument("--top-n", type=int, default=20, help="Number of entries to show")
    lb_p.add_argument("--save-snapshot", action="store_true", help="Save snapshot to database")

    subparsers.add_parser("rank", help="Get your current rank")

    sync_p = subparsers.add_parser("wait-and-sync", help="Wait for evaluation and sync to DB")
    sync_p.add_argument("--run-name", help="Run name to update (auto-detects if not provided)")
    sync_p.add_argument("--timeout", type=int, default=default_timeout, help="Timeout (seconds)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        if has_config:
            logger.info(f"\nUsing config from: {CONFIG_PATH}")
            logger.info(
                f"  Token: {'*' * 8}...{default_token[-4:] if len(default_token) > 4 else '****'}"
            )
            logger.info(f"  Username: {default_username}")
            logger.info(f"  Server: {default_url}")
        else:
            logger.warning(f"\nNo credentials in {CONFIG_PATH}. Use --token and --username.")
        return

    cli = ServerCLI(
        token=args.token,
        username=args.username,
        server_url=args.server_url,
        leaderboard_path=args.leaderboard_path,
        db_path=args.db_path,
        max_retries=args.max_retries,
    )

    if args.command == "submit":
        cli.submit(args.model_path, args.run_name, args.wait, args.timeout)
    elif args.command == "status":
        cli.status(args.run_name)
    elif args.command == "leaderboard":
        cli.leaderboard(args.top_n, args.save_snapshot)
    elif args.command == "rank":
        cli.rank()
    elif args.command == "wait-and-sync":
        cli.wait_and_sync(args.run_name, args.timeout)


if __name__ == "__main__":
    main()
