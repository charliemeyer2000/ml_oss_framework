"""Server API client with retries and rate limit handling."""

import logging
import time
from typing import Any, cast

import requests  # type: ignore[import-untyped]
from bs4 import BeautifulSoup

logger = logging.getLogger("training")


class ServerAPI:
    """API client for model submission server."""

    def __init__(self, token: str, username: str, server_url: str) -> None:
        self.token = token
        self.username = username
        self.server_url = server_url.rstrip("/")

    def submit_model(
        self,
        model_path: str,
        max_retries: int = 3,
        wait_on_rate_limit: bool = True,
        rate_limit_wait_minutes: int = 16,
        max_rate_limit_retries: int = 5,
    ) -> dict[str, Any] | None:
        """Submit model with automatic rate limit handling and retries."""
        url = f"{self.server_url}/submit"
        rate_limit_count = 0

        while rate_limit_count <= max_rate_limit_retries:
            for attempt in range(max_retries):
                try:
                    with open(model_path, "rb") as f:
                        files = {"file": f}
                        data = {"token": self.token}

                        logger.info(
                            f"Submitting model to {url} (attempt {attempt + 1}/{max_retries})..."
                        )
                        response = requests.post(url, data=data, files=files, timeout=60)

                        if response.status_code == 200:
                            result = response.json()
                            logger.info("Submission successful!")
                            return {
                                "success": True,
                                "message": result.get("message"),
                                "attempt": result.get("attempt"),
                                **result,
                            }

                        elif response.status_code == 429:
                            rate_limit_count += 1
                            if wait_on_rate_limit and rate_limit_count <= max_rate_limit_retries:
                                logger.warning(
                                    f"Rate limit hit ({rate_limit_count}/{max_rate_limit_retries}). "
                                    f"Waiting {rate_limit_wait_minutes} minutes..."
                                )
                                for remaining in range(rate_limit_wait_minutes, 0, -1):
                                    logger.info(f"   {remaining} minutes remaining...")
                                    time.sleep(60)
                                logger.info("Retrying submission...")
                                break
                            else:
                                logger.error("Rate limit exceeded max retries")
                                return {
                                    "success": False,
                                    "error": "Rate limit exceeded",
                                    "status_code": 429,
                                }

                        else:
                            logger.error(f"Submission failed with status {response.status_code}")
                            if attempt < max_retries - 1:
                                wait_time = 2**attempt
                                logger.info(f"Retrying in {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                return {
                                    "success": False,
                                    "error": response.text,
                                    "status_code": response.status_code,
                                }

                except requests.exceptions.Timeout:
                    logger.error("Request timed out")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)

                except Exception as e:
                    logger.error(f"Error submitting model: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        return {"success": False, "error": str(e)}
            else:
                break

        return None

    def check_status(self, max_retries: int = 3) -> list[dict[str, Any]] | None:
        """Check submission status for this token."""
        url = f"{self.server_url}/submission-status/{self.token}"

        for attempt in range(max_retries):
            try:
                logger.info(f"Checking status at {url}...")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    return cast(list[dict[str, Any]], response.json())
                elif response.status_code == 404:
                    logger.warning("No submissions found")
                    return []
                else:
                    logger.error(f"Status check failed: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)

            except requests.exceptions.Timeout:
                logger.error("Request timed out")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

            except Exception as e:
                logger.error(f"Error checking status: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return None

    def scrape_leaderboard(
        self, leaderboard_path: str = "/leaderboard", max_retries: int = 3
    ) -> list[dict[str, Any]] | None:
        """Scrape the public leaderboard."""
        url = f"{self.server_url}{leaderboard_path}"

        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching leaderboard from {url}...")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    table = soup.find("table")

                    if not table:
                        logger.warning("No leaderboard table found")
                        return []

                    entries = []
                    rows = table.find_all("tr")[1:]

                    for row in rows:
                        cols = row.find_all("td")
                        if len(cols) >= 4:
                            try:
                                entry = {
                                    "rank": int(cols[0].text.strip()),
                                    "team_name": cols[1].text.strip(),
                                    "f1_score": float(cols[2].text.strip()),
                                    "model_size_mb": float(cols[3].text.strip()),
                                }
                                entries.append(entry)
                            except (ValueError, IndexError):
                                continue

                    logger.info(f"Fetched {len(entries)} leaderboard entries")
                    return entries

                else:
                    logger.error(f"Leaderboard fetch failed: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)

            except requests.exceptions.Timeout:
                logger.error("Request timed out")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

            except Exception as e:
                logger.error(f"Error scraping leaderboard: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return None

    def get_our_rank(self, leaderboard_path: str = "/leaderboard") -> dict[str, Any] | None:
        """Get our team's entry from the leaderboard."""
        leaderboard = self.scrape_leaderboard(leaderboard_path)
        if not leaderboard:
            return None

        for entry in leaderboard:
            if entry["team_name"] == self.username:
                return entry

        return None

    def wait_for_evaluation(self, timeout: int = 1800, check_interval: int = 30) -> bool:
        """Wait for a pending submission to be evaluated."""
        start_time = time.time()
        logger.info(f"Waiting for evaluation (timeout: {timeout}s, check every {check_interval}s)")

        while (time.time() - start_time) < timeout:
            attempts = self.check_status()

            if not attempts:
                logger.warning("Could not check status, retrying...")
                time.sleep(check_interval)
                continue

            latest = attempts[-1]
            status = latest.get("status", "unknown")

            if status == "successful":
                logger.info("Evaluation complete!")
                return True
            elif status == "failed":
                logger.error("Evaluation failed")
                return False
            elif status == "pending":
                elapsed = int(time.time() - start_time)
                logger.info(f"Still pending... ({elapsed}s elapsed)")
                time.sleep(check_interval)
            else:
                logger.warning(f"Unknown status: {status}")
                time.sleep(check_interval)

        logger.error("Timeout waiting for evaluation")
        return False

    def get_metrics_from_leaderboard(
        self, leaderboard_path: str = "/leaderboard"
    ) -> dict[str, Any] | None:
        """Get our metrics from the leaderboard."""
        our_entry = self.get_our_rank(leaderboard_path)

        if not our_entry:
            return None

        return {
            "server_rank": our_entry["rank"],
            "server_f1_score": our_entry["f1_score"],
            "server_model_size_mb": our_entry["model_size_mb"],
        }
