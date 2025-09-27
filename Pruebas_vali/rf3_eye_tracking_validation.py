"""RF-3 Eye tracking validation utility.

This script evaluates the requirement RF-3: eye detection and tracking reliability.
It reads gaze samples produced by GazeFollower and reports:
  * Detection success rate based on the CSV columns `tracking_status` and `status`.
  * Eye openness continuity checks for both eyes.
  * Temporal continuity (frame cadence) derived from the timestamp column.

The report mirrors the manual analysis performed over the demo session located at
`GazeFollower-1.0.1/data/free_viewing_pygame_demo.csv`.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import List

DEFAULT_CSV = Path("C:\\proyecto_aplica\\proyecto-aplicacion\\Pruebas_vali\\resultados\\test_session.csv")
DEFAULT_INTERVAL_THRESHOLD = 0.1  # seconds (100 ms)


class RF3Metrics:
    def __init__(self, rows: List[dict], interval_threshold: float) -> None:
        self.rows = rows
        self.interval_threshold = interval_threshold
        self.total_samples = len(rows)
        self.valid_tracking = 0
        self.valid_status = 0
        self.left_eye_positive = 0
        self.right_eye_positive = 0
        self.intervals: List[float] = []

        self._analyse()

    def _analyse(self) -> None:
        previous_timestamp = None
        for row in self.rows:
            if row.get("tracking_status") == "1":
                self.valid_tracking += 1
            if row.get("status") == "1":
                self.valid_status += 1

            left_open = float(row.get("left_eye_openness", 0.0))
            right_open = float(row.get("right_eye_openness", 0.0))
            if left_open > 0:
                self.left_eye_positive += 1
            if right_open > 0:
                self.right_eye_positive += 1

            try:
                timestamp = int(row["timestamp"])
            except (KeyError, ValueError):
                timestamp = None

            if timestamp is not None and previous_timestamp is not None:
                delta = (timestamp - previous_timestamp) / 1e9
                if delta >= 0:
                    self.intervals.append(delta)
            previous_timestamp = timestamp if timestamp is not None else previous_timestamp

    def detection_rate(self) -> float:
        return self._ratio(self.valid_tracking, self.total_samples)

    def status_rate(self) -> float:
        return self._ratio(self.valid_status, self.total_samples)

    def left_eye_rate(self) -> float:
        return self._ratio(self.left_eye_positive, self.total_samples)

    def right_eye_rate(self) -> float:
        return self._ratio(self.right_eye_positive, self.total_samples)

    def continuity_rate(self) -> float:
        if not self.intervals:
            return 0.0
        within = sum(1 for dt in self.intervals if dt <= self.interval_threshold)
        return self._ratio(within, len(self.intervals))

    def avg_interval(self) -> float:
        return statistics.mean(self.intervals) if self.intervals else float("nan")

    def min_interval(self) -> float:
        return min(self.intervals) if self.intervals else float("nan")

    def max_interval(self) -> float:
        return max(self.intervals) if self.intervals else float("nan")

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return float("nan")
        return numerator / denominator


def format_percent(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def load_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows")
    return rows


def build_summary(metrics: RF3Metrics, csv_path: Path) -> str:
    avg_interval = metrics.avg_interval()
    approx_fps = (1.0 / avg_interval) if avg_interval and not math.isnan(avg_interval) and avg_interval > 0 else float("nan")
    summary_lines = [
        "RF-3 Check",
        f"Archivo analizado: {csv_path}",
        f"Muestras totales: {metrics.total_samples}",
        f"tracking_status == 1: {metrics.valid_tracking} ({format_percent(metrics.detection_rate())})",
        f"status == 1: {metrics.valid_status} ({format_percent(metrics.status_rate())})",
        f"Apertura ojo izquierdo > 0: {metrics.left_eye_positive} ({format_percent(metrics.left_eye_rate())})",
        f"Apertura ojo derecho > 0: {metrics.right_eye_positive} ({format_percent(metrics.right_eye_rate())})",
        f"Intervalo medio: {avg_interval * 1000:.2f} ms (aprox. {approx_fps:.2f} fps)",
        f"Intervalo minimo: {metrics.min_interval() * 1000:.2f} ms",
        f"Intervalo maximo: {metrics.max_interval() * 1000:.2f} ms",
        f"Intervalos <= {metrics.interval_threshold * 1000:.0f} ms: {format_percent(metrics.continuity_rate())}",
    ]
    return "\n".join(summary_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-3 eye detection and tracking validation")
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help="Ruta al CSV con muestras de GazeFollower",
    )
    parser.add_argument(
        "--interval-threshold",
        type=float,
        default=DEFAULT_INTERVAL_THRESHOLD,
        dest="interval_threshold",
        help="Umbral de continuidad en segundos (default: 0.1)",
    )

    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    metrics = RF3Metrics(rows, args.interval_threshold)
    print(build_summary(metrics, args.csv_path))


if __name__ == "__main__":
    main()
