"""
RF-3 Eye tracking validation utility (exporta JSON y resumen en texto).

Uso:
  python rf3_validator.py [ruta_csv] --interval-threshold 0.1 \
    --json-out rf3_resultados.json --summary-out rf3_resumen.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import List, Optional, Dict, Any

DEFAULT_CSV = Path(r"C:\proyecto_aplica\proyecto-aplicacion\Pruebas_vali\resultados\test_session.csv")
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
        previous_timestamp: Optional[int] = None
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
            return float("nan")
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
        f"Intervalo medio: {safe_ms(avg_interval):.2f} ms (aprox. {safe_float(approx_fps, none_as=float('nan')):.2f} fps)",
        f"Intervalo minimo: {safe_ms(metrics.min_interval()):.2f} ms",
        f"Intervalo maximo: {safe_ms(metrics.max_interval()):.2f} ms",
        f"Intervalos <= {metrics.interval_threshold * 1000:.0f} ms: {format_percent(metrics.continuity_rate())}",
    ]
    return "\n".join(summary_lines)


def build_json(metrics: RF3Metrics, csv_path: Path) -> Dict[str, Any]:
    avg_interval = metrics.avg_interval()
    approx_fps = (1.0 / avg_interval) if avg_interval and not math.isnan(avg_interval) and avg_interval > 0 else float("nan")
    return {
        "rf": "RF-3",
        "csv_path": str(csv_path),
        "interval_threshold_seconds": metrics.interval_threshold,
        "samples": {
            "total": metrics.total_samples,
            "tracking_status_1": metrics.valid_tracking,
            "status_1": metrics.valid_status,
            "left_eye_openness_gt0": metrics.left_eye_positive,
            "right_eye_openness_gt0": metrics.right_eye_positive,
        },
        "rates": {
            "detection_rate": safe_float(metrics.detection_rate()),
            "status_rate": safe_float(metrics.status_rate()),
            "left_eye_rate": safe_float(metrics.left_eye_rate()),
            "right_eye_rate": safe_float(metrics.right_eye_rate()),
            "continuity_rate_le_threshold": safe_float(metrics.continuity_rate()),
        },
        "timing": {
            "avg_interval_seconds": safe_float(avg_interval),
            "min_interval_seconds": safe_float(metrics.min_interval()),
            "max_interval_seconds": safe_float(metrics.max_interval()),
            "approx_fps": safe_float(approx_fps),
            "threshold_ms": metrics.interval_threshold * 1000.0,
        },
    }


def safe_float(x: float, none_as: Optional[float] = None) -> Optional[float]:
    if x is None:
        return none_as
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def safe_ms(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    return x * 1000.0


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
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Ruta del archivo JSON de salida (opcional)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Ruta del archivo de resumen en texto (opcional)",
    )

    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    metrics = RF3Metrics(rows, args.interval_threshold)

    # 1) Imprime el resumen como antes
    summary = build_summary(metrics, args.csv_path)
    print(summary)

    # 2) Exporta JSON si se solicita
    if args.json_out:
        payload = build_json(metrics, args.csv_path)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n→ Resultados guardados en JSON: {args.json_out}")

    # 3) Exporta el mismo resumen en texto si se solicita
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        # Asegura salto de línea final para herramientas tipo `cat` / `type`.
        with args.summary_out.open("w", encoding="utf-8", newline="\n") as f:
            f.write(summary + "\n")
        print(f"→ Resumen guardado en texto: {args.summary_out}")


if __name__ == "__main__":
    main()
