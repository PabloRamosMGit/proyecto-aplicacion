"""
RR-6 Data logging interval validator
-----------------------------------
(Con ruta por defecto al CSV si no se especifican archivos)

Uso básico (usa la ruta por defecto):
  python rr6_validator.py

Con archivos explícitos:
  python rr6_validator.py path\to\file.csv otro.csv

Con exportación:
  python rr6_validator.py ^
    --json-out rr6_resultados.json ^
    --summary-out rr6_resumen.txt

Opciones:
  --max-interval      Umbral del requisito (segundos). Default: 5.0
  --recommended       Umbral recomendado (segundos). Default: 0.25
  --units             auto|ns|ms|s  (auto por defecto)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MAX_INTERVAL = 5.0       # segundos (requisito actual)
DEFAULT_RECOMMENDED = 0.25       # segundos (nuevo umbral propuesto)
DEFAULT_UNITS = "auto"           # auto|ns|ms|s
DEFAULT_CSVS = [Path(r"C:\proyecto_aplica\proyecto-aplicacion\Pruebas_vali\resultados\test_session.csv")]

Units = str  # alias


def load_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    return rows


def parse_timestamps(rows: List[dict]) -> List[float]:
    ts: List[float] = []
    for r in rows:
        v = r.get("timestamp")
        if v is None:
            continue
        try:
            ts.append(float(v))
        except ValueError:
            continue
    return ts


def infer_units_from_deltas(raw_deltas: List[float]) -> Units:
    if not raw_deltas:
        return "s"
    med = statistics.median(raw_deltas)
    if med > 1e6:
        return "ns"
    if med > 1e3:
        return "ms"
    return "s"


def scale_deltas_to_seconds(raw_deltas: List[float], units: Units) -> List[float]:
    if units == "ns":
        return [d / 1e9 for d in raw_deltas]
    if units == "ms":
        return [d / 1e3 for d in raw_deltas]
    return list(raw_deltas)


def compute_deltas_seconds(timestamps: List[float], units_mode: Units) -> Tuple[List[float], Units]:
    if len(timestamps) < 2:
        return ([], "s")
    raw_deltas: List[float] = []
    prev = timestamps[0]
    for t in timestamps[1:]:
        d = t - prev
        if d >= 0:
            raw_deltas.append(d)
            prev = t
    inferred_units = infer_units_from_deltas(raw_deltas) if units_mode == "auto" else units_mode
    deltas_s = scale_deltas_to_seconds(raw_deltas, inferred_units)
    return deltas_s, inferred_units


def pct_within_threshold(values: List[float], threshold: float) -> float:
    if not values:
        return float("nan")
    within = sum(1 for v in values if v <= threshold)
    return within / len(values)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    arr = sorted(values)
    k = (len(arr) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return d0 + d1


def safe_float(x: float) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def analyze_csv(csv_path: Path, max_interval: float, recommended: float, units: Units) -> Dict[str, Any]:
    rows = load_rows(csv_path)
    timestamps = parse_timestamps(rows)
    deltas_s, used_units = compute_deltas_seconds(timestamps, units)

    stats: Dict[str, Any] = {
        "file": str(csv_path),
        "samples_total": len(rows),
        "timestamps_found": len(timestamps),
        "deltas_count": len(deltas_s),
        "units_mode": used_units,
        "interval": {
            "avg_s": safe_float(statistics.mean(deltas_s)) if deltas_s else None,
            "min_s": safe_float(min(deltas_s)) if deltas_s else None,
            "max_s": safe_float(max(deltas_s)) if deltas_s else None,
            "p95_s": safe_float(percentile(deltas_s, 95.0)) if deltas_s else None,
            "p99_s": safe_float(percentile(deltas_s, 99.0)) if deltas_s else None,
        },
        "thresholds": {
            "requirement_s": max_interval,
            "recommended_s": recommended,
            "pct_within_requirement": safe_float(pct_within_threshold(deltas_s, max_interval)),
            "pct_within_recommended": safe_float(pct_within_threshold(deltas_s, recommended)),
            "pass_requirement": bool(deltas_s and max(deltas_s) <= max_interval),
            "pass_recommended": bool(deltas_s and max(deltas_s) <= recommended),
        },
    }
    return stats


def build_summary(stats: Dict[str, Any]) -> str:
    f = stats["file"]
    n = stats["samples_total"]
    d = stats["deltas_count"]
    units = stats["units_mode"]
    interval = stats["interval"]
    thr = stats["thresholds"]

    def fmt_pct(x: Optional[float]) -> str:
        return "n/a" if (x is None) else f"{x*100:.2f}%"

    def fmt_s(x: Optional[float]) -> str:
        return "n/a" if (x is None) else f"{x*1000:.2f} ms"

    lines = [
        "RR-6 Check",
        f"Archivo analizado: {f}",
        f"Muestras totales (filas CSV): {n}",
        f"Intervalos calculados: {d} (unidades inferidas: {units})",
        f"Intervalo medio: {fmt_s(interval['avg_s'])}",
        f"Intervalo mínimo: {fmt_s(interval['min_s'])}",
        f"Intervalo máximo: {fmt_s(interval['max_s'])}",
        f"P95 intervalo: {fmt_s(interval['p95_s'])}",
        f"P99 intervalo: {fmt_s(interval['p99_s'])}",
        f"<= {thr['requirement_s']:.2f} s: {fmt_pct(thr['pct_within_requirement'])}  "
        f"(PASS={thr['pass_requirement']})",
        f"<= {thr['recommended_s']:.3f} s: {fmt_pct(thr['pct_within_recommended'])}  "
        f"(PASS={thr['pass_recommended']})",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="RR-6 logging interval validator")
    p.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        default=DEFAULT_CSVS,  # ← por defecto usa test_session.csv
        help="Ruta(s) a CSV con muestras (columna 'timestamp'). Si no se indica, usa la ruta por defecto.",
    )
    p.add_argument(
        "--max-interval",
        type=float,
        default=DEFAULT_MAX_INTERVAL,
        help=f"Umbral del requisito (segundos). Default: {DEFAULT_MAX_INTERVAL}",
    )
    p.add_argument(
        "--recommended",
        type=float,
        default=DEFAULT_RECOMMENDED,
        help=f"Umbral recomendado (segundos). Default: {DEFAULT_RECOMMENDED}",
    )
    p.add_argument(
        "--units",
        type=str,
        choices=["auto", "ns", "ms", "s"],
        default=DEFAULT_UNITS,
        help="Unidad de 'timestamp' (auto|ns|ms|s). Default: auto",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Ruta del archivo JSON de salida (opcional)",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Ruta del archivo de resumen en texto (opcional)",
    )

    args = p.parse_args()

    all_stats: List[Dict[str, Any]] = []
    summaries: List[str] = []

    for csv_path in args.csv_paths:
        try:
            stats = analyze_csv(csv_path, args.max_interval, args.recommended, args.units)
        except Exception as e:
            err_payload = {"file": str(csv_path), "error": str(e)}
            all_stats.append(err_payload)
            summaries.append(f"RR-6 Check\nArchivo analizado: {csv_path}\nERROR: {e}")
            print(summaries[-1]); print()
            continue

        all_stats.append(stats)
        summary = build_summary(stats)
        summaries.append(summary)
        print(summary); print()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"→ Resultados guardados en JSON: {args.json_out}")

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n\n".join(summaries) + "\n")
        print(f"→ Resumen guardado en texto: {args.summary_out}")


if __name__ == "__main__":
    main()
