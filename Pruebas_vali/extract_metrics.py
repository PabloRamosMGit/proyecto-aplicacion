#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador de métricas de rendimiento (CPU/RSS/heap) desde CSV.

- Lee columnas: t,cpu,rss_mb,vms_mb,py_cur_mb,py_peak_mb
- Resume CPU (psutil-style, puede >100%), y lo convierte a "Task Manager" (global)
- Resume memoria RSS/VMS/heap y estima tendencia de RSS (MB/min)
- Permite excluir warm-up inicial (--warmup-seconds)
- Exporta resumen a JSON/CSV si se solicita
"""

from __future__ import annotations
import argparse, csv, json, math, statistics
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

try:
    import psutil
    CPU_LOGICAL_DEFAULT = psutil.cpu_count(logical=True) or 1
except Exception:
    import os
    CPU_LOGICAL_DEFAULT = os.cpu_count() or 1


@dataclass
class MetricsRow:
    t: float
    cpu: float
    rss_mb: float
    vms_mb: float
    py_cur_mb: float
    py_peak_mb: float


@dataclass
class CpuSummary:
    samples: int
    duration_s: float
    cpu_mean: float
    cpu_median: float
    cpu_min: float
    cpu_max: float
    cpu_p95: float
    cpu_tm_mean: float     # equivalente Task Manager (% global ≈ cpu / N)
    cpu_tm_p95: float


@dataclass
class MemSummary:
    rss_mean: float
    rss_min: float
    rss_max: float
    rss_std: float
    rss_span_mb: float
    rss_slope_mb_per_min: float
    vms_mean: float
    py_cur_mean: float
    py_peak_max: float


@dataclass
class Summary:
    n_cpus_logical: int
    window_used: Tuple[float, float]
    warmup_excluded_s: float
    cpu: CpuSummary
    mem: MemSummary


def parse_csv(path: str) -> List[MetricsRow]:
    rows: List[MetricsRow] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        expected = ["t","cpu","rss_mb","vms_mb","py_cur_mb","py_peak_mb"]
        if reader.fieldnames is None:
            raise ValueError("El CSV no tiene encabezado de columnas")
        for need in expected:
            if need not in reader.fieldnames:
                raise ValueError(f"Falta columna '{need}' en el CSV")
        for r in reader:
            rows.append(MetricsRow(
                t=float(r["t"]),
                cpu=float(r["cpu"]),
                rss_mb=float(r["rss_mb"]),
                vms_mb=float(r["vms_mb"]),
                py_cur_mb=float(r["py_cur_mb"]),
                py_peak_mb=float(r["py_peak_mb"]),
            ))
    rows.sort(key=lambda x: x.t)
    return rows


def percentile(values: List[float], p: float) -> float:
    """Percentil simple (p en [0,100]) con interpolación lineal."""
    if not values:
        return float("nan")
    x = sorted(values)
    k = (len(x) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return x[int(k)]
    return x[f] + (x[c] - x[f]) * (k - f)


def ols_slope(x: List[float], y: List[float]) -> float:
    """Pendiente OLS (y ~ a + b*x). Devuelve b."""
    n = len(x)
    if n < 2:
        return float("nan")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    if den == 0.0:
        return float("nan")
    return num / den


def summarize(rows: List[MetricsRow],
              warmup_seconds: float,
              n_cpus_logical: int) -> Summary:
    if not rows:
        raise ValueError("No hay datos")

    # Excluir warm-up inicial si aplica
    t0 = rows[0].t
    used = [r for r in rows if (r.t - t0) >= warmup_seconds] if warmup_seconds > 0 else rows
    if not used:
        used = rows  # fallback si warmup deja vacío

    t_start = used[0].t
    t_end = used[-1].t
    duration = t_end - t_start
    cpu_vals = [r.cpu for r in used]
    rss_vals = [r.rss_mb for r in used]
    vms_vals = [r.vms_mb for r in used]
    py_cur_vals = [r.py_cur_mb for r in used]
    py_peak_vals = [r.py_peak_mb for r in used]

    # CPU resumen
    cpu_mean = statistics.fmean(cpu_vals)
    cpu_med = statistics.median(cpu_vals)
    cpu_min = min(cpu_vals); cpu_max = max(cpu_vals)
    cpu_p95 = percentile(cpu_vals, 95)

    # Equivalente Task Manager (global del equipo) ≈ cpu% / N
    tm_mean = cpu_mean / max(1, n_cpus_logical)
    tm_p95 = cpu_p95 / max(1, n_cpus_logical)

    cpu_sum = CpuSummary(
        samples=len(used),
        duration_s=duration,
        cpu_mean=cpu_mean,
        cpu_median=cpu_med,
        cpu_min=cpu_min,
        cpu_max=cpu_max,
        cpu_p95=cpu_p95,
        cpu_tm_mean=tm_mean,
        cpu_tm_p95=tm_p95
    )

    # Memoria resumen
    rss_mean = statistics.fmean(rss_vals)
    rss_min = min(rss_vals); rss_max = max(rss_vals)
    rss_std = statistics.pstdev(rss_vals) if len(rss_vals) > 1 else 0.0
    rss_span = rss_max - rss_min

    # Tendencia de RSS (MB/min)
    xs = [r.t for r in used]            # segundos
    slope_mb_per_s = ols_slope(xs, rss_vals) if len(xs) >= 2 else float("nan")
    slope_mb_per_min = slope_mb_per_s * 60.0 if not math.isnan(slope_mb_per_s) else float("nan")

    mem_sum = MemSummary(
        rss_mean=rss_mean,
        rss_min=rss_min,
        rss_max=rss_max,
        rss_std=rss_std,
        rss_span_mb=rss_span,
        rss_slope_mb_per_min=slope_mb_per_min,
        vms_mean=statistics.fmean(vms_vals),
        py_cur_mean=statistics.fmean(py_cur_vals),
        py_peak_max=max(py_peak_vals),
    )

    return Summary(
        n_cpus_logical=n_cpus_logical,
        window_used=(t_start, t_end),
        warmup_excluded_s=warmup_seconds,
        cpu=cpu_sum,
        mem=mem_sum
    )


def render_human(summary: Summary) -> str:
    s = summary
    lines = []
    lines.append("="*72)
    lines.append("RESUMEN DE MÉTRICAS")
    lines.append("="*72)
    lines.append(
        f"Muestras usadas: {s.cpu.samples} | Ventana: {s.window_used[0]:.2f}s → {s.window_used[1]:.2f}s "
        f"(duración {s.cpu.duration_s:.2f}s) | Warm-up excluido: {s.warmup_excluded_s:.2f}s"
    )
    lines.append(f"Núcleos lógicos (Task Manager base): {s.n_cpus_logical}")
    lines.append("")
    lines.append("CPU (psutil-style, puede >100% por uso multi-hilo):")
    lines.append(
        f"  Media:  {s.cpu.cpu_mean:6.1f}%   Mediana: {s.cpu.cpu_median:6.1f}%   "
        f"P95: {s.cpu.cpu_p95:6.1f}%   Min–Max: {s.cpu.cpu_min:6.1f}%–{s.cpu.cpu_max:6.1f}%"
    )
    lines.append("CPU equivalente Task Manager (porcentaje global del equipo):")
    lines.append(
        f"  Media TM: {s.cpu.cpu_tm_mean:5.2f}%   P95 TM: {s.cpu.cpu_tm_p95:5.2f}%  "
        f"(TM% ≈ CPU% / {s.n_cpus_logical})"
    )
    lines.append("")
    lines.append("Memoria RSS (MB):")
    lines.append(
        f"  Media: {s.mem.rss_mean:7.2f}   Min–Max: {s.mem.rss_min:7.2f}–{s.mem.rss_max:7.2f}   "
        f"σ: {s.mem.rss_std:5.2f}   Δ: {s.mem.rss_span_mb:5.2f}"
    )
    slope = s.mem.rss_slope_mb_per_min
    slope_txt = "nan" if math.isnan(slope) else f"{slope:0.3f} MB/min"
    lines.append(f"  Tendencia (OLS): {slope_txt}")
    lines.append("")
    lines.append("Memoria VMS (MB) y Heap Python (MB):")
    lines.append(
        f"  VMS media: {s.mem.vms_mean:7.2f}   Py heap medio: {s.mem.py_cur_mean:5.2f}   "
        f"Py peak máx: {s.mem.py_peak_max:5.2f}"
    )
    lines.append("="*72)
    return "\n".join(lines)


def print_human(summary: Summary) -> None:
    print(render_human(summary))


def write_human_txt(path: str, summary: Summary) -> None:
    text = render_human(summary)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, summary: Summary) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)


def write_csv(path: str, summary: Summary) -> None:
    # CSV "flat" con las métricas más importantes
    fields = [
        "n_cpus_logical",
        "t_start", "t_end", "duration_s", "warmup_excluded_s",
        "cpu_mean", "cpu_median", "cpu_min", "cpu_max", "cpu_p95",
        "cpu_tm_mean", "cpu_tm_p95",
        "rss_mean", "rss_min", "rss_max", "rss_std", "rss_span_mb", "rss_slope_mb_per_min",
        "vms_mean", "py_cur_mean", "py_peak_max"
    ]
    row = {
        "n_cpus_logical": summary.n_cpus_logical,
        "t_start": summary.window_used[0],
        "t_end": summary.window_used[1],
        "duration_s": summary.cpu.duration_s,
        "warmup_excluded_s": summary.warmup_excluded_s,
        "cpu_mean": summary.cpu.cpu_mean,
        "cpu_median": summary.cpu.cpu_median,
        "cpu_min": summary.cpu.cpu_min,
        "cpu_max": summary.cpu.cpu_max,
        "cpu_p95": summary.cpu.cpu_p95,
        "cpu_tm_mean": summary.cpu.cpu_tm_mean,
        "cpu_tm_p95": summary.cpu.cpu_tm_p95,
        "rss_mean": summary.mem.rss_mean,
        "rss_min": summary.mem.rss_min,
        "rss_max": summary.mem.rss_max,
        "rss_std": summary.mem.rss_std,
        "rss_span_mb": summary.mem.rss_span_mb,
        "rss_slope_mb_per_min": summary.mem.rss_slope_mb_per_min,
        "vms_mean": summary.mem.vms_mean,
        "py_cur_mean": summary.mem.py_cur_mean,
        "py_peak_max": summary.mem.py_peak_max,
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Resumen de métricas de rendimiento desde CSV.")
    ap.add_argument("--warmup-seconds", type=float, default=5.0,
                    help="Segundos iniciales a excluir del análisis (default: 5.0)")
    ap.add_argument("--cpus", type=int, default=CPU_LOGICAL_DEFAULT,
                    help=f"Núcleos lógicos para conversión a Task Manager (default: detectado={CPU_LOGICAL_DEFAULT})")
    ap.add_argument("--json-out", type=str, default=None, help="Ruta para exportar resumen en JSON")
    ap.add_argument("--csv-out", type=str, default=None, help="Ruta para exportar resumen en CSV (una fila)")
    ap.add_argument("--txt-out", type=str, default=None, help="Ruta para exportar el resumen humano en TXT")
    args = ap.parse_args()

    rows = parse_csv(r"C:\proyecto_aplica\proyecto-aplicacion\Pruebas_vali\resultados\perf_metrics.csv")
    summary = summarize(rows, warmup_seconds=args.warmup_seconds, n_cpus_logical=args.cpus)

    # Siempre imprime en consola
    print_human(summary)

    # Exportaciones opcionales
    if args.json_out:
        write_json(args.json_out, summary)
        print(f"\nJSON guardado en: {args.json_out}")
    if args.csv_out:
        write_csv(args.csv_out, summary)
        print(f"CSV (resumen) guardado en: {args.csv_out}")
    if args.txt_out:
        write_human_txt(args.txt_out, summary)
        print(f"TXT (resumen humano) guardado en: {args.txt_out}")


if __name__ == "__main__":
    main()
