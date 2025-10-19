# merge_txt_to_excel.py
# -*- coding: utf-8 -*-
"""
Escanea recursivamente desde el directorio actual todos los archivos .txt
de resultados de calibración (formato con línea 'MeanPixelDistance,valor')
y los combina en un solo Excel: 'calibration_results_merged.xlsx'.

Por cada .txt crea dos hojas:
  - <stem>_per_point  (tabla de puntos)
  - <stem>_summary    (MeanPixelDistance)
"""

import re
from io import StringIO
from pathlib import Path
from datetime import datetime

import pandas as pd

# -------- Utilidades --------
def parse_calibration_txt(raw: str):
    """Devuelve (df_per_point, mean_value). Lanza ValueError si no se puede parsear."""
    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("El archivo está vacío.")

    # Ubicar la línea de resumen
    summary_idx = next((i for i, ln in enumerate(lines)
                        if ln.strip().startswith("MeanPixelDistance")), None)

    csv_lines = lines[:summary_idx] if summary_idx is not None else lines
    csv_text = "\n".join(csv_lines)

    # Parseo de la tabla
    df = pd.read_csv(StringIO(csv_text))

    # Extraer MeanPixelDistance (si existe)
    mean_value = None
    if summary_idx is not None:
        m = re.match(r"^MeanPixelDistance\s*,\s*([0-9.]+)\s*$", lines[summary_idx])
        if m:
            try:
                mean_value = float(m.group(1))
            except ValueError:
                mean_value = None

    # Asegurar tipos numéricos cuando corresponde (convierte a numérico o deja NaN si no es convertible)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, mean_value


def safe_sheet_name(name: str) -> str:
    r"""
    Normaliza el nombre de hoja para Excel:
    - Reemplaza caracteres inválidos: : \ / ? * [ ]
    - Limita a 31 caracteres
    """
    name = re.sub(r'[:\\/?*\[\]]', '_', name)
    return name[:31] if len(name) > 31 else name


def unique_sheet_name(base: str, existing: set) -> str:
    """Garantiza que el nombre de hoja sea único en el libro."""
    name = base
    i = 1
    while name in existing:
        suffix = f"_{i}"
        truncated = base[: max(0, 31 - len(suffix))]
        name = f"{truncated}{suffix}"
        i += 1
    existing.add(name)
    return name


# -------- Configuración de búsqueda --------
ROOT = Path.cwd()
PATTERN = "calibration_differences_*.txt"   # ajusta si usas otro patrón
OUTPUT_XLSX = Path("calibration_results_merged.xlsx")

# Busca recursivamente todos los .txt que coincidan
txt_files = sorted(ROOT.rglob(PATTERN))

if not txt_files:
    print(f"⚠️ No se encontraron archivos con el patrón '{PATTERN}' bajo: {ROOT}")
    exit(0)

print(f"🔎 Archivos encontrados ({len(txt_files)}):")
for p in txt_files:
    print(f"  - {p.relative_to(ROOT)}")

# -------- Escritura del Excel --------
# Usamos xlsxwriter si está disponible; si no, pandas usará el motor por defecto (openpyxl).
try:
    writer = pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter")
    engine_ok = "xlsxwriter"
except ModuleNotFoundError:
    writer = pd.ExcelWriter(OUTPUT_XLSX)  # motor por defecto
    engine_ok = "default"

print(f"\n📝 Creando Excel: {OUTPUT_XLSX} (engine={engine_ok})")

sheet_names_in_use = set()

with writer:
    for path in txt_files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            df, mean_val = parse_calibration_txt(raw)
        except Exception as e:
            print(f"❌ Error al parsear '{path}': {e}")
            continue

        stem = path.stem  # nombre sin extensión
        per_point_sheet = unique_sheet_name(safe_sheet_name(f"{stem}_per_point"), sheet_names_in_use)
        summary_sheet   = unique_sheet_name(safe_sheet_name(f"{stem}_summary"),   sheet_names_in_use)

        # Hoja per_point
        df.to_excel(writer, index=False, sheet_name=per_point_sheet)

        # Ajuste de ancho (si el motor lo permite)
        try:
            ws = writer.sheets[per_point_sheet]
            for i, col in enumerate(df.columns):
                max_len = max(len(col), df[col].astype(str).map(len).max() + 2)
                ws.set_column(i, i, min(max_len, 28))
        except Exception:
            pass  # con openpyxl puede no aplicar

        # Hoja summary
        summary_df = pd.DataFrame({"Metric": ["MeanPixelDistance"], "Value": [mean_val]})
        summary_df.to_excel(writer, index=False, sheet_name=summary_sheet)

        # Ajuste de ancho simple
        try:
            ws2 = writer.sheets[summary_sheet]
            ws2.set_column(0, 0, 22)
            ws2.set_column(1, 1, 18)
        except Exception:
            pass

print(f"\n✅ Listo. Excel combinado: {OUTPUT_XLSX.resolve()}")
print(f"🗂️ Hojas creadas: {', '.join(sorted(sheet_names_in_use))}")
