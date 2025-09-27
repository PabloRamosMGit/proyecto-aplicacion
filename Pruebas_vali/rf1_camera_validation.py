"""
RF-1: Validación de captura de entradas visuales

Verifica que la cámara capture en tiempo real a 30 FPS
y con resolución mínima 720p (1280x720).

Basado en el estilo del ejemplo pygame de GazeFollower,
usando la clase WebCamCamera del paquete gazefollower.
"""

import time
import os
import sys
import json
import argparse
from datetime import datetime
from collections import deque
import threading
from typing import List, Any, Optional, cast

import pygame
import cv2
import numpy as np

# Asegurar que el paquete local 'gazefollower' sea importable
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
GAZEFOLLOWER_ROOT = os.path.join(PROJECT_ROOT, "GazeFollower-1.0.1")
if GAZEFOLLOWER_ROOT not in sys.path:
    sys.path.insert(0, GAZEFOLLOWER_ROOT)

from gazefollower.camera.WebCamCamera import WebCamCamera
from gazefollower.logger.Logger import Log


TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30


class FpsMeter:
    def __init__(self, window_sec: float = 2.0):
        self.times = deque()
        self.window = window_sec

    def tick(self, t: float) -> float:
        self.times.append(t)
        # Mantener eventos dentro de la ventana de tiempo
        cutoff = t - self.window
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
        # FPS medidos
        if len(self.times) >= 2:
            elapsed = self.times[-1] - self.times[0]
            return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0
        return 0.0


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, data: dict):
    # Escribimos encabezado si el archivo no existe
    header = (
        "timestamp_iso,target_width,target_height,target_fps,"
        "reported_width,reported_height,reported_fps,"
        "frame_width,frame_height,measured_fps,duration_seconds,result\n"
    )
    line = (
        f"{data['timestamp_iso']},{data['target_width']},{data['target_height']},{data['target_fps']},"
        f"{data['reported_width']},{data['reported_height']},{data['reported_fps']},"
        f"{data['frame_width']},{data['frame_height']},{data['measured_fps']:.2f},{data['duration_seconds']:.3f},"
        f"{data['result']}\n"
    )
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(line)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="RF-1 Validación de cámara (720p @ 30fps)")
    parser.add_argument("--outdir", default=os.path.join(CURRENT_DIR, "resultados"), help="Directorio de salida")
    parser.add_argument(
        "--format", choices=["csv", "json", "both"], default="csv", help="Formato de salida"
    )
    parser.add_argument("--name", default="rf1_camera_validation", help="Nombre base del archivo de salida")
    parser.add_argument("--cam", type=int, default=0, help="Índice de cámara (default: 0)")
    parser.add_argument("--backend", choices=["DSHOW", "MSMF", "V4L2", "ANY"], default="ANY",
                        help="Backend de captura preferido")
    parser.add_argument("--codec", choices=["MJPG", "YUY2", "YUYV", "ANY"], default="ANY",
                        help="FOURCC preferido para la cámara")
    return parser.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.outdir)
    # Inicializa logger requerido por gazefollower
    log_dir = os.path.join(args.outdir, "logs")
    _ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"log_rf1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt")
    Log.init(log_path)
    # Inicializa pygame con una ventana del tamaño objetivo
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    #pygame.display.set_caption("RF-1 Validación Cámara (720p @ 30fps)")
    #font = pygame.font.SysFont(None, 28)

    # Inicializa la cámara del paquete gazefollower
    cam = WebCamCamera(webcam_id=0, img_height=TARGET_HEIGHT, img_width=TARGET_WIDTH, cam_fps=TARGET_FPS)

    latest_frame = None  # type: Optional[np.ndarray]
    frame_lock = threading.Lock()
    fps_meter = FpsMeter(window_sec=2.0)
    measured_fps = 0.0

    def on_frame(state, timestamp_ns, frame):
        nonlocal latest_frame, measured_fps
        # Marca temporal en segundos
        t = time.time()
        measured_fps = fps_meter.tick(t)
        # Guardar último frame disponible (ya viene en RGB y redimensionado por WebCamCamera)
        # Usamos un lock ligero de pygame para minimizar riesgos entre hilos
        with frame_lock:
            latest_frame = frame

    # Suscribir callback y arrancar en modo preview (sin carga pesada de inferencia)
    cam.set_on_image_callback(on_frame)
    cam.start_previewing()

    # Tras abrir la cámara, consultar valores reportados por el driver
    # Nota: el backend de OpenCV puede no reportar FPS reales en todas las cámaras
    try:
        reported_w = int(cam._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_h = int(cam._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = float(cam._cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        reported_w, reported_h, reported_fps = 0, 0, 0.0

    start_time = time.time()
    validation_done = False
    validation_pass = False
    VALIDATION_MIN_SECONDS = 5.0  # medir unos segundos para estabilizar

    clock = pygame.time.Clock()

    running = True
    frame = None  # type: Optional[np.ndarray]
    # Inicializar tamaños para evitar variables posiblemente no asignadas al final
    fw, fh = 0, 0
    while running:
        # Pylance: ayudar al tipado del resultado de pygame.event.get()
        events = cast(List[Any], pygame.event.get())
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                    running = False

        # Dibujar el último frame si disponible
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            # OpenCV entrega ndarray (H, W, 3) RGB; pygame espera (W, H) y origen superior-izquierdo
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            win.blit(surf, (0, 0))

        # Reglas de validación
        # - Resolución: usar los valores reportados por el driver si existen; fallback a dimensiones del frame
        if frame is not None:
            fh, fw = frame.shape[:2]
        else:
            fh, fw = 0, 0

        effective_w = reported_w or fw
        effective_h = reported_h or fh

        enough_time = (time.time() - start_time) >= VALIDATION_MIN_SECONDS
        res_ok = (effective_w >= TARGET_WIDTH and effective_h >= TARGET_HEIGHT)
        fps_ok = (measured_fps >= TARGET_FPS - 1)  # margen pequeño por jitter

        if not validation_done and enough_time:
            validation_pass = res_ok and fps_ok
            validation_done = True

        # Overlay de textos informativos
        lines = [
            f"Reportado cam: {reported_w}x{reported_h} @ {reported_fps:.1f} fps",
            f"Frame actual: {fw}x{fh}",
            f"FPS medido: {measured_fps:.1f}",
            f"Resolución OK (>=1280x720): {'SI' if res_ok else 'NO'}",
            f"FPS OK (>=30): {'SI' if fps_ok else 'NO'}",
        ]
        if validation_done:
            lines.append(f"Resultado RF-1: {'PASS' if validation_pass else 'FAIL'}")
            lines.append("Presiona ENTER o ESC para salir")

        y = 8
        for txt in lines:
            color = (0, 255, 0) if ("OK" in txt and "NO" not in txt) or ("PASS" in txt) else (255, 255, 255)
            if "FAIL" in txt:
                color = (255, 80, 80)

        pygame.display.flip()
        clock.tick(60)

    duration = time.time() - start_time

    # Cerrar recursos
    try:
        cam.stop_previewing()
    except Exception:
        pass
    cam.release()
    pygame.quit()

    # Preparar datos de resultado
    result_payload = {
        "timestamp_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "target_fps": TARGET_FPS,
        "reported_width": reported_w,
        "reported_height": reported_h,
        "reported_fps": round(reported_fps, 2),
        "frame_width": fw,
        "frame_height": fh,
        "measured_fps": round(measured_fps, 2),
        "duration_seconds": round(duration, 3),
        "result": "PASS" if validation_pass else "FAIL",
    }

    # Guardar según formato
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{args.name}_{ts}"
    if args.format in ("csv", "both"):
        save_csv(os.path.join(args.outdir, base + ".csv"), result_payload)
    if args.format in ("json", "both"):
        save_json(os.path.join(args.outdir, base + ".json"), result_payload)

    # Log en consola para CI/registro
    print("--- RF-1 Validación ---")
    print(f"Reportado: {reported_w}x{reported_h} @ {reported_fps:.1f} fps")
    print(f"FPS medido: {measured_fps:.2f}")
    print(f"Resultado: {'PASS' if validation_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
