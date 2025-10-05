import os, sys, time, csv, threading
import pygame
from pygame.locals import KEYDOWN, K_RETURN
import psutil
import tracemalloc

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
GAZEFOLLOWER_ROOT = os.path.join(PROJECT_ROOT, "GazeFollower-1.0.1")
if GAZEFOLLOWER_ROOT not in sys.path:
    sys.path.insert(0, GAZEFOLLOWER_ROOT)

from gazefollower import GazeFollower
from gazefollower.gaze_estimator import MGazeNetGazeEstimator

# ---------- Monitor de rendimiento en hilo aparte ----------
class PerfMonitor:
    def __init__(self, interval_sec: float = 0.5):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.latest = {"t": 0.0, "cpu": 0.0, "rss_mb": 0.0, "vms_mb": 0.0,
                       "py_cur_mb": 0.0, "py_peak_mb": 0.0}
        self.rows = []
        # Primera llamada "priming" para cpu_percent
        self.proc.cpu_percent(None)

    def start(self):
        tracemalloc.start()  # mide memoria solo de Python
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            cpu = self.proc.cpu_percent(None)  # % total (puede >100% si usa varios núcleos)
            mem = self.proc.memory_info()
            cur, peak = tracemalloc.get_traced_memory()
            now = time.perf_counter() - t0
            data = {
                "t": now,
                "cpu": cpu,
                "rss_mb": mem.rss / (1024**2),  # RAM real del proceso
                "vms_mb": mem.vms / (1024**2),  # memoria virtual mapeada
                "py_cur_mb": cur / (1024**2),   # heap Python actual
                "py_peak_mb": peak / (1024**2), # pico de heap Python
            }
            with self._lock:
                self.latest = data
                self.rows.append(data)
            time.sleep(self.interval)

    def snapshot(self):
        with self._lock:
            return dict(self.latest)

    def stop_and_dump_csv(self, out_path: str):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        # guardar CSV
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["t","cpu","rss_mb","vms_mb","py_cur_mb","py_peak_mb"])
            writer.writeheader()
            writer.writerows(self.rows)

# ---------- Tu main con overlay ----------
if __name__ == '__main__':
    # init pygame
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    # init GazeFollower
    gf = GazeFollower()
    gf.preview(win=win)
    gf.calibrate(win=win)
    gf.start_sampling()
    pygame.time.wait(100)

    # Inicia monitor de rendimiento
    mon = PerfMonitor(interval_sec=0.5)
    mon.start()

    img_folder = 'images'
    images = ['C:\\proyecto_aplica\\proyecto-aplicacion\\GazeFollower-1.0.1\\example\\images\\grid.jpg']

    for _img in images:
        win.fill((128, 128, 128))
        im = pygame.image.load(os.path.join(img_folder, _img))
        win.blit(im, (0, 0))
        pygame.display.flip()
        gf.send_trigger(202)

        got_key = False
        max_duration = 120 * 1000 # 120 s
        t_start = pygame.time.get_ticks()
        pygame.event.clear()
        gx, gy = -65536, -65536

        while not (got_key or (pygame.time.get_ticks() - t_start) >= max_duration):
            for ev in pygame.event.get():
                if ev.type == KEYDOWN and ev.key == K_RETURN:
                    got_key = True

            # Dibuja imagen
            win.blit(im, (0, 0))

            # Dibuja cursor de mirada
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                gx = int(gaze_info.filtered_gaze_coordinates[0])
                gy = int(gaze_info.filtered_gaze_coordinates[1])
                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)

            # ----- Overlay de métricas -----
            m = mon.snapshot()
            overlay_lines = [
                f"CPU: {m['cpu']:.1f}%",
                f"RAM RSS: {m['rss_mb']:.1f} MB",
                f"Py Heap: {m['py_cur_mb']:.1f} MB (pico {m['py_peak_mb']:.1f} MB)",
                f"FPS: {clock.get_fps():.1f}"
            ]
            y = 10
            for line in overlay_lines:
                surf = font.render(line, True, (0, 0, 0))
                win.blit(surf, (10, y))
                y += 22
            # -------------------------------

            pygame.display.flip()
            clock.tick(60)  # limita a ~60 FPS para estabilidad de medición

    pygame.time.wait(100)
    gf.stop_sampling()

    # Guarda datos del gaze follower
    data_dir = "C:\\proyecto_aplica\\proyecto-aplicacion\\Pruebas_vali\\resultados"
    os.makedirs(data_dir, exist_ok=True)
    gf.save_data(os.path.join(data_dir, "test_session.csv"))

    # Detén monitor y guarda CSV con métricas
    perf_csv = os.path.join(data_dir, "perf_metrics.csv")
    mon.stop_and_dump_csv(perf_csv)

    gf.release()
    pygame.quit()
