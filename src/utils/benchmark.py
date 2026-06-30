import psutil
import time
import threading

class ResourceMonitor:
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.is_running = False
        self.monitor_thread = None

    def _monitor_loop(self):
        while self.is_running:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            print(f"📊 [System Edge Benchmark] CPU: {cpu}% | RAM: {ram}%")
            time.sleep(self.interval)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("🟢 Resource monitoring started.")

    def stop(self):
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            print("🔴 Resource monitoring stopped.")