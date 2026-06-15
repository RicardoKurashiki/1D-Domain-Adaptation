import time
import threading
import torch
import psutil


class ComputationalMetrics():
    def __init__(self, trainable_params: int, model_size: float):
        self.trainable_params = trainable_params
        self.model_size = model_size
        self.start_time = None
        self.end_time = None
        self.peak_cpu_memory = None
        self.peak_gpu_memory = None
        self._process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None

    def _monitor(self, interval):
        while self._monitoring:
            rss = self._process.memory_info().rss / 1024**2
            if self.peak_cpu_memory is None or rss > self.peak_cpu_memory:
                self.peak_cpu_memory = rss
            time.sleep(interval)

    def start(self, interval=0.5):
        self.start_time = time.time()
        self.peak_cpu_memory = None
        self.peak_gpu_memory = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor, args=(interval,), daemon=True)
        self._monitor_thread.start()

    def finish(self):
        self.end_time = time.time()
        self._monitoring = False
        if self._monitor_thread is not None:
            self._monitor_thread.join()
            self._monitor_thread = None
        rss = self._process.memory_info().rss / 1024**2
        if self.peak_cpu_memory is None or rss > self.peak_cpu_memory:
            self.peak_cpu_memory = rss
        if torch.cuda.is_available():
            self.peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            self.peak_gpu_memory = 0.0

    @property
    def training_time(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_json(self):
        return {
            "trainable_params": self.trainable_params,
            "model_size_mb": round(self.model_size, 6) if self.model_size is not None else "",
            "training_time_s": round(self.training_time, 6) if self.training_time is not None else "",
            "peak_gpu_memory_mb": round(self.peak_gpu_memory, 6) if self.peak_gpu_memory is not None else "",
            "peak_cpu_memory_mb": round(self.peak_cpu_memory, 6) if self.peak_cpu_memory is not None else "",
        }
