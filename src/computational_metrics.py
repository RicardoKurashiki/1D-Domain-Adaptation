class ComputationalMetrics():
    def __init__(self, trainable_params:int, model_size:float):
        self.trainable_params = trainable_params
        self.model_size = model_size
        self.start_time = None
        self.end_time = None
        self.peak_cpu_memory = None
        self.peak_gpu_memory = None

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.end_time = time.time()
