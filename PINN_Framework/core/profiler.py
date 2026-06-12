import os
import time
import threading
import subprocess

class HardwareMonitor:
    def __init__(self, log_dir="outputs/profiling", interval=1.0, num_gpus=1):
        self.log_dir = log_dir
        self.interval = interval
        self.num_gpus = num_gpus
        self.running = False
        self.thread = None
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "hardware_metrics.log")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def _monitor_loop(self):
        with open(self.log_file, "w") as f:
            f.write("Time,Backend,GPU_ID,VRAM_MB,Power_W,Temp_C\n")
            while self.running:
                # 尝试 AMD_SMI 探测多卡
                try:
                    import amdsmi
                    amdsmi.amdsmi_init()
                    devices = amdsmi.amdsmi_get_processor_handles()
                    
                    # 遍历探测到的所有 GPU
                    for i, handle in enumerate(devices):
                        try:
                            temp = amdsmi.amdsmi_get_temp_metric(handle, amdsmi.AmdSmiTemperatureType.EDGE, amdsmi.AmdSmiTemperatureMetric.CURRENT)
                            power = amdsmi.amdsmi_get_power_info(handle).average_socket_power
                            vram = amdsmi.amdsmi_get_vram_usage(handle) / (1024 * 1024)
                            f.write(f"{time.time():.2f},AMD_API,GPU_{i},{vram:.1f},{power:.1f},{temp}\n")
                        except:
                            pass
                    f.flush()
                    time.sleep(self.interval)
                    continue
                except Exception:
                    pass
                
                # 如果 AMD_SMI 失败，尝试 nvidia-smi 探测多卡
                try:
                    res = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,memory.used,power.draw,temperature.gpu", "--format=csv,noheader,nounits"], 
                        stderr=subprocess.STDOUT
                    ).decode()
                    for line in res.strip().split('\n'):
                        parts = line.split(',')
                        if len(parts) == 4:
                            idx, vram, power, temp = parts
                            f.write(f"{time.time():.2f},NVIDIA,GPU_{idx.strip()},{vram.strip()},{power.strip()},{temp.strip()}\n")
                    f.flush()
                except Exception:
                    # CPU 或无探测工具
                    pass
                
                time.sleep(self.interval)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

class ProfilerContext:
    def __init__(self, use_profiler=False, log_dir="outputs/profiling/tensorboard_traces"):
        self.use_profiler = use_profiler
        self.log_dir = log_dir
        self.prof = None

    def __enter__(self):
        if self.use_profiler:
            import torch
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.prof.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_profiler and self.prof:
            self.prof.stop()
