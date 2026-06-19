# 基于 AMD ROCm 的高性能分布式数据并行与 HIP 静态图二维纳维-斯托克斯方程 PINN 求解器

本项目是一个工业级、针对 AMD ROCm 架构深度优化的物理信息神经网络 (PINN) 求解器，专门用于高性能求解稳态二维 Navier-Stokes 方程 (Kovasznay Flow 基准测试)。

该求解器框架专为 **AMD ROCm** 生态（如 Radeon PRO W7900 或 W7000 系列显卡）进行底层硬件级优化，提供分布式数据并行 (DDP) 扩展以及 HIP 静态图 (HIP Graph) 加速。

---

## 📖 数学物理方程

稳态二维不可压缩 Navier-Stokes 方程定义如下：

x 方向动量方程：
$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0
$$

y 方向动量方程：
$$
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0
$$

连续性方程（质量守恒）：
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

其中运动粘度系数 *ν* = 0.05，*u* 和 *v* 分别为速度分量，*p* 为压力。

### Kovasznay Flow 基准测试
求解域为 [-0.5, 1.0] × [-0.5, 1.5]。其解析解用于边界条件定义和误差精度验证：

$$
u_{true}(x, y) = 1 - e^{\lambda x} \cos(2\pi y)
$$

$$
v_{true}(x, y) = \frac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y)
$$

$$
p_{true}(x, y) = \frac{1}{2} (1 - e^{2\lambda x})
$$

其中参数 *λ* = 10 - √(100 + 4π²)。

---

## ⚡ ROCm 平台硬件优化亮点

传统 PINN 实现由于求解二阶导数时计算图过于复杂，极易面临严重的 CPU 瓶颈。本项目通过以下五项核心优化打破了性能瓶颈：

### 1. HIP 静态图录制 (HIP Graph)
- **瓶颈问题**：在 PyTorch 默认的 Eager 模式下，计算 Hessian 二阶导数会顺序启动数百个细碎的 GPU 算子。CPU-GPU 间的内核启动延迟极高，导致显卡核心利用率往往低于 10%。
- **解决方案**：针对单卡训练，我们将前向传播、二阶求导（Hessian）以及反向传播的整个过程录制为一个静态的 GPU 执行序列 (**HIP Graph**，对应 PyTorch 的 CUDAGraph API）。训练时 CPU 仅需调用一次 `g.replay()` 即可让 GPU 满载连续运行。这使显卡 SM 利用率从不足 20% 飙升至 90% 以上，训练速度提升达 3~10 倍。

### 2. GPUDataLoader (零拷贝高性能数据引擎)
- **瓶颈问题**：在每个 Epoch 重建 PyTorch 原生 `DataLoader` 迭代器会引入巨大的 CPU 调度开销。
- **解决方案**：我们使用自定义的 `GPUDataLoader` 替代了原生加载器。所有配点数据预先存放在 GPU 显存中。当批大小等于全量数据集时，会触发**指针零拷贝直通**（Zero-Copy Bypass），直接获取原始张量。在 mini-batch 模式下，数据的随机打乱 (`torch.randperm`) 同样完全在 GPU 显存内部执行。

### 3. PyTorch 原生 Autograd 自动微分优化
- **瓶颈问题**：传统第三方框架（如 DeepXDE）由于封装过深，经常在全局 Python 字典中缓存梯度，导致严重的 Python 运行时开销。
- **解决方案**：我们直接调用 PyTorch 原生的 `torch.autograd.grad` 进行高效一阶和二阶导数求解。通过巧妙运用 `grad_outputs=torch.ones_like(u)`，可在单次反向传播中同时获取对 x 和对 y 的偏导数，使 autograd 调用次数减半。

### 4. 动态精度与混合精度控制
- **BFloat16 AMP 混合精度**：能够充分利用硬件的 Tensor Cores（在 AMD 上为 Matrix Cores）加速矩阵乘法，同时保持与 Float32 相同的动态范围指数，防止标准 Float16 下计算二阶导数时频繁发生的数值下溢（Underflow）现象。
- **Float32 满精度**：当对物理场预测精度要求极高时，可通过 `--precision float32` 关闭混合精度。这能消除二阶导数中的量化噪音，将最大速度预测误差降至 0.005 以下。

### 5. 多卡分布式数据并行 (DDP)
- 支持基于 PyTorch DDP 的多卡（最高 8 卡）分布式并行加速训练。
- 自定义了可复现随机种子的 `DistributedSampler` 采样器，确保配点数据在不同卡之间均匀切分不重复。

---

## 🛠️ 安装依赖与 Docker 环境配置

本项目推荐使用 Docker 进行容器化部署。

### 1. Docker 环境配置 (推荐比赛评测使用)
项目目录下提供了适用于 AMD ROCm 显卡的容器配置文件 [Dockerfile_rocm](file:///home/elite/Antigravity-Project/PINN_Framework/Dockerfile_rocm)：

*   **构建 Docker 镜像**：
    ```bash
    docker build -f Dockerfile_rocm -t pinn-solver:rocm .
    ```
*   **启动容器运行**（挂载 AMD 显卡计算与渲染设备，并启用宿主机共享内存）：
    ```bash
    docker run -it --rm --device=/dev/kfd --device=/dev/dri --ipc=host pinn-solver:rocm
    ```

### 2. 本地 Python 环境安装
若直接在物理机环境运行，请确保系统已安装 **ROCm 驱动与 ROCm 版本的 PyTorch**，并在 Python 虚拟环境中执行依赖安装：
```bash
pip install -r requirements_linux_rocm.txt
```

---

## 🏃‍♂️ 比赛一键测试与运行指南

### 1. 一键测试基准脚本
项目根目录下提供了 [run_and_push.sh](file:///home/elite/Antigravity-Project/PINN_Framework/run_and_push.sh) 自动化一键测试脚本。该脚本会自动更新依赖、调度多卡 DDP 训练、清理旧的历史权重，并自动绘制硬件负载图。

*   **单卡一键测试**：
    ```bash
    chmod +x run_and_push.sh
    ./run_and_push.sh --scale small --precision float32 --gpus 0
    ```
*   **多卡一键并行测试**（以 8 卡并行 DDP 训练为例）：
    ```bash
    chmod +x run_and_push.sh
    ./run_and_push.sh --scale extreme --precision float32 --gpus 0,1,2,3,4,5,6,7
    ```

### 2. 精细常规测试命令
若需手动微调训练参数，可直接执行 `main.py` 入口：

*   **高精度单卡运行 (Float32)**：
    ```bash
    python main.py --scale large --batch_size 200000 --precision float32 --gpus 0
    ```
*   **高性能单卡运行 (BFloat16 AMP + HIP Graph 静态图加速)**：
    ```bash
    python main.py --scale large --batch_size 200000 --precision bfloat16 --gpus 0
    ```
*   **多卡 DDP 并行精细运行**（以 4 卡为例）：
    ```bash
    torchrun --nproc_per_node=4 main.py --scale extreme --precision float32 --batch_size 16384 --gpus 0,1,2,3
    ```

---

## 📊 硬件性能监控与分析

框架运行期间会自动将显存占用、功耗和利用率等性能指标记录在 `outputs_xxxx/profiling/hardware_metrics.log` 中。

可通过运行以下分析脚本一键绘制学术论文级别的负载变化图表：
```bash
python analyze_hardware.py
```
这将在 `outputs_xxxx/figures/` 目录下生成：
*   **`hardware_academic_profile.png`**：包含显存 OOM 边界线及 Hessian 阶段的 3x1 学术监控大图。
*   **单项指标变化图**：`vram_usage.png`, `power_usage.png`, `gpu_utilization.png`。

---

## 📂 项目目录结构说明

- `core/`：核心算法文件夹
  - [network.py](file:///home/elite/Antigravity-Project/PINN_Framework/core/network.py)：配置网络隐藏层结构及浮点参数。
  - [pde_def.py](file:///home/elite/Antigravity-Project/PINN_Framework/core/pde_def.py)：基于原生 Autograd 构建的二维 Navier-Stokes 方程约束。
  - [trainer.py](file:///home/elite/Antigravity-Project/PINN_Framework/core/trainer.py)：主 DDP 训练循环、HIP Graph 静态图及 GPUDataLoader 逻辑。
  - [profiler.py](file:///home/elite/Antigravity-Project/PINN_Framework/core/profiler.py)：监控并记录显存、功耗等物理硬件指标。
  - [visualizer.py](file:///home/elite/Antigravity-Project/PINN_Framework/core/visualizer.py)：流动预测场、误差场等学术图表的可视化脚本。
- [main.py](file:///home/elite/Antigravity-Project/PINN_Framework/main.py)：命令行解析与 DDP 初始化主入口。
- [analyze_hardware.py](file:///home/elite/Antigravity-Project/PINN_Framework/analyze_hardware.py)：提取性能日志并一键绘制图表。
