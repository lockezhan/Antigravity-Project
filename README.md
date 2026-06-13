# Antigravity-Project 🌌

Welcome to the **Antigravity-Project** repository! This repository serves as a centralized hub for high-performance scientific computing, Physics-Informed Machine Learning (PIML / PINNs), and advanced AI engineering implementations developed under the Antigravity Project initiative.

Our goal is to build industry-grade, highly optimized solver frameworks for physical systems that maximize computational efficiency on cutting-edge hardware, supporting both NVIDIA CUDA and AMD ROCm architectures.

---

## 📂 Project Directory Index

The repository is structured to host multiple independent scientific computing projects. Below is the index of current and upcoming projects:

### 1. 🚀 [PINN_Framework](./PINN_Framework/) (Active)
An industry-grade Physics-Informed Neural Network (PINN) solver for the **2D Navier-Stokes equations** (Kovasznay Flow).
* **Core Technologies**: CUDA Graphs static recording, `GPUDataLoader` zero-copy memory batching, Native PyTorch Autograd, Multi-GPU DDP (DistributedDataParallel), and dynamic FP32/BF16 precision selection.
* **Hardware Support**: Fully compatible with NVIDIA GPUs (e.g., RTX 4090 D) and AMD ROCm GPUs (e.g., Radeon PRO W7900).
* **Documentation**: See the detailed [PINN_Framework/README.md](./PINN_Framework/README.md) for architecture, installation, and usage instructions.

### 2. ⏳ [Upcoming Solvers & HPC Projects] (Planned)
* **3D Turbulent Flow Solver**: Extending the current 2D PINN framework to support complex 3D Navier-Stokes boundary conditions.
* **Sympy-to-PyTorch Automatic PDE Compiler**: A module to compile symbolic PDE definitions into optimized native PyTorch computational graphs automatically.
* **DeepOnet & Operator Learning Framework**: High-performance implementations of Operator Networks for real-time PDE modeling.

---

## 🛠️ Getting Started

To get started with the projects in this repository:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lockezhan/Antigravity-Project.git
   cd Antigravity-Project
   ```

2. **Navigate to the target project**:
   Each subdirectory is self-contained with its own dependencies and configuration files. For example, to run the PINN framework:
   ```bash
   cd PINN_Framework
   # Refer to PINN_Framework/README.md for specific setup steps.
   ```

---

## 📈 System Architecture & Multi-GPU Profiling

We emphasize hardware efficiency. Every project in this repository includes unified profiling tools to capture:
* **VRAM footprint [GB]**
* **GPU Core Utilization (SM load) [%]**
* **Real-time Power Draw [W]**

These metrics are compiled into academic-standard 3x1 multi-panel stacked plots for reporting and publication purposes.

---

## 🤝 Contributing

Contributions to high-performance solvers, optimizations, and new physical model implementations are welcome. Please open an issue or submit a pull request on the relevant branch.

## 📄 License

This repository is licensed under the MIT License. See individual directories for specific licenses if applicable.
