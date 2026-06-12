import os
# 设置为 PyTorch 后端
os.environ["DDE_BACKEND"] = "pytorch"

import argparse
import torch
import torch.distributed as dist
from core.pde_def import get_ns_equation_data
from core.network import build_network
from core.trainer import train_model
from core.visualizer import plot_ns_results

def parse_args():
    parser = argparse.ArgumentParser(description="PINN Framework with 8-GPU DDP Support")
    parser.add_argument("--epochs", type=int, default=15000, help="Number of training epochs")
    parser.add_argument("--scale", type=str, choices=["small", "large", "extreme"], default="small", 
                        help="Data scale: small (local test), large (server smooth), extreme (OOM boundary/8-GPU max)")
    parser.add_argument("--precision", type=str, choices=["float32", "float16", "bfloat16"], default="float32",
                        help="Precision format. Note: float16 underflows Hessian. bfloat16 recommended for ROCm.")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler for performance tracing")
    parser.add_argument("--batch_size", type=int, default=0, help="Mini-batch size. 0 means auto-scale to saturate GPU based on scale.")
    return parser.parse_args()

def init_distributed():
    """初始化 DDP 分布式进程组。兼容单卡与多卡。"""
    is_ddp = False
    local_rank = 0
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl") # NCCL on ROCm defaults to RCCL
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_ddp = True
    return is_ddp, local_rank

def print_hardware_info(is_ddp):
    print("=" * 60)
    print("Hardware & Environment Info:")
    print(f"DeepXDE Backend: {os.environ['DDE_BACKEND']}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"DDP Multi-GPU Mode: {'ENABLED' if is_ddp else 'DISABLED'}")
    if is_ddp:
        print(f"World Size: {dist.get_world_size()} GPUs")
    if torch.cuda.is_available():
        print("Device Available: True (GPU Detected)")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("Device Available: False (Running on CPU)")
    print("=" * 60)

def main():
    # 1. 尝试初始化 DDP
    is_ddp, local_rank = init_distributed()
    args = parse_args()
    
    # 2. 动态自适应 Batch Size 分配 (榨干硬件算力)
    if args.batch_size == 0:
        if args.scale == "small":
            args.batch_size = 2000     # 全量运行，不切分
        elif args.scale == "large":
            args.batch_size = 40000    # 大约占用 6-7GB VRAM
        elif args.scale == "extreme":
            args.batch_size = 150000   # 极限压榨：大约占用 24GB VRAM (刚好塞满 4090，在 W7900 上也能跑出极高并发)

    if local_rank == 0:
        print_hardware_info(is_ddp)
        print(f"[Main] Scale: {args.scale.upper()} | Precision: {args.precision.upper()} | Batch: {args.batch_size}")

    # 2. 获取几何结构与函数，不再生成深耦合的 dde.data.PDE
    geom, pde, funcs, num_domain, num_boundary, num_test = get_ns_equation_data(scale_factor=args.scale)

    # 3. 构建神经网络
    net = build_network(scale_factor=args.scale, precision=args.precision)

    # 4. 执行 DDP/单卡自适应的自定义 PyTorch 训练循环
    trained_net, loss_history = train_model(
        geom=geom, pde_fn=pde, funcs=funcs,
        num_domain=num_domain, num_boundary=num_boundary,
        net=net, epochs=args.epochs, batch_size=args.batch_size, profile=args.profile
    )

    # 5. 仅在主进程进行流场生成，防止 8 个进程同时读写 IO 冲突
    if local_rank == 0:
        print("\n[Main] Training finished. Generating 2D Navior-Stokes visualizations...")
        
        # 因为在 custom trainer 中生成了 loss_history 列表，保存它
        import numpy as np
        import os
        os.makedirs("outputs/figures", exist_ok=True)
        np.savetxt("outputs/figures/loss.dat", loss_history, header="Epoch, PDE_Loss, BC_Loss", comments="")
        
        # 为了给可视化函数喂测试数据，我们在主进程单独采样
        X_test = geom.random_points(num_test)
        # 解包 DDP net 提取底层模块用于推理
        base_net = trained_net.module if hasattr(trained_net, 'module') else trained_net
        base_net.eval()
        
        # 使用 visualizer 中改造过的绘制接口
        plot_ns_results(base_net, funcs, X_test)
        print("\n✅ All artifacts saved in outputs/ directory.")

if __name__ == "__main__":
    main()
