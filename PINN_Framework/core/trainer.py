import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
import os
from .profiler import ProfilerContext, HardwareMonitor

def train_model(geom, pde_fn, funcs, num_domain, num_boundary, net, epochs=15000, batch_size=8192, profile=False):
    # DeepXDE 默认将全局设备设置为 cuda，这会导致 DataLoader 内部基于 CPU 的随机生成器 (Generator) 崩溃。
    # 因为我们已经在代码里手动使用了 .to(device) 转移张量，所以这里安全地将全局默认恢复为 cpu
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device('cpu')
        
    is_ddp = dist.is_initialized()
    local_rank = dist.get_rank() if is_ddp else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    net = net.to(device)
    if is_ddp:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
        
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # 1. 独立采样并构建 DataLoader (Mini-Batch) 以极大幅度削减系统 RAM 压力
    if local_rank == 0:
        print("[Trainer] Generating collocation points...")
    X_domain = geom.random_points(num_domain)
    X_bc = geom.random_boundary_points(num_boundary)
    
    dataset_domain = TensorDataset(torch.tensor(X_domain, dtype=torch.float32))
    dataset_bc = TensorDataset(torch.tensor(X_bc, dtype=torch.float32))
    
    # 启用 DistributedSampler 在多卡间分发数据
    sampler_domain = DistributedSampler(dataset_domain) if is_ddp else None
    sampler_bc = DistributedSampler(dataset_bc) if is_ddp else None
    
    loader_domain = DataLoader(dataset_domain, batch_size=batch_size, sampler=sampler_domain, shuffle=(sampler_domain is None), drop_last=False)
    bc_batch_size = max(1, int(batch_size * (len(X_bc) / len(X_domain))))
    loader_bc = DataLoader(dataset_bc, batch_size=bc_batch_size, sampler=sampler_bc, shuffle=(sampler_bc is None), drop_last=False)
    
    u_func, v_func, p_func = funcs
    
    if local_rank == 0:
        os.makedirs("outputs/checkpoints", exist_ok=True)
        # 传入所有 GPU 个数用于多路监控
        num_gpus = dist.get_world_size() if is_ddp else 1
        monitor = HardwareMonitor(interval=2.0, num_gpus=num_gpus)
        monitor.start()

    loss_history = []
    
    if local_rank == 0:
        print("[Trainer] Starting custom PyTorch DDP Mini-Batch training loop...")

    with ProfilerContext(use_profiler=(profile and local_rank == 0), log_dir="outputs/profiling/tensorboard_traces"):
        for epoch in range(epochs):
            if is_ddp:
                sampler_domain.set_epoch(epoch)
                sampler_bc.set_epoch(epoch)
                
            net.train()
            epoch_loss_pde = 0.0
            epoch_loss_bc = 0.0
            batches = 0
            
            for (batch_domain,), (batch_bc,) in zip(loader_domain, loader_bc):
                batch_domain = batch_domain.to(device)
                batch_bc = batch_bc.to(device)
                
                optimizer.zero_grad()
                
                # --- PDE Loss ---
                batch_domain.requires_grad_(True)
                y_pred_domain = net(batch_domain)
                residuals = pde_fn(batch_domain, y_pred_domain)
                loss_pde = sum(torch.mean(r**2) for r in residuals)
                
                # --- BC Loss ---
                y_pred_bc = net(batch_bc)
                u_pred, v_pred, p_pred = y_pred_bc[:, 0:1], y_pred_bc[:, 1:2], y_pred_bc[:, 2:3]
                
                u_true = torch.tensor(u_func(batch_bc.cpu().detach().numpy()), dtype=torch.float32, device=device)
                v_true = torch.tensor(v_func(batch_bc.cpu().detach().numpy()), dtype=torch.float32, device=device)
                p_true = torch.tensor(p_func(batch_bc.cpu().detach().numpy()), dtype=torch.float32, device=device)
                
                loss_bc = torch.mean((u_pred - u_true)**2) + \
                          torch.mean((v_pred - v_true)**2) + \
                          torch.mean((p_pred - p_true)**2)
                          
                loss = loss_pde + loss_bc * 10.0 # 增强 BC 权重
                
                loss.backward()
                optimizer.step()
                
                # 【核心修复】清除 DeepXDE 的全局梯度缓存！
                # DeepXDE 的 dde.grad 为了避免重复计算，会在后台用全局字典缓存前向计算图。
                # 由于我们使用了 DataLoader，每个 Batch 会产生新的张量，如果不清理，缓存会无限增大导致 VRAM 瞬间 OOM！
                import deepxde as dde
                dde.grad.clear()
                
                epoch_loss_pde += loss_pde.item()
                epoch_loss_bc += loss_bc.item()
                batches += 1
            
            # 计算当前卡上的平均 Batch Loss
            epoch_loss_pde /= max(1, batches)
            epoch_loss_bc /= max(1, batches)
            
            # 同步各卡之间的 Loss 以用于准确的终端打印
            if is_ddp:
                pde_t = torch.tensor(epoch_loss_pde, device=device)
                bc_t = torch.tensor(epoch_loss_bc, device=device)
                dist.all_reduce(pde_t, op=dist.ReduceOp.AVG)
                dist.all_reduce(bc_t, op=dist.ReduceOp.AVG)
                epoch_loss_pde = pde_t.item()
                epoch_loss_bc = bc_t.item()
            
            if local_rank == 0 and epoch % 100 == 0:
                print(f"Epoch {epoch:5d} | PDE Loss: {epoch_loss_pde:.4e} | BC Loss: {epoch_loss_bc:.4e}")
                loss_history.append((epoch, epoch_loss_pde, epoch_loss_bc))
                if epoch % 1000 == 0:
                    torch.save(net.state_dict(), f"outputs/checkpoints/model_ep{epoch}.pt")

    if local_rank == 0:
        monitor.stop()
        
    return net, loss_history
