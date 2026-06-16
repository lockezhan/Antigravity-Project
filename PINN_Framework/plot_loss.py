import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot loss curve from existing loss.dat")
    parser.add_argument("--dir", type=str, default="outputs", help="Directory containing figures/loss.dat")
    args = parser.parse_args()
    
    loss_file = os.path.join(args.dir, "figures/loss.dat")
    if not os.path.exists(loss_file):
        print(f"Error: {loss_file} not found. Make sure the training in this directory has completed.")
        return
        
    try:
        # 加载数据，跳过第一行表头
        data = np.loadtxt(loss_file, skiprows=1)
        # 如果只有一行数据，进行维度适配
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        epochs = data[:, 0]
        pde_losses = data[:, 1]
        bc_losses = data[:, 2]
        total_losses = pde_losses + bc_losses
        
        plt.figure(figsize=(10, 6), dpi=300)
        plt.semilogy(epochs, pde_losses, label='PDE Loss', color='#1f77b4', alpha=0.8, linewidth=1.5)
        plt.semilogy(epochs, bc_losses, label='BC Loss', color='#ff7f0e', alpha=0.8, linewidth=1.5)
        plt.semilogy(epochs, total_losses, label='Total Loss', color='#2ca02c', alpha=0.9, linewidth=2.0)
        
        plt.xlabel('Epoch', fontsize=11, fontweight='semibold')
        plt.ylabel('Loss (Log Scale)', fontsize=11, fontweight='semibold')
        plt.title('PINN Navier-Stokes Solver Convergence History', fontsize=13, fontweight='bold', pad=12)
        plt.grid(True, which="both", linestyle=':', alpha=0.5)
        plt.legend(fontsize=10, loc='upper right')
        
        out_image = os.path.join(args.dir, "figures/loss_curve.png")
        plt.savefig(out_image, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ Loss curve successfully plotted and saved to: {out_image}")
    except Exception as e:
        print(f"Error loading or plotting loss: {e}")

if __name__ == "__main__":
    main()
