import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot():
    log_file = "outputs/profiling/hardware_metrics.log"
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found.")
        return
        
    print(f"Parsing distributed hardware metrics from {log_file}...")
    
    # 使用 pandas 解析带标题的 CSV
    try:
        df = pd.read_csv(log_file)
        # 清理列名和数据中的空格
        df.columns = [c.strip() for c in df.columns]
        for col in ['Backend', 'GPU_ID']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    except Exception as e:
        print(f"Error parsing log: {e}")
        return
        
    if df.empty or 'GPU_ID' not in df.columns:
        print("No valid distributed hardware data found.")
        return
        
    # 时间归一化
    t0 = df['Time'].min()
    df['Time_Min'] = (df['Time'] - t0) / 60.0
    
    os.makedirs("outputs/figures", exist_ok=True)
    
    # 颜色循环以支持多达 8 卡
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # === 图1：多卡 VRAM 占用折线图 ===
    plt.figure(figsize=(12, 6))
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        vram_gb = group['VRAM_MB'] / 1024.0
        c = colors[i % len(colors)]
        plt.plot(group['Time_Min'], vram_gb, label=f"VRAM ({gpu_id})", color=c, linewidth=2, alpha=0.8)
        
    plt.xlabel("Training Time (Minutes)", fontsize=12)
    plt.ylabel("VRAM Usage (GB)", fontsize=12)
    plt.title("Multi-GPU VRAM Consumption over Time", fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/figures/vram_usage.png", dpi=300)
    plt.close()
    
    # === 图2：多卡功耗折线图 ===
    plt.figure(figsize=(12, 6))
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        c = colors[i % len(colors)]
        plt.plot(group['Time_Min'], group['Power_W'], label=f"Power ({gpu_id})", color=c, linewidth=2, alpha=0.8)
        
    plt.xlabel("Training Time (Minutes)", fontsize=12)
    plt.ylabel("Power Draw (Watts)", fontsize=12)
    plt.title("Multi-GPU Power Draw over Time", fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/figures/power_usage.png", dpi=300)
    plt.close()
    
    print("\n✅ Multi-GPU Analysis Complete!")
    print("  - VRAM Curves: outputs/figures/vram_usage.png")
    print("  - Power Curves: outputs/figures/power_usage.png")

if __name__ == "__main__":
    analyze_and_plot()
