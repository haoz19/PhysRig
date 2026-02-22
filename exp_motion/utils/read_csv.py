import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# /scratch/bbqk/haozhang/.conda/envs/physdreamer/bin/python read_csv.py

def plot_youngs_distribution(csv_file, youngs_column='Youngs_Modulus', bins=50):
    """
    从csv_file中读取Young's Modulus值，并绘制直方图+KDE曲线。
    
    参数：
        csv_file (str): CSV 文件路径
        youngs_column (str): CSV 中存储 Young's 值的列名
        bins (int): 直方图 bin 的数量
    """
    # 1. 读取CSV
    df = pd.read_csv(csv_file)
    
    # 2. 提取 Young's Modulus 数据
    if youngs_column not in df.columns:
        raise ValueError(f"列名 '{youngs_column}' 在 CSV 文件中不存在，请检查文件列名。")
    
    youngs_data = df[youngs_column].values

    # 3. 获取 CSV 文件所在的目录，并生成 PNG 文件路径
    output_dir = os.path.dirname(csv_file)
    output_png = os.path.join(output_dir, "youngs_distribution.png")
    
    # 4. 使用 seaborn 绘制直方图和 KDE 曲线
    plt.figure(figsize=(8, 5))
    sns.histplot(youngs_data, bins=bins, kde=True, color='blue', edgecolor='black')
    
    # 5. 画面设置
    plt.title("Distribution of Young's Modulus")
    plt.xlabel("Young's Modulus")
    plt.ylabel("Frequency")
    
    # 6. 保存到 CSV 文件所在目录
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[Info] Figure saved to {output_png}")
    plt.show()


if __name__ == "__main__":
    
    # 你的 CSV 文件路径
    csv_file = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/output/whale/whale_sp0_mulmat_AdamW_1.1_youngs_10000.0_lr_800.0_substep_100_iters_30_sw_60/youngs_map.csv"
    
    # 运行绘制函数
    plot_youngs_distribution(csv_file, youngs_column='Youngs_Modulus', bins=50)