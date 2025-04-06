import torch
import os
import matplotlib

matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt


def generate_tsp_dataset(save_path, num_instances=1000, num_nodes=20, scale=750):
    """生成并保存数据集"""
    coords = torch.rand(num_instances, num_nodes, 2) * scale

    data_dir = os.path.dirname(save_path)
    os.makedirs(data_dir, exist_ok=True)

    torch.save({'x': coords}, f"{save_path}.pkl")
    print(f"数据集已保存至: {os.path.abspath(save_path)}.pkl")


def visualize_tsp_instance(data_path, instance_idx=0, save_path=None):
    """可视化并保存为图片"""
    data = torch.load(data_path, map_location='cpu')
    coords = data['x'][instance_idx].numpy()

    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50)
    plt.title(f"TSP{coords.shape[0]} Visualization")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {os.path.abspath(save_path)}")
    plt.close()


# 生成并可视化
if __name__ == "__main__":
    # 生成数据集
    generate_tsp_dataset(
        save_path="data/visual_friendly_tsp20",
        num_instances=1000,
        num_nodes=20,
        scale=750
    )

    # 可视化并保存
    visualize_tsp_instance(
        "data/visual_friendly_tsp20.pkl",
        instance_idx=0,
        save_path="data/tsp_visualization.png"
    )