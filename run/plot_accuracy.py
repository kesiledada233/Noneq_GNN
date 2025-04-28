import json
import matplotlib.pyplot as plt

# 定义文件路径
# file_paths = [
#     "./results/example_cpu/agg/train/stats.json",
#     "./results/example_noneq_1/agg/train/stats.json",
#     "./results/example_noneq_2/agg/train/stats.json",
#     "./results/example_noneq_3/agg/train/stats.json",
#     "./results/example_noneq_4/agg/train/stats.json",
# ]

file_paths = [
    "./results/example_cpu/agg/val/stats.json",
    "./results/example_noneq_5/agg/val/stats.json",
    "./results/example_noneq_6/agg/val/stats.json",
    "./results/example_noneq_7/agg/val/stats.json",
    "./results/example_noneq_9/agg/val/stats.json",
]

# 定义配置名称（用于图例）
config_names = ["cpu", "noneq_5", "noneq_6", "noneq_7", "noneq_9"]

# 初始化存储数据的列表
all_epochs = []
all_accuracies = []
average_accuracies = []

# 读取每个文件的数据
for file_path in file_paths:
    epochs = []
    accuracies = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data["epoch"])
            accuracies.append(data["accuracy"])
    all_epochs.append(epochs)
    all_accuracies.append(accuracies)

    # 计算当前配置的 accuracy 平均值（剔除第一个值）
    avg_accuracy = sum(accuracies[1:]) / len(accuracies[1:])
    average_accuracies.append(avg_accuracy)
    
# 打印每个配置的平均 accuracy
for config_name, avg_accuracy in zip(config_names, average_accuracies):
    print(f"Average accuracy for {config_name}: {avg_accuracy:.4f}")

# 绘制图表
plt.figure(figsize=(10, 6))

# 为每个配置绘制曲线
for i in range(len(file_paths)):
    plt.plot(all_epochs[i], all_accuracies[i], label=config_names[i])

# 添加标题和标签
plt.title("Accuracy Comparison Across Configurations(Validaiton)", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True)

# # 设置显示范围
# plt.xlim(200, 400)  # 限制 x 轴范围为 200 到 400
# plt.ylim(0.8, 0.9)  # 限制 y 轴范围为 0.8 到 0.9

# 添加图例
plt.legend(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()