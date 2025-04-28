import json
import matplotlib.pyplot as plt

# 读取example_noneq验证集文件
val_noneq_file_path = "./results/example_noneq_4/agg/val/stats.json"
val_data_noneq = []

with open(val_noneq_file_path, "r") as f:
    for line in f:
        val_data_noneq.append(json.loads(line))

# 读取example_cpu验证集文件
val_cpu_file_path = "./results/example_cpu/agg/val/stats.json"
val_data_cpu = []

with open(val_cpu_file_path, "r") as f:
    for line in f:
        val_data_cpu.append(json.loads(line))

# 提取验证集 noneq & cpu 的 epoch 和 accuracy 数据
val_noneq_epochs = [entry["epoch"] for entry in val_data_noneq]
val_noneq_accuracy = [entry["accuracy"] for entry in val_data_noneq]

val_cpu_epochs = [entry["epoch"] for entry in val_data_cpu]
val_cpu_accuracy = [entry["accuriacy"] for entry in val_data_cpu]

# 绘制图表
plt.figure(figsize=(10, 6))

# 验证集 accuracy 曲线
plt.plot(val_noneq_epochs, val_noneq_accuracy, label="Validation Accuracy of Example_noneq", color="blue", linestyle="-", marker="o")

# 训练集 accuracy 曲线
plt.plot(val_cpu_epochs, val_cpu_accuracy, label="Validation Accuracy of Example_cpu", color="orange", linestyle="--", marker="x")

# 添加图表标题和标签
plt.title("Validation Accuracy Comparison: Example_noneq_4 vs Example_cpu", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.ylim(0.8, 0.9)  # 设置 y 轴范围以突出对比
plt.grid(True)

# 添加图例
plt.legend(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()