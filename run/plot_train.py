import json
import matplotlib.pyplot as plt

# 读取 stats.json 文件
file_path = "./results/example_cpu/agg/train/stats.json"
data = []

with open(file_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

# Energy parameter:
energy_alpha = 0.5
energy_beta = 2.0
energy_reg = True

# 提取数据
epochs = [entry["epoch"] for entry in data]
loss = [entry["loss"] for entry in data]
accuracy = [entry["accuracy"] for entry in data]
learning_rate = [entry["lr"] for entry in data]
time_iter = [entry["time_iter"] for entry in data]

# Sum total time cost
total_time = sum(time_iter)
print(f"Total time cost: {total_time:.2f} seconds")

# 绘制图表
plt.figure(figsize=(12, 8))

# Loss 曲线
plt.subplot(2, 2, 1)
plt.plot(epochs, loss, label="Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.grid(True)
plt.legend()

# Accuracy 曲线
plt.subplot(2, 2, 2)
plt.plot(epochs, accuracy, label="Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.grid(True)
plt.legend()

# Learning Rate 曲线
plt.subplot(2, 2, 3)
plt.plot(epochs, learning_rate, label="Learning Rate", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate over Epochs")
plt.grid(True)
plt.legend()

# Time per Iteration 曲线
plt.subplot(2, 2, 4)
plt.plot(epochs, time_iter, label="Time per Iteration", color="red")
plt.xlabel("Epoch")
plt.ylabel("Time per Iteration (s)")
plt.title("Time per Iteration over Epochs")
plt.grid(True)
plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.show()