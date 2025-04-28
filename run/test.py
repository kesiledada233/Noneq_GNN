import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu = nn.Dropout(0.5)

model = MyModule()

print(list(model.children()))