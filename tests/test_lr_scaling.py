import torch
import torch.nn as nn
import torch.optim as optim

# simple one-step training to check LR scaling
x = torch.tensor([[1.0]])
y = torch.tensor([[2.0]])
loss_fn = nn.MSELoss()

def run(lr):
    m = nn.Linear(1,1,bias=False)
    m.weight.data.fill_(1.0)
    opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    opt.zero_grad()
    loss_fn(m(x), y).backward()
    opt.step()
    return m.weight.item()

w1 = run(0.05)
w2 = run(0.1)
print("update_ratio", round((1-w1)/(1-w2), 3))
