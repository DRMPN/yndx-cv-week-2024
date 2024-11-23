import torch
from torch import nn

d, nh, D = 32, 200, 28 * 28

# Скорректированный энкодер
enc = nn.Sequential(
    nn.Linear(D, nh),
    nn.ReLU(),
    nn.Linear(nh, nh),
    nn.ReLU(),
    nn.Linear(nh, 2 * d)  # Выход размерности 2*d
)

# Скорректированный декодер
dec = nn.Sequential(
    nn.Linear(d, nh),
    nn.ReLU(),
    nn.Linear(nh, nh),
    nn.ReLU(),
    nn.Linear(nh, D)
)
