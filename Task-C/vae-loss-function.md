# C. VAE Loss function

|   |   |
|---|---|
| Ограничение времени | 20 секунд |
| Ограничение памяти | 256Mb |
| Ввод | |
| Вывод | стандартный вывод или tests.log |

Исправьте ошибки в функции потерь вариационного автоэнкодера. Ниже указан формат посылки для вашего решения.

```
import torch
from torch.distributions import Independent, Normal, Bernoulli

d, nh, D = 32, 200, 28 * 28

def loss_vae(x, encoder, decoder):
    pass
```
