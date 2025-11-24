import math
import numpy as np
import matplotlib.pyplot as plt

def softmax_with_temperature(logits, T=1.0):
    """带温度的 softmax"""
    logits = np.array(logits, dtype=float)
    exps = np.exp(logits / T)
    return exps / exps.sum()

logits = [1, 2, 3]
temperatures = [0.5, 1.0, 2.0, 5.0]

x = np.arange(len(logits))  # 类别索引 0,1,2
width = 0.18

fig, ax = plt.subplots()

for i, T in enumerate(temperatures):
    probs = softmax_with_temperature(logits, T)
    ax.bar(x + i * width, probs, width, label=f"T={T}")

ax.set_xticks(x + width * (len(temperatures) - 1) / 2)
ax.set_xticklabels(['class1', 'class2', 'class3'])
ax.set_ylabel('Probability')
ax.set_title('Softmax with Different Temperature (logits=[1,2,3])')
ax.legend()

plt.show()
