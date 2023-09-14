import torch
import matplotlib.pyplot as plt
import numpy as np
 
loss = torch.load('./loss1.pth')
num = len(loss)
x = [i + 1 for i in range(num)]
print(num)
plt.figure(1)
plt.plot(loss[:])
# plt.show()
plt.savefig("loss1.png")
