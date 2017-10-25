import numpy as np
import matplotlib.pyplot as plt

f = open("training_log.txt")

logs = f.read().split()

print logs

x = [i for i in range(len(logs))]
y = [float(i) for i in logs]

plt.plot(x,y,'*', x, y,'-')
plt.show()
