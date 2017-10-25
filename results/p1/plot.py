import numpy as np
import matplotlib.pyplot as plt

f = open("training_log.txt")

logs = f.read().split()

print logs

x = [i for i in range(len(logs))]
y = [float(i) for i in logs]

x = [i for i in range(len(logs)/2)]

i = 0
y = []
acc = []
while i < len(logs):
    y.append(float(logs[i]))
    i+=1
    acc.append(float(logs[i]))
    i+=1

#plt.plot(x,y,'*', x, y,'-')
#plt.ylabel('accuracy rate')
plt.ylabel('log-likelihood')
plt.xlabel('iteration')
plt.plot(x,y,'*', x, y,'-')
plt.show()
