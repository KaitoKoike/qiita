import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.random.random(100)
y = 2*x + 0.5
y += np.random.random(100)-0.5

data = np.concatenate((x.reshape(100,1),y.reshape(100,1)),axis=1)
print(data.shape)
np.savetxt("linear_data.csv",data,delimiter=",")
plt.figure()
plt.scatter(x,y)
#plt.show()
plt.savefig("../image/linear_sample.jpg")
