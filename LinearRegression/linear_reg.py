import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = np.loadtxt("data/linear_data.csv",delimiter=",")
x = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)

print(x.shape)
reg = LinearRegression().fit(x,y)
print(reg.coef_)
print(reg.intercept_)

mesh_x = np.arange(0,1,0.001)
mesh_y = reg.coef_*mesh_x+reg.intercept_

plt.figure()
plt.scatter(x,y)
plt.scatter(mesh_x,mesh_y,s=1)
#plt.show()
plt.savefig("image/linear_reg.jpg")