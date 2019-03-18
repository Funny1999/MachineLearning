import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = 'F:/Coursera-ML-AndrewNg-Notes-master/code/ex1-linear regression/ex1data1.txt'
data = pd.read_csv(path, header=None,names=['Population','Profit'])


#data.plot(kind="scatter", x='Population', y='Profit', figsize=(12,8))
#plt.show()
def computeCost(X, y, theta):
	inner = np.power((X*theta.T)-y,2)
	return np.sum(inner)/(2*len(X))

def graientDescent(X,y,theta,alpha,iters):
	temp = np.matrix(np.zeros(theta.shape))
	parameters = int(theta.ravel().shape[1])
	cost = np.zeros(iters)

	for i in range(iters):
		error = (X*theta.T)-y

		for j in range(parameters):
			term = np.multiply(error,X[:,j])
			temp[0,j] = theta[0,j] - (alpha/len(X)*np.sum(term))

		theta = temp
		cost[i] = computeCost(X,y,theta)

	return theta, cost

data.insert(0,'Ones',1)
cols = data.shape[1] #PS! data.shape[0]：多少行 data.shape[1]:多少列
X = data.iloc[:,0:cols-1]#PS! 保留除最后一列
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)#PS! np.matrix将X转换为numpy矩阵
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))#PS！ theta为一个(1,2)矩阵
alpha = 0.01
iters = 1000

g ,cost = graientDescent(X,y,theta,alpha,iters)


x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0] + (g[0,1] * x)
a = computeCost(X,y,g)
print(a)
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label="Predection")
ax.scatter(data.Population,data.Profit,label="Traning Data")
ax.legend(loc=1) #PS! label的位置
ax.set_xlabel("Population")
ax.set_ylabel("Profit")
ax.set_title("predicted Profic vs. Population")
plt.show()







