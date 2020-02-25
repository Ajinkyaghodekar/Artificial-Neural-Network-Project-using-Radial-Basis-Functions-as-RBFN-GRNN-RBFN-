import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split 

data = pd.read_csv('international-airline-passengers.csv')
data.drop(data.tail(1).index,inplace=True)

print(data.head())
print(data['Month'])



output_df = pd.DataFrame()

output_df['Output'] = data["Day"]

plt.plot(output_df)
plt.title('Data')
plt.show()

# taking out the series from 1700 to 1979
series = output_df['Output'].values

# to check the data
print(series)
print('Length of series',len(series))


def preparing_dataset(series,look_back):
    data = np.empty([look_back+1])
    for idx in range(series.shape[0]):
        if idx >= look_back:
            temp_data = series[idx-look_back : idx+1]
            data = np.vstack([data, temp_data])

    # creating X and Y
    x = data[1:, :-1]
    y = data[1:, -1]
    return x,y
x,y = preparing_dataset(series,20)

# prepare train and test data
train_size = 100
x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#y_train = np.ravel(scaler.fit_transform(y_train.reshape(-1, 1)))
#y_test = np.ravel(scaler.transform(y_test.reshape(-1, 1)))

print(x_train)
# MLP 
# Train on scaled data
# selecting 25 hidden neurons in total with 5*5 dimensions to prevent over fitting
reg = MLPRegressor(hidden_layer_sizes=(1000,),activation='logistic',solver='lbfgs')
reg.fit(x_train,y_train)


# predict the values after the train data
y_pred = np.empty([y_test.shape[0]])
for i in range(x_test.shape[0]):
    pred = reg.predict(x_test[i].reshape(1,-1))
    y_pred[i] = pred


plt.plot(np.arange(0, y_train.shape[0]), y_train)
plt.plot(np.arange(y_train.shape[0],y_train.shape[0]+y_test.shape[0]), y_test)
plt.plot(np.arange(y_train.shape[0],y_train.shape[0]+y_pred.shape[0]), y_pred)
plt.legend(['train', 'test', 'prediction'])
plt.title('MLP')
plt.show()
# Mean error
mse = np.mean((y_pred - y_test) ** 2)
print("MLP mse: ", mse)

# Linear Regression
from sklearn.linear_model import LinearRegression
reg_1 = LinearRegression()
reg_1.fit(x_train,y_train)


# predict the values after the train data
y_pred = np.empty([y_test.shape[0]])
for i in range(x_test.shape[0]):
    pred = reg_1.predict(x_test[i].reshape(1,-1))
    y_pred[i] = pred


plt.plot(np.arange(0, y_train.shape[0]), y_train)
plt.plot(np.arange(y_train.shape[0],y_train.shape[0]+y_test.shape[0]), y_test)
plt.plot(np.arange(y_train.shape[0],y_train.shape[0]+y_pred.shape[0]), y_pred)
plt.legend(['train', 'test', 'prediction'])
plt.title('Linear Regression')
plt.show()

# Mean error
mse = np.mean((y_pred - y_test) ** 2)
print("Linear Regression mse: ", mse)

# Generalized Regression Neural Nets GRNN
from neupy import algorithms

grnn = algorithms.GRNN(std=0.05, verbose=False)
grnn.fit(x_train,y_train)

# predict the values after the train data
y_pred = np.empty([y_test.shape[0]])
for i in range(x_test.shape[0]):
    pred = grnn.predict(x_test[i].reshape(1,-1))
    y_pred[i] = pred

plt.plot(np.arange(0, y_train.shape[0]), y_train)
plt.plot(np.arange(y_train.shape[0],y_train.shape[0]+y_test.shape[0]), y_test)
plt.plot(np.arange(y_train.shape[0],y_train.shape[0]+y_pred.shape[0]), y_pred)
plt.legend(['train', 'test', 'prediction'])
plt.title('Generalized Regression')
plt.show()

# Mean error
mse = np.mean((y_pred - y_test) ** 2)
print("GRNN mse: ", mse)


# RBF Nets

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

def kmeans(X, k):
    """Performs k-means clustering for 1D input
    
    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters
    
    Returns:
        ndarray -- A kx1 array of final cluster centers
    """
 
    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
 
    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
 
        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)
 
        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
 
        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()
 
    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)
 
    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])
 
    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))
 
    return clusters, stds

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
 
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
        
    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std 
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)

# making the x axis of the series for prediction
x = np.arange(series.size)

# choose the learning rate . Usually 0.01 works well.
# choose the clusters in the right amount(k)
# choosing more k value will result in over fitting and less value may lead to less accuracy
rbfnet = RBFNet(lr=1e-2, k=50)
rbfnet.fit(x,series)

y_pred = rbfnet.predict(x)
 
plt.plot(x, series, '-o', label='true')
plt.plot(x, y_pred, '-o', label='RBF-Net')
plt.legend()
 
plt.tight_layout()
plt.show()

