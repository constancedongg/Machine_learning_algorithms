import numpy as np 
class KNN:
    def __init__(self , X_train , y_train , K):
        self.X_train = X_train
        self.y_train = y_train
        self.K = K
        
    def predict(self , X):
        y_pred = np.array([])
        for row in X:
            # calculate the distance between observation in test data and each observations of training data
            dist = np.sum((row - self.X_train) ** 2 , axis=1)

            # form 2d numpy array, first column is true label, second column is distance
            y_dist = np.concatenate((self.y_train.reshape(self.y_train.shape[0] , 1), dist.reshape(dist.shape[0],1)) , axis = 1)
            
            # sort the 2d array with ascending distance 
            y_dist = y_dist[y_dist[:, 1].argsort()]

            K_neighbours = y_dist.iloc[ : self.K , 0]
            
            # find unique values in a numpy array with frequency & indices
            (values,counts) = np.unique(K_neighbours.astype(int), return_counts = True)

            # majority vote, use the label with highest frequency as prediction
            y_pred = np.append(y_pred, values[np.argmax(counts)])
        return y_pred