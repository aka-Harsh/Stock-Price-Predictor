import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self):
        self.model= RandomForestRegressor(n_estimators=100)

    def fit(self, X_train,y_train):
        X_train_2d = X_train.reshape(X_train.shape[0],-1)
        self.model.fit(X_train_2d, y_train)
    
    def fit(self,X_test):
        X_test_2d=X_test.reshape(X_test.shape[0],-1)
        predictions=[]
        for i in range(X_test_2d.shape[0]):
            pred=self.model.predict(X_test_2d[i:i+1])
            predictions.append(pred[0])
            if i < X_test_2d.shape[0] - 1:
                X_test_2d[i+1, :-5] = X_test_2d[i, 5:]
                X_test_2d[i+1, -5:] = pred
        return np.array(predictions).reshape(-1, 1)
