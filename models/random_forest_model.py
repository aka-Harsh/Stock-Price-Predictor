import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self):
        self.model= RandomForestRegressor(n_estimators=100)

    def fit(self, X_train,y_train):
        X_train_2d = X_train.reshape(X_train.shape[0],-1)
        self.model.fit(X_train_2d, y_train)
