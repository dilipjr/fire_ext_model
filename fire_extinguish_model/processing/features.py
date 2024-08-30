from typing import Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder



class LabelEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str):
        self.variable = variable
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X[self.variable])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.variable] = self.label_encoder.transform(X[self.variable])
        return X

    def inverse_transform(self, X, y=None):
        X = X.copy()
        X[self.variable] = self.label_encoder.inverse_transform(X[self.variable])
        return X

