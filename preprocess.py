from data_utils import Data
import numpy as np

def get_features(train=0):
    data = Data()
    train_data = data.get_data(train)
    for x_raw, y in train_data:
        

