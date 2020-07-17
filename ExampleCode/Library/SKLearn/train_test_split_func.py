import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = np.reshape(np.random.randn(20),(10,2)) # 10 training examples
labels = np.random.randint(2, size=10) # 10 labels


