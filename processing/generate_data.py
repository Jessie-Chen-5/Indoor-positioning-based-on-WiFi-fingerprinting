import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filename):
	data = pd.read_csv(filename)
	return data.get_values().T[:520].T, data.get_values().T[[520, 521, 522, 523], :].T

train_csv_path = '../UJIndoorLoc/trainingData.csv'
validation_csv_path = '../UJIndoorLoc/validationData.csv'

X, y = load_data(train_csv_path)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.15)
val_x, val_y = load_data(validation_csv_path)

pd.DataFrame(train_x).to_csv("../data/train_x.csv", header=None, index=None)
pd.DataFrame(train_y).to_csv("../data/train_y.csv", header=None, index=None)
pd.DataFrame(test_x).to_csv("../data/test_x.csv", header=None, index=None)
pd.DataFrame(test_y).to_csv("../data/test_y.csv", header=None, index=None)
pd.DataFrame(val_x).to_csv("../data/val_x.csv", header=None, index=None)
pd.DataFrame(val_y).to_csv("../data/val_y.csv", header=None, index=None)