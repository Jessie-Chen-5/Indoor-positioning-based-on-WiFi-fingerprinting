import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import tree

def normalizeX(arr):
	res = np.copy(arr).astype(np.float)
	res[res==100] = 0
	res[res!=100] *= -0.01
	return res

longitude_scaler = MinMaxScaler()
latitude_scaler = MinMaxScaler()
def normalizeY(longitude_arr, latitude_arr):
	global longitude_scaler
	global latitude_scaler
	longitude_arr = np.reshape(longitude_arr, [-1, 1])
	latitude_arr = np.reshape(latitude_arr, [-1, 1])
	longitude_scaler.fit(longitude_arr)
	latitude_scaler.fit(latitude_arr)
	return np.reshape(longitude_scaler.transform(longitude_arr), [-1]), \
			np.reshape(latitude_scaler.transform(latitude_arr), [-1])

def reverse_normalizeY(longitude_arr, latitude_arr):
	global longitude_scaler
	global latitude_scaler
	longitude_arr = np.reshape(longitude_arr, [-1, 1])
	latitude_arr = np.reshape(latitude_arr, [-1, 1])
	return np.reshape(longitude_scaler.inverse_transform(longitude_arr), [-1]), \
			np.reshape(latitude_scaler.inverse_transform(latitude_arr), [-1])

def compute_error(test_y, predict_longitude, predict_latitude, predict_floor, predict_building):
	longitude_error = np.sum(np.sqrt(np.square(test_y[:,0]-predict_longitude)))/len(test_y)
	latitude_error = np.sum(np.sqrt(np.square(test_y[:,1]-predict_latitude)))/len(test_y)
	floor_error = (len(test_y) - np.sum(np.equal(test_y[:,2], predict_floor)))/len(test_y)
	building_error = (len(test_y) - np.sum(np.equal(test_y[:,3], predict_building)))/len(test_y)
	return longitude_error, latitude_error, floor_error, building_error

train_x = pd.read_csv("../data/train_x.csv", header=None).get_values()
train_y = pd.read_csv("../data/train_y.csv", header=None).get_values()
test_x = pd.read_csv("../data/test_x.csv", header=None).get_values()
test_y = pd.read_csv("../data/test_y.csv", header=None).get_values()

class DecisionTree():
	def __init__(self):
		self.longitude_regressor = tree.DecisionTreeRegressor()
		self.latitude_regressor = tree.DecisionTreeRegressor()
		self.floor_classifier = tree.DecisionTreeClassifier()
		self.building_classifier = tree.DecisionTreeClassifier()
	def train(self, train_x, train_y):
		train_x = normalizeX(train_x)
		longitude,latitude = normalizeY(train_y[:,0],train_y[:,1])
		self.longitude_regressor.fit(train_x, longitude)
		self.latitude_regressor.fit(train_x, latitude)
		self.floor_classifier.fit(train_x, train_y[:,2])
		self.building_classifier.fit(train_x, train_y[:,3])
	def predict(self, test_x):
		test_x = normalizeX(test_x)
		predict_longitude = self.longitude_regressor.predict(test_x)
		predict_latitude = self.latitude_regressor.predict(test_x)
		predict_longitude, predict_latitude = reverse_normalizeY(predict_longitude, predict_latitude)
		predict_floor = self.floor_classifier.predict(test_x)
		predict_building = self.building_classifier.predict(test_x)
		return predict_longitude, predict_latitude, predict_floor, predict_building

t = DecisionTree()
t.train(train_x, train_y)
predict_longitude, predict_latitude, predict_floor, predict_building = t.predict(test_x)

print(compute_error(test_y, predict_longitude, predict_latitude, predict_floor, predict_building))

result = pd.DataFrame([predict_longitude, predict_latitude, predict_floor, predict_building]).transpose()
result.to_csv("../result/DT.csv")