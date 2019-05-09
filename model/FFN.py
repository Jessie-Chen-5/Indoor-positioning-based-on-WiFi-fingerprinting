import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils  import to_categorical

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
val_x = pd.read_csv("../data/val_x.csv", header=None).get_values()
val_y = pd.read_csv("../data/val_y.csv", header=None).get_values()

class FFN():
	def __init__(self):
		self.longitude_regressor = Sequential()
		self.longitude_regressor.add(Dense(128, input_shape = (520,), activation = 'relu'))
		self.longitude_regressor.add(Dense(1))
		self.longitude_regressor.compile(loss='mse', optimizer='RMSprop', metrics=['mse'])
		
		self.latitude_regressor = Sequential()
		self.latitude_regressor.add(Dense(128, input_shape = (520,), activation = 'relu'))
		self.latitude_regressor.add(Dense(1))
		self.latitude_regressor.compile(loss='mse', optimizer='RMSprop', metrics=['mse'])
		
		self.floor_classifier = Sequential()
		self.floor_classifier.add(Dense(128, input_shape = (520,), activation = 'relu'))
		self.floor_classifier.add(Dense(5, activation = 'softmax'))
		self.floor_classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		self.building_classifier = Sequential()
		self.building_classifier.add(Dense(128, input_shape = (520,), activation = 'relu'))
		self.building_classifier.add(Dense(3, activation = 'softmax'))
		self.building_classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	def train(self, train_x, train_y, val_x, val_y):
		longitude_history = self.longitude_regressor.fit(train_x, train_y[:,0], epochs=50, verbose = 1, batch_size=100,validation_data=(val_x, val_y[:,0]))
		latitude_history = self.latitude_regressor.fit(train_x, train_y[:,1], epochs=50, verbose = 1, batch_size=100,validation_data=(val_x, val_y[:,1]))
		floor_history = self.floor_classifier.fit(train_x, to_categorical(train_y[:,2]), epochs=50, verbose = 1, batch_size=100,validation_data=(val_x, to_categorical(val_y[:,2])))
		building_history = self.building_classifier.fit(train_x, to_categorical(train_y[:,3]), epochs=50, verbose = 1, batch_size=100,validation_data=(val_x, to_categorical(val_y[:,3])))
        
	def predict(self, test_x):
		predict_longitude = self.longitude_regressor.predict(test_x)
		predict_latitude = self.latitude_regressor.predict(test_x)
		predict_floor = np.argmax(self.floor_classifier.predict(test_x), axis=1)
		predict_building = np.argmax(self.building_classifier.predict(test_x), axis=1)
		return predict_longitude, predict_latitude, predict_floor, predict_building

model = FFN()

train_x = normalizeX(train_x)
val_x = normalizeX(val_x)
test_x = normalizeX(test_x)
train_y[:,0], train_y[:,1] = normalizeY(train_y[:,0],train_y[:,1])
val_y[:,0] = np.reshape(longitude_scaler.transform(val_y[:,0]), [-1])
val_y[:,1] = np.reshape(latitude_scaler.transform(val_y[:,1]), [-1])
model.train(train_x, train_y, val_x, val_y)
predict_longitude, predict_latitude, predict_floor, predict_building = model.predict(test_x)
predict_longitude, predict_latitude = reverse_normalizeY(predict_longitude, predict_latitude)

print(compute_error(test_y, predict_longitude, predict_latitude, predict_floor, predict_building))

result = pd.DataFrame([predict_longitude, predict_latitude, predict_floor, predict_building]).transpose()
result.to_csv("../result/FFN.csv")