import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

test_y = pd.read_csv("../data/test_y.csv", header=None).get_values()
ground_truth_longitude = test_y[:,0] * -1
ground_truth_latitude = test_y[:,1]
ground_truth_floor = test_y[:,2]
ground_truth_building = test_y[:,3]

knn_result = pd.read_csv("../result/FFN.csv",index_col = 0, header=0)
knn_longitude = knn_result.get_values()[:,0] * -1
knn_latitude = knn_result.get_values()[:,1]
knn_floor = knn_result.get_values()[:,2]
knn_building = knn_result.get_values()[:,3]

# knn_result = pd.read_csv("../result/final.csv", header=None)
# knn_floor = knn_result.get_values()[:,10]
# knn_building = knn_result.get_values()[:,11]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_bgcolor((0, 0, 0))

ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.5))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.5))
# ax.xaxis.label.set_color('white')
# ax.yaxis.label.set_color('white')
# ax.zaxis.label.set_color('white')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.tick_params(axis='z', colors='white')
xdata = []
ydata = []
zdata = []
for i in range(len(ground_truth_floor)):
	if ground_truth_floor[i] == knn_floor[i]:
		continue
	else:
		# ax.plot([ground_truth_longitude[i], ground_truth_longitude[i]], [ground_truth_latitude[i], ground_truth_latitude[i]], [ground_truth_floor[i], knn_floor[i]], 'r')
		# ax.plot([ground_truth_longitude[i], knn_longitude[i]], [ground_truth_latitude[i], knn_latitude[i]], [ground_truth_floor[i], knn_floor[i]], c=[0.27450980392156865, 0.5098039215686274, 0.782608695652174], linewidth=1, alpha=0.5)
		ax.plot([ground_truth_longitude[i], knn_longitude[i]], [ground_truth_latitude[i], knn_latitude[i]], [ground_truth_floor[i], knn_floor[i]], c=[0 ,0.5, 0.5],alpha=0.5, linewidth=2)
		
		# ax.plot([ground_truth_latitude[i], knn_latitude[i]], [ground_truth_longitude[i], knn_longitude[i]], [ground_truth_floor[i], knn_floor[i]], 'r')
		# xdata.append(ground_truth_longitude[i])
		# ydata.append(ground_truth_latitude[i])
		xdata.append(knn_longitude[i])
		ydata.append(knn_latitude[i])
		zdata.append(knn_floor[i])
ax.scatter3D(xdata, ydata, zdata, c='r')
ax.scatter3D(ground_truth_longitude,ground_truth_latitude, ground_truth_floor, c=[0.5, 0.5, 0.5], alpha=0.5, s=5)

# ax.scatter3D(xdata, ydata, zdata, c=[0.6980392156862745, 0.13333333333333333, 0.13333333333333333], alpha=0.5)
# ax.scatter3D(ground_truth_longitude,ground_truth_latitude, ground_truth_floor, c=[0.8549019607843137, 0.6470588235294118, 0.043137254901960784], alpha=0.5, s=5)
# ax.scatter3D(xdata, ydata, zdata, c='b', alpha=0.5)
# ax.scatter3D(ground_truth_longitude,ground_truth_latitude, ground_truth_floor, c=[10]*len(ground_truth_latitude), alpha=0.5, s=5)

# ax.scatter3D(ydata, xdata, zdata, c='g')
# ax.scatter3D(ground_truth_latitude,ground_truth_longitude, ground_truth_floor, c='b')
# ax.grid(False)

plt.show()
