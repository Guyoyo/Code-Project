#Import useful libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#Read data with numpy

data = np.genfromtxt("inputdata7.csv", delimiter=",", dtype=str)


#slice the array to get rainfall column only and convert to float datatype

x = np.asarray(data[1:,0], dtype='float64')


#slice the array to get productivity column only and convert to float datatype

y = np.asarray(data[1:, 1], dtype='float64')


#Plot rainfall vs productivity

fig, ax = plt.subplots(figsize=(16, 12), dpi=200)

plt.title("Amount of Rainfall vs Productivity", fontsize = 20)
plt.xlabel("Rainfall", fontsize = 16)
plt.ylabel("Productivity", fontsize = 16)
plt.scatter(x, y, color ="blue", s =16)

plt.show()


#Reshape x and y to 2-D array

x_,y_=x.reshape(-1,1), y.reshape(-1,1)


#Create simple linear regression model

model = LinearRegression()
model.fit(x_, y_)


#Generate line of best fit using the linear model

y_pred = model.predict(x_)



#plot line of best fit on the scatter plot

fig, ax = plt.subplots(figsize=(16, 12), dpi=200)

plt.title("Amount of Rainfall vs Productivity", fontsize = 20)
plt.xlabel("Rainfall", fontsize = 16)
plt.ylabel("Productivity", fontsize = 16)
plt.scatter(x, y, color ="blue", s=14)
plt.plot(x, y_pred, color ="black", label ="Regression Line")
plt.legend(loc = 'upper right')
plt.show()


#Predict productivity using the model
pred_value =310

pred = model.predict([[pred_value]])

print(pred)


#plot the predicted value on the chart

fig, ax = plt.subplots(figsize=(16, 12), dpi=200)

plt.title("Amount of Rainfall vs Productivity", fontsize = 20)
plt.xlabel("Rainfall", fontsize = 16)
plt.ylabel("Productivity", fontsize = 16)
plt.scatter(x, y, color ="blue", s =12)
plt.plot(x, y_pred, color ="black", label ="Regression Line")
plt.scatter(pred_value, pred, color ="red",s=100, label ="Prediction")

#Annotate the predicted point on the chart
plt.annotate(round(float(pred),3), (pred_value + 8, pred), fontsize=15)

plt.legend(loc = 'upper right')
plt.show()






