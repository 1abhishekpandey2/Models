Linear_regression.py is a simple linear regression model
You may use it by importing it in your file directly as "import linear_regression as lr"
OR
As a module You may use "from Models import linear_regression as lr"

Example to Use
data= pd.read_csv('linear.csv')

model = lr.LinearRegression(learning_rate=0.01, epochs=700)

m, c = model.fit(data)

plt.scatter(data.iloc[:,0],data.iloc[:,1])
plt.plot(data.iloc[:,0],m*data.iloc[:,0]+c,color='red')

pred = model.predict(4) 
print(pred)
