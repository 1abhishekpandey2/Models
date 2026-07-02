import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0
        self.c = 0

    def loss_fn(self, m, c, data):
        E=0
        for i in range(len(data)):
            x=data.iloc[i,0]
            y=data.iloc[i,1]
            E+=(y-(m*x+c))**2
        return E/(2*len(data))
    
    def fit(self, data): 
        
        for i in range(self.epochs):
            
            
            self.m,self.c = self.gradient_descent(self.m, self.c, data, self.learning_rate)

            loss = self.loss_fn(self.m, self.c, data)
            print(f"Epoch {i}: Loss = {loss:.4f}")
        return self.m, self.c



    def gradient_descent(self, m_now, c_now, data, l):
        m_gradient=0
        c_gradient=0
        for j in range(len(data)):
            x=data.iloc[j,0]
            y=data.iloc[j,1]
            m_gradient+= -((y-m_now*x-c_now)*x)
            c_gradient+= -((y-m_now*x-c_now))

        m_gradient /= len(data)
        c_gradient /= len(data)

        return m_now - m_gradient*l, c_now - c_gradient*l

    def predict(self, x):
        return self.m*x + self.c
    

if __name__ == "__main__":
    data= pd.read_csv('linear.csv')

    model = LinearRegression(learning_rate=0.01, epochs=700)

    m, c = model.fit(data)

    plt.scatter(data.iloc[:,0],data.iloc[:,1])
    plt.plot(data.iloc[:,0],m*data.iloc[:,0]+c,color='red')

    predictions = [model.predict(x) for x in [12, 5, 100]]
    print(predictions)

