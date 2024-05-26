import pandas
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.liner_model import LinearRegression
from sklearn.metrics import mean_squared_error

boston = load_boston()
print(boston.DESCR)

df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target
x = df.RM.to_frame()
y = df.MEDV
print(x)
print(y)

x_train, y_train, x_test, y_test = train_test_split(x,y,test_size=0.3)

lr = LinearRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
print(y_pred[0]*1000)

mse = mean_squared_error(y_test, y_pred)
print(mse)
