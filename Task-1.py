from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['bath'] = train['BsmtFullBath']+0.5*train['BsmtHalfBath']+0.5*train['HalfBath']+train['FullBath']
test['bath'] = test['BsmtFullBath']+test['BsmtHalfBath']+test['HalfBath']+test['FullBath']

X = train[['bath','LotArea','BedroomAbvGr']]
y = train['SalePrice']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101,test_size=0.3)
scaler = StandardScaler()
scaler.fit_transform(X_train,y_train)
scaler.transform(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train,y_train)

pred = linear_model.predict(X_test)
sns.scatterplot(x = pred,y =y_test)
plt.show()



