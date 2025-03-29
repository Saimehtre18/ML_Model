#Import Data Manipulation Libraries
import pandas as pd
import numpy as np

# Loading Dataset 
url="https://raw.githubusercontent.com/Saimehtre18/ML_Model/refs/heads/main/Concrete_Data%20(1).csv"
df = pd.read_csv(url)

#Fit the Dataset into features (X) and targets (Y)
X=df.drop(columns='Concrete compressive strength(MPa, megapascals) ',axis=1)
y=df['Concrete compressive strength(MPa, megapascals) ']
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Model Building
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
RF=RandomForestRegressor()
RF.fit(X_train,y_train)
y_pred_RF=RF.predict(X_test)
r2_score_RF=r2_score(y_test,y_pred_RF)
print(f"The Model accuracy is: {r2_score_RF*100}%")