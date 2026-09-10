import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-------------------------------------
# Stpe 1 : Load the data 
#-------------------------------------

df = pd.read_csv("california_housing.csv")
print("Shape of Dataset : ",df.shape)
print("First few records : ",df.head())

#-------------------------------------
# Step 2 : Separate Fetures and Labels
#-------------------------------------

X = df.drop("target",axis=1)
Y = df["target"]

print("Shape of X : ",X.shape)
print("Shape of Y : ",Y.shape)

#-------------------------------------
# Step 3 : Split dataset for traiing and testing
#-------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

#-------------------------------------
# Step 4 : Create the model
#-------------------------------------

model = DecisionTreeRegressor(random_state=42)

#-------------------------------------
# Step 5 : Train the model
#-------------------------------------

model = model.fit(X_train,Y_train)

#-------------------------------------
# Step 6 : Test the model
#-------------------------------------

Y_pred = model.predict(X_test)

#-------------------------------------
# Step 7 : Evaluate the model
#-------------------------------------

print("MSE : ",mean_squared_error(Y_test, Y_pred))
print("R2 : ",r2_score(Y_test, Y_pred))