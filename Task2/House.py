import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error

path_file = "Task2\\Housing.csv"
house_pd = pd.read_csv(path_file)

print("Head of Dataset: ")
print(house_pd.head())

print("Info of the Dataset: ")
print(house_pd.info)

print("Describtion of the dataset: ")
print(house_pd.describe())

print("Columns of the dataset: ")
print(house_pd.columns)

print("checking missing values in dataset: ")
print(house_pd.isna().sum())

count = house_pd["furnishingstatus"].value_counts()
print(count)

plt.figure(figsize=(8,5))
sns.barplot(x=count.index, y=count.values)
plt.title("Number of Houses by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Number of Houses")
plt.show()

plt.figure(figsize=(8,10))
sns.scatterplot(data=house_pd,x="area",y="price", hue="bedrooms")
plt.title("Price and Area relationship")
plt.legend(loc="upper left")
plt.show()

plt.figure(figsize=(8,10))
sns.boxplot(data=house_pd,x="furnishingstatus",y="price", hue="bedrooms")
plt.title("Price and Furnishing relationship")
plt.legend(loc="upper left")
plt.show()

plt.figure(figsize=(8,10))
sns.boxplot(data=house_pd,x="furnishingstatus",y="price", hue="stories")
plt.title("Price and Furnishing relationship")
plt.legend(loc="upper left")
plt.show()

X = house_pd.drop("price",axis=1)
y = house_pd["price"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print("Train size of X train and X test")
print(f"Train size: {round(len(X_train) / len(X) * 100)}% \n\
Test size: {round(len(X_test) / len(X) * 100)}%")

transformer = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(sparse_output=False, drop='first'),
     ["furnishingstatus","mainroad","guestroom","basement",
      "hotwaterheating","airconditioning","prefarea"]),
    ('scale', StandardScaler(),["area"])], remainder='passthrough')


model = Pipeline(steps=[
    ("transformer", transformer),
    ("model", GradientBoostingRegressor())
])

model.fit(X_train,y_train)
y_preds = model.predict(X_test)

result = pd.DataFrame({"Actual" : y_test, "Predicted" : y_preds})
result["Predicted"] = result["Predicted"].apply(lambda x: int(x))
print(f"Actual vs Predicted: \n{result}")


plt.figure(figsize=(10,6))
sns.scatterplot(x=result["Actual"], y=result["Predicted"])
plt.plot([result["Actual"].min(), result["Actual"].max()], 
         [result["Actual"].min(), result["Actual"].max()], 
         color='red', linestyle='--')  # Line y=x for perfect prediction
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

MAE = mean_absolute_error(y_test,y_preds)
MSE = mean_squared_error(y_test,y_preds)
RMSE = np.sqrt(MSE)

print(f'Mean absolute error: {MAE:.2f}')
print(f'Mean squared error: {MSE:.2f}')
print(f'Root mean squared error: {RMSE:.2f}')


actual_minus_predicted = sum((y_test - y_preds)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)     
