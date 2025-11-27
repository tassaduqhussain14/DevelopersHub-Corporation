import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


path_file = "Task3\\HeartDiseaseTrain-Test.csv"
heart_df = pd.read_csv(path_file)
print(heart_df)

print("Info of the Dataset: ")
print(heart_df.info())

print("Describtion of the dataset: ")
print(heart_df.describe())

print("Columns name of the Dataset: ")
print(heart_df.columns)

print("Check the missing values in the Dataset: ")
print(heart_df.isna().sum())

sns.histplot(
    data=heart_df,
    x="age",
    hue="sex",
    multiple="dodge",
    bins=10,   
)
plt.title("Gender by Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,10))
sns.scatterplot(
    data=heart_df,
    x="resting_blood_pressure",
    y="cholestoral",
    hue="chest_pain_type",
    style="target",
    size="exercise_induced_angina",
)
plt.title("Blood Pressure vs Cholesterol by Chest Pain Type")
plt.legend(loc="upper right")
plt.show()

X = heart_df.drop("target",axis=1)
y = heart_df["target"]

print(X)
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print("Train size of X train and X test")
print(f"Train size: {round(len(X_train) / len(X) * 100)}% \n\
Test size: {round(len(X_test) / len(X) * 100)}%")


cat_columns = ["sex","chest_pain_type","fasting_blood_sugar","rest_ecg","exercise_induced_angina","slope","vessels_colored_by_flourosopy","thalassemia"]

for col in cat_columns:
    print(heart_df[col].value_counts())  # Count of each unique value
    print("-"*30)

scaled_cols = ["age","resting_blood_pressure","cholestoral","Max_heart_rate"]
# for col in scaled_cols:
#     print(heart_df[col].value_counts())  # Count of each unique value
#     print("-"*30)

transfer = ColumnTransformer(transformers=[
    ("ohe",OneHotEncoder(sparse_output=False,drop='first'),cat_columns),
    ('scaling',StandardScaler(),scaled_cols)],remainder="passthrough")

model = Pipeline(steps=[
    ("transformer", transfer),
    ("model", LogisticRegression())
])

model.fit(X_train,y_train)
y_preds = model.predict(X_test)

from sklearn.metrics import roc_curve,accuracy_score,confusion_matrix

print(roc_curve(y_test,y_preds))
print(accuracy_score(y_test,y_preds))
print(confusion_matrix(y_test,y_preds))

result = pd.DataFrame({"Actual" : y_test, "Predicted" : y_preds})
print(f"Actual vs Predicted: \n{result}")

#Plot of confusing Matrix

cm = confusion_matrix(y_test, y_preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
