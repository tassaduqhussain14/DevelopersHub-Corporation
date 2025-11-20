import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path_file = "Iris DatasSet\\Iris.csv"
iris_df = pd.read_csv(path_file)
print(iris_df)

print("Shape of the dataset: ")
print(iris_df.shape)

print("Columns of dataset: ")
print(iris_df.columns)

print("Info of the dataset: ")
print(iris_df.info())

print("Describtion of dataset: ")
print(iris_df.describe())

print("Check missing data: ")
print(iris_df.isna().sum())

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=iris_df,
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species",       
    style="Species",     
    s=50                
)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=iris_df,
    x="SepalWidthCm",
    y="PetalWidthCm",
    hue="Species",
    style="Species",
    s=100
)
plt.title("Sepal Width vs Petal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()


plt.figure(figsize=(7,5))
sns.histplot(
    data=iris_df, 
    x="SepalLengthCm", 
    bins=10, hue="Species", 
    multiple="stack")
plt.title("Histogram of Sepal Length")
plt.show()


plt.figure(figsize=(7,5))
sns.histplot(
    data=iris_df, 
    x="PetalLengthCm", 
    bins=10, 
    hue="Species", 
    multiple="stack")
plt.title("Histogram of Petal Length")
plt.show()

features = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]

for feature in features:
    plt.figure(figsize=(7,5))
    sns.boxplot(data=iris_df, x="Species", y=feature, palette=["red","green","blue"])
    plt.title(f"Box Plot of {feature} by Species")
    plt.show()
