import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


path_file = pd.read_csv("Task3\\HeartDiseaseTrain-Test.csv")
heart_df = pd.read_csv(path_file)
print(heart_df)

print("Info of the Dataset: ")
print(heart_df.info)

print("Describtion of the dataset: ")
print(heart_df.describe())

print("Columns name of the Dataset: ")
print(heart_df.columns)

print("Check the missing values in the Dataset: ")
print(heart_df.isna().sum())

