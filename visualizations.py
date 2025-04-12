import numpy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Finding Missing Values
df = pd.read_csv("ai_ghibli_trend_dataset_v2.csv")
missing_values = df.isnull().sum()
print(missing_values)

#Visualizations
df['gpu_usage'].hist(bins=10)
plt.xlabel('gpu_usage')
plt.ylabel('Frequency')
plt.title(f'Histogram of GPU Usage')
plt.show() 

df['generation_time'].hist(bins=10)
plt.xlabel('generation_time')
plt.ylabel('Frequency')
plt.title("Histogram of Generation Time")
plt.show()

plt.boxplot(df['generation_time'])
plt.title("Generation Time Boxplot")
plt.show()

plt.boxplot(df['gpu_usage'])
plt.title("GPU_Usage Boxplot")
plt.show()