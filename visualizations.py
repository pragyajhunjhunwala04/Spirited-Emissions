import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Finding Missing Values
df = pd.read_csv("ai_ghibli_trend_dataset_v2.csv")
df = pd.DataFrame(df)
# missing_values = df.isnull().sum()
# print(missing_values)

#Visualizations
# df['gpu_usage'].hist(bins=10)
# plt.xlabel('gpu_usage')
# plt.ylabel('Frequency')
# plt.title(f'Histogram of GPU Usage')
# plt.show() 

# df['generation_time'].hist(bins=10)
# plt.xlabel('generation_time')
# plt.ylabel('Frequency')
# plt.title("Histogram of Generation Time")
# plt.show()

# plt.boxplot(df['generation_time'])
# plt.title("Generation Time Boxplot")
# plt.show()

# plt.boxplot(df['gpu_usage'])
# plt.title("GPU_Usage Boxplot")
# plt.show()

# grouped_values = [df[df['platform'] == cat]['gpu_usage'] for cat in df['platform'].unique()]
# labels = df['platform'].unique()
# # Plot stacked histogram
# plt.hist(grouped_values, bins=10, stacked=True, label=labels, edgecolor = 'black')
# plt.legend()
# plt.xlabel('GPU Usage * 100')
# plt.ylabel('Frequency')
# plt.title('Stacked Histogram of GPU Usage by Platform')
# plt.show()

# plt.scatter(df['gpu_usage'], df['generation_time'])
# plt.xlabel("Generation Time * 100")
# plt.ylabel("gpu_usage (percentage)")
# plt.title("Generation Time by GPU_Usage")
# plt.show()

# df['y_bin'] = pd.cut(df['generation_time'], bins=np.arange(0, 110, 10))
# df['x_bin'] = pd.cut(df['gpu_usage'], bins=np.arange(0, 110, 10))
# heatmap_data = df.groupby(['y_bin', 'x_bin']).size().unstack(fill_value=0)
# sns.heatmap(heatmap_data, fmt="d")
# plt.title("Frequency Heatmap")
# plt.xlabel("gpu_usage")
# plt.ylabel("generation time")
# plt.show()

# plt.scatter(df['file_size_kb'], df['style_accuracy_score'])
# plt.xlabel("GPU_Usage (percentage)")
# plt.ylabel("Accuracy")
# plt.title("GPU Usage by Accuracy of Image")
# plt.show()