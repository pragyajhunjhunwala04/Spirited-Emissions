import requests
import json
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import visualizations as v
import matplotlib.pyplot as plt

def get_json_details():
    response = requests.get(
        "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=US-CAL-CISO",
        headers={
            "auth-token": f"RPSsQWuEMXICCFEhgOXI"
        }
    )   
    response = response.json()
    return response.get("carbonIntensity")

def calculate_carbon_emissions(gpu_usage, duration):
    GPU_TDP_W = 300  # A100 max power in Watts
    usage_percent = gpu_usage / 100
    duration_hours = duration / 60
    carbon_intensity_g_per_kWh = get_json_details()
    power_draw_W = GPU_TDP_W * usage_percent
    energy_kWh = (power_draw_W * duration_hours) / 1000
    emissions_g = energy_kWh * carbon_intensity_g_per_kWh # returns kWh
    return emissions_g

# Adding a new column to the dataframe
df = pd.DataFrame(v.df)
df.loc[:, 'carbon_emissions'] = calculate_carbon_emissions(df['gpu_usage'], df['generation_time'])

freq = df.groupby(['gpu_usage', 'carbon_emissions']).size().reset_index(name='count')
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    freq['gpu_usage'],
    freq['carbon_emissions'],
    s=freq['count'] * 100,  # size scaled by count
    c=freq['count'],        # color mapped to count
    cmap='viridis',
    alpha=0.6,
    edgecolors='k'
)

plt.colorbar(scatter, label='Frequency')
plt.xlabel('gpu_usage')
plt.ylabel('carbon_emissions')
plt.title('GPU_Usage by Carbon Emissions')
plt.grid(True)
plt.show()

# plt.scatter(df['gpu_usage'], df['carbon_emissions'])
# plt.xlabel("GPU_usage")
# plt.ylabel("Carbon Emissions")
# plt.title("GPU Usage by Carbon Emissions")
# plt.show()


formula = 'carbon_emissions ~ gpu_usage + generation_time'
model = smf.glm(formula=formula, data=df, family=sm.families.Gaussian()) # Gaussian family for linear regression
result = model.fit()
print(result.summary())
