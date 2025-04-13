import requests
import json

def get_json_details():
    response = requests.get(
        "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=US-CAL-CISO",
        headers={
            "auth-token": f"RPSsQWuEMXICCFEhgOXI"
        }
    )   
    response = response.json()
    a = json.loads(response)
    return a["carbonIntensity"]

def calculate_carbon_emissions(gpu_usage, duration):
    GPU_TDP_W = 300  # A100 max power in Watts
    usage_percent = gpu_usage / 100
    duration_hours = duration / 60
    carbon_intensity_g_per_kWh = get_json_details()
    power_draw_W = GPU_TDP_W * usage_percent
    energy_kWh = (power_draw_W * duration_hours) / 1000
    emissions_g = energy_kWh * carbon_intensity_g_per_kWh
    return emissions_g