import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import regex

CSV_FILE = "ai_ghibli_trend_dataset_v2.csv"

def extract_prompt_gpu():
    """
    Opens the ai ghibli trend dataset and extract columns "prompt"
    and "gpu_usage"
    """
    dataframe = pd.read_csv(CSV_FILE, usecols=["prompt", "gpu_usage"])
    # print(dataframe.values)
    return dataframe.values


def remove_ghibli_style(prompts: list):
    """
    Given a list of prompts, extract duplication like "Ghibli" or "Anime-style"
    """
    # PRAGYA DO THIS!!!!!!


def main():
    raw_numpy_array = extract_prompt_gpu()
    prompts_raw = raw_numpy_array[:,0]
    gpu_usage = raw_numpy_array[:,1]
    prompts = remove_ghibli_style(prompts_raw.tolist())

if __name__ == "__main__":
    main()
