import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sentence_transformers import SentenceTransformer
import re

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
    words_to_remove = ["ghibli", "anime", "inspired", "style", "studio"]
    
    modified_strings = []
    
    for string in prompts:
        modified_string = string
        
        for word in words_to_remove:
            pattern = re.compile(rf'\b{word}\b|-{word}|{word}-', re.IGNORECASE)
            modified_string = pattern.sub('', modified_string)
        
        modified_string = re.sub(r'\s+', ' ', modified_string)
        modified_string = re.sub(r'\s*-\s*', '-', modified_string)
        modified_string = re.sub(r'-+', '-', modified_string)
        modified_string = modified_string.strip('- ')
        
        modified_strings.append(modified_string)
    
    # print(modified_strings)
    return modified_strings


def main():
    raw_numpy_array = extract_prompt_gpu()
    prompts_raw = raw_numpy_array[:,0]
    gpu_usage = raw_numpy_array[:,1].astype(float)
    prompts = remove_ghibli_style(prompts_raw.tolist())
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(prompts)
    X_train, X_test, Y_train, Y_test = train_test_split(X, gpu_usage, test_size=0.1, shuffle=True)
    learner = MLPRegressor(learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=100, batch_size=250)
    learner.fit(X_train, Y_train)
    error = learner.score(X_test, Y_test)
    print(error)
        

if __name__ == "__main__":

    main()
