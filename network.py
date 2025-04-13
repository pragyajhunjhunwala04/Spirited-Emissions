import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

CSV_FILE = "ai_ghibli_trend_dataset_v2.csv"
GPU_MODEL = None
TIME_MODEL = None
SEN_SCORE = None

def extract_data():
    """
    Open the data file and extracts: 
    - prompt
    - gpu_usage
    - generation_time
    Return numpy array 
    """
    data = pd.read_csv(CSV_FILE, usecols=["prompt", "gpu_usage", "generation_time"])
    return data.values

def remove_ghibli_style(prompts: list):
    """
    Given a list of str prompts, remove duplications like:
    - ghibli
    - anime
    - style
    - inspired
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
    
    return modified_strings


def regressor_gpu_sen(X, Y):
    """
    Perform regression for gpu_usage
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
    learner = MLPRegressor(learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=(100,), batch_size=250)
    learner.fit(X_train, Y_train)
    print(f"GPU prediciton error: {learner.score(X_test, Y_test)}")

def regressor_gpu_len(X, Y):
    """
    Perform regression for gpu_usage
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
    learner = MLPRegressor(learning_rate_init=0.01, max_iter=500, hidden_layer_sizes=(100,), batch_size=250)
    learner.fit(X_train, Y_train)
    print(f"GPU prediction error: {learner.score(X_test, Y_test)}")

def regressor_time_sen(X, Y):
    """
    Perform regression for gpu_usage
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
    learner = MLPRegressor(learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=(100,), batch_size=250)
    learner.fit(X_train, Y_train)
    print(f"Time prediction error: {learner.score(X_test, Y_test)}")


def regressor_time_len(X, Y):
    """
    Perform regression for generation_time
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
    learner = MLPRegressor(learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=(50,), batch_size=250)
    learner.fit(X_train, Y_train)
    print(f"Time prediction error: {learner.score(X_test, Y_test)}")

def train_model():
    """
    For the calculator
    """
    # training
    raw_numpy_data = extract_data()
    raw_prompts = raw_numpy_data[:,0]
    prompts = remove_ghibli_style(raw_prompts.tolist())
    gen_time = raw_numpy_data[:,1]
    gpu_usage = raw_numpy_data[:,2]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    global SEN_SCORE, TIME_MODEL, GPU_MODEL
    SEN_SCORE = model
    X = model.encode(prompts)  
    # generating time
    time_learner = MLPRegressor(learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=(100,), batch_size=250)
    time_learner.fit(X, gen_time)
    TIME_MODEL = time_learner
    # gpu usage
    gpu_learner = MLPRegressor(learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=(100,), batch_size=250)
    gpu_learner.fit(X, gpu_usage)
    GPU_MODEL = gpu_learner


def calc_gpu(prompt: str):
    """
    For the calculator
    """
    if SEN_SCORE and GPU_MODEL:
        processed = remove_ghibli_style(prompt)
        prediction = GPU_MODEL.predict(SEN_SCORE.encode([processed]))
        return prediction[0]

def calc_time(prompt: str):
    """
    For the calculator
    """
    if SEN_SCORE and TIME_MODEL:
        processed = remove_ghibli_style(prompt)
        prediction = TIME_MODEL.predict(SEN_SCORE.encode([processed]))
        return prediction[0]

def testing():
    """
    For testing purposes only
    """
    raw_numpy_data = extract_data()
    raw_prompts = raw_numpy_data[:,0]
    prompts = remove_ghibli_style(raw_prompts.tolist())
    gen_time = raw_numpy_data[:,1]
    gpu_usage = raw_numpy_data[:,2]
    # input 1 : sentiment
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X1 = model.encode(prompts)  
    # gpu usage
    print("Sentiment analysis")
    regressor_gpu_sen(X1, gpu_usage)
    regressor_time_sen(X1, gen_time)
    # input 2: word count
    X2 = np.array([[len(prompt)] for prompt in prompts])
    print("Length analysis")
    regressor_gpu_len(X2, gpu_usage)
    regressor_time_len(X2, gen_time)

if __name__ == "__main__":
    # testing()
    train_model()
    test = "My friend gets abducted by aliens in Ghibli style"
    print(calc_gpu(test))
    print(calc_time(test))