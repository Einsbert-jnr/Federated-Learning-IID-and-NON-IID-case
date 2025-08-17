import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from model_ann import get_ann
from sklearn.utils import shuffle
from preprocessing import load_and_preprocess_data, FlowerClient
from sklearn.metrics import mean_squared_error, r2_score

# load and preprocess data
env = "rural"
X_train, y_train, X_test, y_test = load_and_preprocess_data(env)


fl.client.start_client(server_address="localhost:8080", 
                        client=FlowerClient(get_ann(), X_train, y_train, X_test, y_test).to_client())
