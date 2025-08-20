import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from model_ann import get_ann
from sklearn.utils import shuffle
from preprocessing import load_and_preprocess_data, FlowerClient

# load and preprocess data
env = "urban"
X_train, y_train, X_test, y_test, y_scaler = load_and_preprocess_data(env)


fl.client.start_client(server_address="localhost:8080",
                            client=FlowerClient(get_ann(), X_train, y_train, X_test, y_test, y_scaler).to_client())