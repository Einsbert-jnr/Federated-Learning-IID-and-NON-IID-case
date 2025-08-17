import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import flwr as fl
from sklearn.metrics import mean_squared_error, r2_score
from model_ann import get_ann


env = "urban"
def splittingDataset(dataset_path):
    # Load dataset
    data = pd.read_excel(dataset_path)
    print(data.head())
    data = shuffle(data).reset_index(drop=True)

    # Split dataset
    X = data.drop(columns=["OptimalPower_dBm", 'CommMode'])
    y = data['OptimalPower_dBm']
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pd.DataFrame(X_train).to_excel(f"X_train_{env}.xlsx", index=False)
    pd.DataFrame(X_test).to_excel(f"X_test_{env}.xlsx", index=False)
    pd.DataFrame(y_train).to_excel(f"y_train_{env}.xlsx", index=False)
    pd.DataFrame(y_test).to_excel(f"y_test_{env}.xlsx", index=False)


def preprocessing(X_train, y_train, X_test, y_test):
    # Normalize data
    scalar = StandardScaler()
    scalar.fit(X_train)
    X_train_scaled = scalar.transform(X_train)
    # y_train_scaled = scalar.transform(y_train) # scaling isn't needed
    # y_test_scaled = scalar.transform(y_test) # scalig isn't needed
    X_test_scaled = scalar.transform(X_test)

    print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")

    return X_train_scaled, y_train, X_test_scaled, y_test



def load_and_preprocess_data(env):
    # Load dataset
    X_train = pd.read_excel(f"./datasets/train/{env}/X_train_{env}.xlsx")
    y_train = pd.read_excel(f"./datasets/train/{env}/y_train_{env}.xlsx")
    X_test = pd.read_excel(f"./datasets/test/{env}/X_test_{env}.xlsx")
    y_test = pd.read_excel(f"./datasets/test/{env}/y_test_{env}.xlsx")

    # Preprocess data
    # X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = preprocessing(X_train, y_train, X_test, y_test)

    return preprocessing(X_train, y_train, X_test, y_test)



VERBOSE = 0
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = get_ann()
        # self.model.build(self, input_shape=(None, 43))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train the model with the provided parameters."""
        self.model.set_weights(parameters)
        
        self.model.fit(self.X_train, self.y_train, validation_split=0.2, epochs=1, batch_size=256, verbose=VERBOSE)

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=VERBOSE)

        # Predictions
        y_pred = self.model.predict(self.X_test, verbose=VERBOSE)

        # Extra regression metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)

        return loss, len(self.X_test), {
            "loss": loss,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }



if __name__ == "__main__":
    dataset_path = "./datasets/urban_data.xlsx"
    splittingDataset(dataset_path)