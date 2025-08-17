import flwr as fl
from sklearn.metrics import classification_report
from model_ann import get_ann
import pickle


VERBOSE = 0
NUM_CLIENTS = 4
def weighted_average(metrics):
    total_samples = sum(num_samples for num_samples, _ in metrics)
    agg_loss = sum(num_samples * m.get("loss",0) for num_samples, m in metrics) / total_samples
    agg_accuracy = sum(num_samples * m.get("accuracy",1) for num_samples, m in metrics) / total_samples

    # Collect classification reports
    classification_reports = [m.get("classification_report", 2) for _, m in metrics]

    # Print each classification report with client number
    for i, report in enumerate(classification_reports):
        print(f"Classification report for client {i+1}:\n{report}\n")

    return {"agg_loss": agg_loss, "agg_accuracy": agg_accuracy}


from typing import Dict, List, Tuple

def get_evaluate_fn(X_test, y_test):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_ann()  # Construct the model
        # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(X_test, y_test, verbose=VERBOSE)
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)
        class_report = classification_report(y_test, y_pred, digits=5)

        return loss, {"loss":loss, "accuracy": accuracy, "Centralised report": class_report}

    return evaluate
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than the number of clients for training
    min_evaluate_clients=NUM_CLIENTS,  # Never sample less than the number of clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all the number of clients are available
    evaluate_metrics_aggregation_fn=weighted_average, # aggregates federated metrics
    # evaluate_fn=get_evaluate_fn(X_test_centralised_scaled, y_test_centralised),## global evaluation function
)

history = fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
    )

with open('history3.pickle', 'wb') as f:
    pickle.dump(history, f)