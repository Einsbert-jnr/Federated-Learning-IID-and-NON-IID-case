import flwr as fl
from sklearn.metrics import classification_report
from model_ann import get_ann
import pickle


VERBOSE = 0
NUM_CLIENTS = 3

def weighted_average(metrics):
    """Aggregate regression metrics from clients"""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    
    # Aggregate regression metrics
    agg_loss = sum(num_samples * m.get("loss", 0) for num_samples, m in metrics) / total_samples
    agg_mae = sum(num_samples * m.get("mae", 0) for num_samples, m in metrics) / total_samples
    agg_mse = sum(num_samples * m.get("mse", 0) for num_samples, m in metrics) / total_samples
    agg_rmse = sum(num_samples * m.get("rmse", 0) for num_samples, m in metrics) / total_samples
    agg_r2 = sum(num_samples * m.get("r2", 0) for num_samples, m in metrics) / total_samples
    
    # Print individual client metrics
    print(f"\n{'='*50}")
    print(f"FEDERATED ROUND METRICS SUMMARY")
    print(f"{'='*50}")
    
    for i, (num_samples, m) in enumerate(metrics):
        print(f"Client {i+1} (samples: {num_samples}):")
        print(f"  Loss: {m.get('loss', 0):.5f}")
        print(f"  MAE: {m.get('mae', 0):.5f}")
        print(f"  MSE: {m.get('mse', 0):.5f}")
        print(f"  RMSE: {m.get('rmse', 0):.5f}")
        print(f"  R²: {m.get('r2', 0):.5f}")
        print()
    
    print(f"AGGREGATED METRICS:")
    print(f"  Weighted Loss: {agg_loss:.5f}")
    print(f"  Weighted MAE: {agg_mae:.5f}")
    print(f"  Weighted MSE: {agg_mse:.5f}")
    print(f"  Weighted RMSE: {agg_rmse:.5f}")
    print(f"  Weighted R²: {agg_r2:.5f}")
    print(f"{'='*50}\n")

    return {
        "agg_loss": agg_loss, 
        "agg_mae": agg_mae,
        "agg_mse": agg_mse,
        "agg_rmse": agg_rmse,
        "agg_r2": agg_r2
    }


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

with open('results/history.pickle', 'wb') as f:
    pickle.dump(history, f)