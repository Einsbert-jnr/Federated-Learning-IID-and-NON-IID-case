import matplotlib.pyplot as plt
import pickle


NUM_CLIENTS = 3
with open('history3_234.pickle', 'rb') as f:
    history = pickle.load(f)
f.close()

print(history)

accuracy = history.metrics_distributed['agg_accuracy']
acc_values = [item[1] for item in accuracy]
loss = history.metrics_distributed['agg_loss']
loss_values = [item[1] for item in loss]

plt.plot(acc_values)
plt.legend(['Accuracy'], loc = 'lower right')
plt.ylabel('Accuracy')
plt.xlabel('Rounds')
plt.title(f"Accuracy curve: Federated learning with {NUM_CLIENTS} clients")
plt.show()