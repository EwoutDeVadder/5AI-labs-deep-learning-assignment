import torch
import matplotlib.pyplot as plt

from data import HousesDataset
from models import LinearRegressionModel
from learners import RegressionLearner
from utilities import plot_house_results, plot_training_process

## PARAMETERS
random_seed = 1
epochs = 25
default_train_size = 5000
test_dataset_size = 20
learning_rates = [0.01, 0.1, 0.25]   # 3 learning rates, adjust after looking at the results
batch_sizes = [32, 64]
other_train_sizes = [1000, 2000, 10000]

# set to None to pick the best combination from exercise 1 automatically,
# or fill in your own choice, e.g. (0.1, 32)
manual_choice = None

## INITIALISATION
def set_seeds():
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

set_seeds()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device:', device)

test_dataset = HousesDataset(test_dataset_size)
train_dataset = HousesDataset(default_train_size)


## HELPERS
def train_model(dataset, lr, batch_size):
    """Train a fresh model with the same initial weights every time."""
    set_seeds()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    model = LinearRegressionModel(1, 1)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    learner = RegressionLearner(model, criterion, optimizer)
    losses = learner.train(dataloader, epochs, device)
    return model, learner, losses


def evaluate_and_plot(model, learner, dataset, title_suffix):
    """Evaluate on the train and test datasets and visualise both."""
    learner.load("leader")  # best version of the parameters, as in the original script

    train_loss = learner.test(dataset, device)
    train_results = learner.predict(dataset, device).cpu()
    test_loss = learner.test(test_dataset, device)
    test_results = learner.predict(test_dataset, device).cpu()

    print(f"[{title_suffix}] train: root mean squared error = {train_loss:.2f} €")
    print(f"[{title_suffix}] test:  root mean squared error = {test_loss:.2f} €")

    # the "unknown" linear function f(x)
    linear_sizes = torch.linspace(dataset.min_house_size, dataset.max_house_size, 1000)
    linear_prices = 5000 * linear_sizes + 100000
    linear_dataset = (linear_sizes.unsqueeze(1), linear_prices.unsqueeze(1))
    linear_results = learner.predict(linear_dataset, device).cpu()

    plot_house_results(dataset.data, dataset.targets, train_results,
                       linear_sizes, linear_prices, linear_results,
                       title=f"Train dataset ({title_suffix})")
    plot_house_results(test_dataset.data, test_dataset.targets, test_results,
                       linear_sizes, linear_prices, linear_results,
                       title=f"Test dataset ({title_suffix})")

    print(f"[{title_suffix}] learned function: g(x) = "
          f"{model.linear_layer.weight.data[0].item():.2f} * x + "
          f"{model.linear_layer.bias.data[0].item():.2f}")
    print()
    return train_loss, test_loss


## EXERCISE 1: learning rate and batch size sweep
print("=== Exercise 1 ===")
results = {}
for batch_size in batch_sizes:
    for lr in learning_rates:
        _, _, losses = train_model(train_dataset, lr, batch_size)
        results[(lr, batch_size)] = losses
        print(f"lr = {lr}, batch size = {batch_size}: final train loss = {losses[-1]:.2f}")

plt.figure(figsize=(9, 6))
for (lr, bs), losses in results.items():
    plt.plot(losses, label=f"lr = {lr}, batch size = {bs}",
             linestyle="-" if bs == 32 else "--")
plt.xlabel("Epoch")
plt.ylabel("Training loss (L1)")
plt.title("Exercise 1: training loss for different learning rates and batch sizes")
plt.legend()

best_lr, best_bs = manual_choice or min(results, key=lambda k: results[k][-1])
print(f"\nSelected combination: lr = {best_lr}, batch size = {best_bs}\n")


## EXERCISE 2: selected combination on the default dataset
print("=== Exercise 2 ===")
model, learner, losses = train_model(train_dataset, best_lr, best_bs)
plot_training_process(losses, [])
evaluate_and_plot(model, learner, train_dataset, f"{default_train_size} samples")


## EXERCISE 3: different training dataset sizes

print("=== Exercise 3 ===")
summary = {}
for size in other_train_sizes:
    set_seeds()
    dataset = HousesDataset(size)
    model, learner, losses = train_model(dataset, best_lr, best_bs)
    plot_training_process(losses, [])
    summary[size] = evaluate_and_plot(model, learner, dataset, f"{size} samples")

print("Summary (train error, test error):")
for size, (tr, te) in summary.items():
    print(f"  {size:>6} samples: train = {tr:.2f} €, test = {te:.2f} €")

plt.show()
