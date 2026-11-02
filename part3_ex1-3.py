import torch
import torchvision
import matplotlib.pyplot as plt

from models import NeuralNetworkClassificationModel
from learners import ClassificationLearner
from utilities import plot_training_process

## PARAMETERS
random_seed = 1
batch_size_train = 64
batch_size_test = 1000
sweep_epochs = 10                      # shorter runs to compare learning rates
final_epochs = 50
learning_rates = [0.001, 0.01, 0.1]    # 3 candidates, adjust after looking at the results
target_accuracy = 0.9735

## INITIALISATION
def set_seeds():
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

set_seeds()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

## DATA
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# seed is set right before the split, so the split is identical in every run
set_seeds()
train_subset, valid_subset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size_train, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size_test, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


## HELPER
def train_model(lr, epochs):
    """Train a fresh model; same initial weights and shuffling for every learning rate."""
    set_seeds()
    model = NeuralNetworkClassificationModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    learner = ClassificationLearner(model, criterion, optimizer)
    train_losses, valid_losses, valid_accuracies = learner.train(
        train_dataloader, valid_dataloader, epochs, device)
    return learner, train_losses, valid_losses, valid_accuracies


## EXERCISE 2a: learning rate sweep
print("=== Learning rate sweep ===")
sweep = {}
for lr in learning_rates:
    _, tl, vl, va = train_model(lr, sweep_epochs)
    sweep[lr] = (tl, vl, va)
    print(f"lr = {lr}: final train loss = {tl[-1]:.4f}, "
          f"final valid loss = {vl[-1]:.4f}, best valid accuracy = {100 * max(va):.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for lr, (tl, vl, va) in sweep.items():
    ax1.plot(vl, label=f"lr = {lr}")
    ax2.plot([100 * a for a in va], label=f"lr = {lr}")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation loss"); ax1.legend()
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation accuracy (%)"); ax2.legend()
fig.suptitle(f"Learning rate sweep (batch size {batch_size_train})")

# pick the learning rate with the lowest final validation loss (override if you disagree)
best_lr = min(sweep, key=lambda lr: sweep[lr][1][-1])
print(f"\nSelected learning rate: {best_lr}\n")


## EXERCISE 2b: final training for 50 epochs
print("=== Final training ===")
learner, train_losses, valid_losses, valid_accuracies = train_model(best_lr, final_epochs)
plot_training_process(train_losses, valid_losses)

min_valid_loss = min(valid_losses)
idx = valid_losses.index(min_valid_loss)
print(f"best valid loss = {min_valid_loss:.4f} (epoch {idx + 1}), "
      f"valid accuracy there = {100 * valid_accuracies[idx]:.2f}%")
print(f"highest valid accuracy = {100 * max(valid_accuracies):.2f}% "
      f"(epoch {valid_accuracies.index(max(valid_accuracies)) + 1})")


## EXERCISE 3: epochs needed for >= 97.35% validation accuracy
print("\n=== Epochs to reach target accuracy ===")
reached = next((i + 1 for i, a in enumerate(valid_accuracies) if a >= target_accuracy), None)
if reached is None:
    print(f"{100 * target_accuracy:.2f}% was not reached in {final_epochs} epochs")
else:
    print(f"{100 * target_accuracy:.2f}% first reached after {reached} epochs "
          f"(accuracy = {100 * valid_accuracies[reached - 1]:.2f}%)")
    print(f"train loss at that epoch = {train_losses[reached - 1]:.4f}, "
          f"valid loss = {valid_losses[reached - 1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, final_epochs + 1), [100 * a for a in valid_accuracies])
plt.axhline(100 * target_accuracy, color="red", linestyle="--", label=f"{100 * target_accuracy:.2f}%")
if reached is not None:
    plt.axvline(reached, color="gray", linestyle=":", label=f"epoch {reached}")
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy (%)")
plt.title("Validation accuracy during training")
plt.legend()
plt.show()