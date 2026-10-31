import os
import math
import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


class Learner():

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def save(self, label):
        if not os.path.exists(os.path.join("checkpoints")):
            os.makedirs(os.path.join("checkpoints"))
        torch.save(self.model.state_dict(), os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label))

    def load(self, label):
        self.model.load_state_dict(torch.load(os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label), weights_only=True))
    

class RegressionLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []

        for epoch in range(epochs):

            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in dataloader:
                data = data.to(device)
                targets = targets.to(device)

                # TODO: Implement the batch update step.
                #we moeten gwn dezelfde stappen als oef 1 uitvoeren, mr dan door pytorch

                self.optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)  # Compute the loss
                
                loss.backward()  # Backpropagate the gradients
                self.optimizer.step()  # Update the model parameters

                train_loss += loss.item() * len(data)
            
            train_loss = train_loss / len(dataloader.dataset)

            train_losses.append(train_loss)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.6f}")   

            if train_loss <= min(train_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return train_losses

    @torch.no_grad()
    def test(self, dataset: Dataset, device: torch.device):

        self.model.to(device)
        self.model.eval()  # set model in evaluation mode

        #TODO: done
        data, targets = dataset[:]
        data, targets = data.to(device), targets.to(device)

        outputs = self.model(data)
        mse = torch.mean((outputs - targets) ** 2)

        return math.sqrt(mse.item())  # RMSE

    @torch.no_grad()
    def predict(self, dataset, device):
        self.model.to(device)
        self.model.eval()
        data = dataset[:][0]
        data = data.to(device)
        results = self.model(data)
        return results


class ClassificationLearner(Learner):

    #Exercise 8: Train and evaluate the classification model
    # Complete the ClassificationLearner class in learners.py.

    # You will need to complete the train method with the update step for each batch using the configured model, criterion, and optimizer.

    # Furthermore, you will need to complete the _evaluate method. This method should evaluate every batch without tracking gradients and return the mean loss and accuracy over the complete dataset.
    # Select the predicted class for each image from the model output.

    # The learner uses _evaluate to measure performance on the validation subset after each training epoch and saves the model with the lowest validation loss. The test method reuses the same evaluation procedure for the test set.

    # Note: Make sure that both loss and accuracy are aggregated over the number of samples rather than the number of batches.

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(epochs):
            
            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in train_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                
                # TODO: Implement the batch update step.

                self.optimizer.zero_grad()                  # Zero the gradients
                outputs = self.model(data)                  # Forward pass through the model (64,10) output
                loss = self.criterion(outputs, targets)     # Compute the loss
                loss.backward()                             # Backpropagate the gradients
                self.optimizer.step()                       # Update the model parameters

                # loss = torch.tensor([0]) IMAGINE ANOTHER LINE OF CODE HERE, BUT IT'S NOT USEFUL!!!
                # id like to make myself believe that planet earth turns slowely, it's hard to say but i'd rather stay awake when im asleep

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_dataloader.dataset)

            self.model.eval()  # set model in evaluation mode

            # evaluate the model on the validation dataset
            valid_loss, valid_accuracy = self._evaluate(valid_dataloader, device)    

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.4f}\tvalid loss = {valid_loss:.4f}\tvalid accuracy = {100 * valid_accuracy:.2f}%")   

            if valid_loss <= min(valid_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return (train_losses, valid_losses, valid_accuracies)

    def test(self, test_dataloader: DataLoader, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        # evaluate the model on the test dataset
        test_loss, test_accuracy = self._evaluate(test_dataloader, device)

        return (test_loss, test_accuracy)

    @torch.no_grad()
    def _evaluate(self, dataloader, device):
        eval_loss = 0
        eval_accuracy = 0

        # TODO: Compute the model's loss and accuracy over the dataset.
        #welcome to the tab-tab
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = self.model(data)  # Forward pass through the model
            loss = self.criterion(outputs, targets)  # Compute the loss

            eval_loss += loss.item() * len(data)  # Accumulate the loss

            # Compute accuracy
            predicted = torch.argmax(outputs, dim=1)  # Get the predicted class
            eval_accuracy += (predicted == targets).sum().item()  # Count correct predictions

        # avr loss and accuracy
        eval_loss = eval_loss / len(dataloader.dataset)
        eval_accuracy = eval_accuracy / len(dataloader.dataset)

        return (eval_loss, eval_accuracy)
