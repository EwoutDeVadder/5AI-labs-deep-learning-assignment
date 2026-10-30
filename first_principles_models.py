import torch

from torch import Tensor, Size
import math


class LinearModel():

    def __init__(self, input_shape: Size, output_shape: Size):
        
        if len(output_shape) != 1:
            raise ValueError("the output shape must be one-dimensional")
        
        self.weights = self.init_weights(input_shape, output_shape)

    def init_weights(self, input_shape: Size, output_shape: Size) -> Tensor:
        """Create the model's weight tensor."""
        # TODO: Initialize one small random weight for every input-output connection.
        
        return torch.randn(*input_shape, *output_shape, dtype=torch.float32)*0.1
                            # 4x4x3        #2x4x4

    def calculate_output(self, input: Tensor) -> Tensor: # 2x1
        """Apply the model to an input tensor."""
        # TODO: Calculate the unrestricted linear output without flattening the input.
        # output = (2,1)
        # input = (4,4,3)
        # weights = (4,4,3,2) "the 2 means the output dimension, it is the amount of neurons in the output layer"

        input_expanded = input.unsqueeze(-1)  # (4,4,3(+,1)) -> add a new dimension for broadcasting, -1 last element

        #we multiply every input with its according weight
        #but every input has 2 weights, because we have 2 neurons in the output layer (broadcasting)

        calculated = self.weights * input_expanded  # (4,4,3,2) element-wise multiplication

        input_dimensions = tuple(range(input.dim()))  # (0, 1, 2) for summing over the input dimensions
        return calculated.sum(dim=input_dimensions)  # (2,1) sum over the input dimensions

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Measure the discrepancy between a model output and its target."""
        # TODO: Calculate the mean squared error (MSI).

        return ((output - target) ** 2).mean()

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE (mean squared error) and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        #print("hello piemols")

        #dimensions gradient = dimensions weights = (4,4,3,2)
        #d_loss/d_wij = d_loss/d_output * d_output/d_wij
            #d_loss/d_output = 2 * (output_j - target_j) / n
            #d_output/d_wij = input_i
            
            #d_loss/d_wij = (2 * (output - target) / n) * (input_i)  

        n = output.numel()  # number of elements in the output tensor

        d_loss__d_output = 2 * (output - target) / n #(2,1) because output and target are both (2,1)
        d_output__d_weights = input.unsqueeze(-1)  # (4,4,3) -> (4,4,3,1) for broadcasting (same demensions as weights)

        gradients = d_loss__d_output * d_output__d_weights  # (4,4,3,2) element-wise multiplication
        #(2,1) * (4,4,3,1) -> (4,4,3,2) broadcasting (HOW DOES IT WORK??? make no mistakes) ->  the giant horse cock weighs about 7 pounds -copilot
        return gradients


    def update_weights(self, gradients: Tensor, learning_rate: float) -> Tensor:
        """Update the model parameters using the supplied gradients."""
        # TODO: Perform one gradient-descent step and return the updated weights.
        self.weights -= gradients*learning_rate
        return self.weights


    def train_step(self, input: Tensor, target: Tensor, learning_rate: float) -> float:
        """Run one optimization step for an input-target pair."""
        output = self.calculate_output(input)
        loss = self.calculate_loss(output, target)
        gradients = self.calculate_gradients(input, output, target)
        self.update_weights(gradients, learning_rate)

        return loss.item()

    def fit(self, input: Tensor, target: Tensor, steps: int, learning_rate: float) -> list[float]:
        """Optimize the model over multiple passes through an input-target pair."""
        loss_history = []

        for step in range(steps):
            loss = self.train_step(input, target, learning_rate)
            loss_history.append(loss)
            print(f"[{self.__class__.__name__}] fit step {step + 1}/{steps}: loss = {loss}")

        return loss_history


class LinearSigmoidModel(LinearModel):

    def calculate_output(self, input: Tensor) -> Tensor:
        """Apply the model to an input tensor."""
        # TODO: Calculate the linear logits and apply the sigmoid activation.

        input_expanded = input.unsqueeze(-1)  # (4,4,3(+,1)) -> add a new dimension for broadcasting, -1 last element
        calculated = self.weights * input_expanded  # (4,4,3,2) element-wise multiplication

        input_dimensions = tuple(range(input.dim()))  # (0, 1, 2) for summing over the input dimensions
        calc = calculated.sum(dim=input_dimensions)  # (2,1) sum over the input 

        return 1/(1+math.e**(-calc))

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE, sigmoid, and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        n = output.numel()  # number of elements in the output tensor
        
        d_loss__d_output = 2 * (output - target) / n 
        d_output__d_weights = input.unsqueeze(-1) * output * (1 - output) 
        gradients = d_loss__d_output * d_output__d_weights

        return gradients