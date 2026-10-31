import torch

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        # TODO: Define the linear regression layer.
        self.linear_layer = torch.nn.Linear(in_feature, out_feature, bias=True, device=None, dtype=None)

    def forward(self, x: torch.Tensor):
        # TODO: Implement the forward pass.
        return self.linear_layer(x)


class NeuralNetworkClassificationModel(torch.nn.Module):


    #Complete the NeuralNetworkClassificationModel class in models.py.

    #First, define a fully connected architecture with three non-linear layers in the constructor. Take into account the dimensionality of the inputs and the
    #number of classes for which the model must produce an output. Determine suitable sizes for the two hidden layers and select appropriate activation functions.

    #Then, implement the forward method so that it returns a prediction for each image in a batch, using the three layers. Here is also the place to implement any final data transformations, such as flattening the images.
    # For this step, You might want to consult the documentation on the built-in modules provided by the torch.nn library.

    #Important: Ensure the model preserves the batch dimension.

    #Hint: When choosing the activation function for the final layer, consider how the criterion (loss function) will use the model's output values. Consult the loss function's documentation to determine the form of input it expects,
    #and make sure that this matches what the model returns.


    # MNIST:
    # • Image Size: 28 × 28 pixels (normalized and anti-aliased from the original 128 × 128 binary NIST files)
    # • Pixel Values: 0 to 255 (where 0 is black and 255 is white; often scaled to 0–1 for neural networks)
    # • Classes: 10 distinct digits (0 through 9)

    # Activation	Formula idea	                                                                        Verdict for this task
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Sigmoid	    squashes to (0, 1)	                                                                    Works, but gradients shrink for large inputs, so training is slower. Mostly used for binary outputs now.
    # Tanh	        squashes to (-1, 1)	                                                                    Better than sigmoid (centered at 0), but has the same flattening problem. Fine for a 3-layer net, just slower.
    # ReLU	        max(0, x): negatives become 0, positives pass through	                                The default choice. Cheap, gradient is exactly 1 for positive inputs, and reliably reaches about 97-98% on MNIST.
    #                                                                                                       Its one weakness is “dying ReLU”, which is rarely a problem in a small network.
    # LeakyReLU	    like ReLU, but negatives are scaled by a small factor (e.g. 0.01x) instead of zeroed	Avoids “dying ReLU”. Basically a drop-in replacement.
    # GELU / SiLU	smooth versions of ReLU	Common in transformers.                                         Slightly more expensive, with no real benefit on MNIST.

    def __init__(self):
        super().__init__()
        # TODO: Define the three classification layers.

        input_dim = 28 * 28 # 784
        num_classes = 10 # digits 0-9

        hidden_layer_1:int = 392 # 28 * 28 / 2 -> optionally / 3 etc.
        hidden_layer_2:int = 196 # hidden_layer_1 / 2
        
        self.layer1 = torch.nn.Linear(input_dim, hidden_layer_1)
        self.layer2 = torch.nn.Linear(hidden_layer_1, hidden_layer_2)
        self.layer3 = torch.nn.Linear(hidden_layer_2, num_classes)

    def forward(self, x: torch.Tensor):
        # TODO: Implement the forward pass.
        # Flatten the input images
        # (batch_size, channels, height, width) = x.shape - (batch_size, 1, 28, 28)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch_size, 784)
        x = torch.relu(self.layer1(x))  # (batch_size, hidden_layer_1)
        x = torch.relu(self.layer2(x)) # (batch_size, hidden_layer_2)
        x = self.layer3(x) # (batch_size, num_classes) end with linear function
        return x
