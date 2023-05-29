import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, input_length, output_length, hidden_layer_nodes=[100]):
        '''
        input_length -- the number of elements in the input vector.
        output_length -- the number of elements in the output vector.
        hidden_layer_nodes -- a list of the number of nodes.
        '''
        super(FullyConnected, self).__init__()

        # Define the input layer
        self.input_layer = nn.Linear(input_length, hidden_layer_nodes[0])
        self.relu = nn.ReLU()

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer_nodes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer_nodes[i], hidden_layer_nodes[i+1]))

        # Define the output layer
        self.output_layer = nn.Linear(hidden_layer_nodes[-1], output_length)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        output = self.output_layer(x)
        return output

# Example usage
input_length = 10
output_length = 5
hidden_layer_nodes = [100, 50, 20]

model = FullyConnected(input_length, output_length, hidden_layer_nodes)
input_data = torch.randn(32, input_length)  # Example input data of shape (batch_size, input_length)
output = model(input_data)
print(output.shape)  # Print the shape of the output tensor
