from math import exp
import random


class Neuron:

    "Feeds forward a value based on the incoming neurons' values."

    def __init__(self):
        self.incoming = []  # Array of tuples, (neuron, weight)
        self.bias = (0)     # Bias is a tuple (so it's passed by reference)
        self.value = None

    def feed_forward(self):
        if self.value != None:
            return self.value

        net_input = self.bias[0]
        # Verbosity.
        # print("bias", self.bias[0])
        for neuron in self.incoming:
            single_input = neuron[0].feed_forward()
            net_input += single_input * neuron[1]
            # Verbosity.
            # print(single_input, "*", neuron[1], "=", single_input * neuron[1])
        # Verbosity.
        # print("net_input", net_input)
        self.value = 1. / (1 + exp(-(net_input)))  # Sigmoid fn.
        # Verbosity.
        # print("value", self.value)
        return self.value

    def connect(self, neuron, weight):
        self.incoming.append((neuron, weight))

    def clear():
        for neuron in self.incoming:
            neuron.clear()
        self.value = None


class NeuralNetwork:

    def __init__(self, layers, biases=None):
        self.layers = layers

        if biases == None:
            self.biases = [(random.random(),) for _ in range(len(layers))]
        else:
            self.biases = biases

        # TODO(gctrindade): Validate layers?

        # Assign layer biases to layer neurons.
        for layer_idx in range(1, len(self.layers)):
            for neuron in self.layers[layer_idx]:
                neuron.bias = self.biases[layer_idx]

        # Connect neurons from the previous layer into the next layer.
        for layer_idx in range(1, len(self.layers)):
            next_layer = self.layers[layer_idx]
            prev_layer = self.layers[layer_idx - 1]

            for next_node in next_layer:
                for prev_node in prev_layer:
                    next_node.connect(prev_node, random.random())

    def feed_forward(self, inputs):
        # TODO(gctrindade): Memoize values.
        # TODO(gctrindade): Validate that len(inputs) == len(layers[0])
        if len(inputs) != len(self.layers[0]):
            print("Error: expected", len(self.layers[0]), "inputs, but received", len(
                inputs), " (", inputs, ").")
            return

        for input_idx in range(len(self.layers[0])):
            input_neuron = self.layers[0][input_idx]
            input_neuron.value = inputs[input_idx]

        results = []
        for neuron in self.layers[-1]:
            results.append(neuron.feed_forward())
        return results

    def get_error(self, hyp):
        pass


##########

def squared_error(output, ideal):
    if len(output) != len(ideal):
        print("Error: output values and ideal values have different lengths (", len(
            output), ",", len(ideal))
        return

    error = .0
    for idx in range(len(output)):
        error += pow(ideal[idx] - output[idx], 2) / 2
    return error


def output_layer_weight_correction(output, ideal, incoming):
    "The output layer backpropagation is different because it does need to account for outgoing values."
    # Derivative of total error with respect to output.
    deriv_error_to_out = (output - ideal)
    # Derivative of output with respect to net input.
    deriv_out_to_net_in = (output * (1 - output))
    # Derivative of net input with respect to weight.
    deriv_net_in_to_w = incoming

    # Verbosity.
    # print(deriv_error_to_out, deriv_out_to_net_in, deriv_net_in_to_w)

    return deriv_error_to_out * deriv_out_to_net_in * deriv_net_in_to_w


def hidden_layer_weight_correction():
    pass


def simple_test():
    "Generates a simple 2-4-1 network, with input [1,1]"

    # Initialize the neuron layers.
    layers = [[], [], []]

    input_layer = layers[0]  # Initialize input layer.
    for _ in range(2):
        input_layer.append(Neuron())

    hidden_layer = layers[1]  # Initialize hidden layer.
    for _ in range(2):
        hidden_layer.append(Neuron())

    output_layer = layers[2]  # Initialize output layer.
    for _ in range(2):
        output_layer.append(Neuron())

    # Build a network out of the layers, with the default configurations.
    # We will override them immediately after.
    network = NeuralNetwork(layers)

    # Override biases.
    network.biases = [(.35,), (.60,)]
    for neuron in hidden_layer:
        neuron.bias = network.biases[0]
    for neuron in output_layer:
        neuron.bias = network.biases[1]

    # Override weights.
    # Weights cannot be assigned, they must be rebuilts, because tuples are
    # immutable.
    hidden_layer[0].incoming[0] = (input_layer[0], .15)
    hidden_layer[0].incoming[1] = (input_layer[1], .20)

    hidden_layer[1].incoming[0] = (input_layer[0], .25)
    hidden_layer[1].incoming[1] = (input_layer[1], .30)

    output_layer[0].incoming[0] = (hidden_layer[0], .40)
    output_layer[0].incoming[1] = (hidden_layer[1], .45)

    output_layer[1].incoming[0] = (hidden_layer[0], .50)
    output_layer[1].incoming[1] = (hidden_layer[1], .55)

    # Evaluate the network that was built.
    input_values = [.05, .1]
    output_values = network.feed_forward(input_values)
    ideal_values = [.01, .99]
    print("Input:", input_values)
    print("Output:", output_values)
    print("Ideal:", ideal_values)

    # Calculate squared error, and begin backpropagation.
    # The corrected weights will only be applied later, so we simply store them for
    # now.
    corrected_weights = {}
    learning_rate = 0.5
    print("Squared error:", squared_error(output_values, ideal_values))

    # Calculate output layer weight corrections.
    for idx in range(len(output_layer)):
        for incoming in output_layer[idx].incoming:
            neuron = output_layer[idx]
            correction = output_layer_weight_correction(
                output_values[idx], ideal_values[idx], incoming[0].value)
            # Verbosity.
            print("corrected ", incoming[1] - correction * learning_rate)
            corrected_weights[incoming[0]] = incoming[
                1] - (correction * learning_rate)

    # Calculate hidden layers' weight corrections.
    for layer in layers:
        for neuron in layer:
            for incoming in neuron.incoming:
                # TODO(gctrindade): :3
                pass


##########

if __name__ == "__main__":
    simple_test()
