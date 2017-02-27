from math import exp


class Neuron:

    def __init__(self):
        self.incoming = []  # Array of tuples, (neuron, weight)

    def calculate(self):
        sum = 0
        for neuron in incoming:
            sum += neuron[0].calculate() * neuron[1]
        return 1 / (1 + exp(-sum))

##########


def simple_test():
    return

if __name__ == "__main__":
    firstFunctionEver()
