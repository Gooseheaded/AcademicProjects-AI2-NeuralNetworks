from NeuralNetwork import NeuralNetwork
import sys

nn_name = "AND"
nn_structure = [2, 10, 1]

def init():
    "Initializes the neural network and the database."
    NeuralNetwork(nn_name, nn_structure, rewrite=True)
    return

def train():
    "Trains the neural network."
    nn = NeuralNetwork(nn_name, nn_structure)
    data = [([0., 0.], [0.]),
            ([1., 0.], [0.]),
            ([0., 1.], [0.]),
            ([1., 1.], [1.])]
    nn.train(data)
    return

def query(input_values):
    nn = NeuralNetwork(nn_name, nn_structure)
    print(nn.feedforward(input_values))
    return

def rtfm():
    print("Usage: {0} OPTION".format(sys.argv[0]))
    print("\nOptions:")
    print("\t--init")
    print("\t--train")
    print("\t--query [input ...]")
    print("\t--help")
    return

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        rtfm()
        return
    
    if sys.argv[1] == "--init":
        init()
    elif sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--query":
        query(sys.argv[2:])
    return

if __name__ == "__main__":
    main()

