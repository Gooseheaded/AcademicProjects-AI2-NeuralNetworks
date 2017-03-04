from math import exp
import random
import sqlite3


class NeuralNetwork():

    def __init__(self, layer_sizes):
        self.db = sqlite3.connect('NeuralNetwork.db')
        self.layers = layer_sizes  # Ignore the input layer.
        cursor = self.db.cursor()

        # Clear the database.
        cursor.execute('''DELETE FROM neurons;''')
        cursor.execute('''DELETE FROM biases;''')
        cursor.execute('''DELETE FROM weights;''')

        # Layers.
        layer = 0
        for size in layer_sizes:
            layer += 1
            for _ in range(size):
                cursor.execute('''
                    INSERT INTO neurons (layer, net_input, output, error)
                    VALUES (?, 0, 0, 0)
                ;''', (str(layer),))

        # Biases.
        # The first biases are set to 0; input layer doesn't need them.
        cursor.execute('''
            INSERT INTO biases (value)
            VALUES (0)
        ;''')
        for layer in range(1, len(layer_sizes)):
            cursor.execute('''
                INSERT INTO biases (value)
                VALUES (?)
            ;''', (str(random.random()),))

        # Weights.
        for layer in range(1, len(layer_sizes) + 1):
            from_ids = [x[0] for x in cursor.execute('''
                SELECT id
                FROM neurons
                WHERE layer = ?
            ''', (str(layer)),)]

            to_ids = [x[0] for x in cursor.execute('''
                SELECT id
                FROM neurons
                WHERE layer = ?
            ''', (str(layer + 1)),)]

            for from_id in from_ids:
                for to_id in to_ids:
                    # print("[{}]({}) => [{}]({})".format(
                        # str(layer), str(from_id), str(layer + 1),
                        # str(to_id)))
                    cursor.execute('''
                        INSERT INTO weights
                        VALUES(?, ?, ?, ?, ?)
                    ''', (str(layer), str(from_id), str(layer + 1), str(to_id), str(random.random())))

        self.db.commit()

    def __set_input_values(self, input_values):
        cursor = self.db.cursor()

        for idx in range(len(input_values)):
            cursor.execute('''
                UPDATE neurons
                SET output = ?
                WHERE layer = 1
                AND id = ?
            ;''', (str(input_values[idx]), str(idx + 1)))

        self.db.commit()

    def feedforward(self, input_values, ideal_values=[]):
        print("input_values", input_values)
        self.__set_input_values(input_values)
        cursor = self.db.cursor()

        biases = [x[0] for x in cursor.execute('''
            SELECT value
            FROM biases
        ;''')]

        for to_layer in range(2, len(self.layers) + 1):
            from_layer = to_layer - 1

            # Get all ids and outputs of the previous layer.
            from_neurons = [x for x in cursor.execute('''
                SELECT id, output
                FROM neurons
                WHERE layer = ?
            ;''', (str(from_layer)))]
            from_ids = [x[0] for x in from_neurons]
            from_outputs = [x[1] for x in from_neurons]
            print("from_outputs", from_outputs)

            # Get all ids of the next layer.
            to_ids = [x[0] for x in cursor.execute('''
                SELECT id
                FROM neurons
                WHERE layer = ?
            ;''', (str(to_layer)))]

            # For every id of the next layer, calculate the net_input, taking
            # the corresponding outputs, biases, and weights into account.
            weights = []
            for idx in to_ids:
                weights = [x[0] for x in cursor.execute('''
                    SELECT value
                    FROM weights
                    WHERE from_layer = ?
                    AND to_id = ?
                ;''', (str(from_layer), str(idx)))]
                print("weights", weights)

                net_input = biases[to_layer - 1]
                print("net_input", net_input)
                for i in range(len(weights)):  # Same len as len(from_outputs)
                    print("from", from_outputs[i], "weight", weights[i])
                    net_input += from_outputs[i] * weights[i]

                output = 1 / (1 + exp(- net_input))
                print("output", output)

                cursor.execute('''
                    UPDATE neurons
                    SET net_input = ?, output = ?
                    WHERE id = ?
                ;''', (str(net_input), str(output), str(idx)))

        self.db.commit()

    def setup_example(self):
        cursor = self.db.cursor()
        cursor.execute('''
            UPDATE weights
            SET value = 0.15
            WHERE from_id = 1 AND to_id = 3
        ;''')
        cursor.execute('''
            UPDATE weights
            SET value = 0.20
            WHERE from_id = 2 AND to_id = 3
        ;''')
        cursor.execute('''
            UPDATE weights
            SET value = 0.25
            WHERE from_id = 1 AND to_id = 4
        ;''')
        cursor.execute('''
            UPDATE weights
            SET value = 0.30
            WHERE from_id = 2 AND to_id = 4
        ;''')
        cursor.execute('''
            UPDATE biases
            SET value = 0.35
            WHERE layer = 2
        ;''')
        #
        cursor.execute('''
            UPDATE weights
            SET value = 0.40
            WHERE from_id = 3 AND to_id = 5
        ;''')
        cursor.execute('''
            UPDATE weights
            SET value = 0.45
            WHERE from_id = 4 AND to_id = 5
        ;''')
        cursor.execute('''
            UPDATE weights
            SET value = 0.50
            WHERE from_id = 3 AND to_id = 6
        ;''')
        cursor.execute('''
            UPDATE weights
            SET value = 0.55
            WHERE from_id = 4 AND to_id = 6
        ;''')
        cursor.execute('''
            UPDATE biases
            SET value = 0.60
            WHERE layer = 3
        ;''')
        self.db.commit()

nn = NeuralNetwork([2, 2, 2])
nn.setup_example()
nn.feedforward([0.05, 0.10])
# nn.feedforward([round(random.random()) for _ in range(2)])
