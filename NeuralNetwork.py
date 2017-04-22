from sys import stdout
from math import exp, floor
import random
import sqlite3

class NeuralNetwork():

    def __init__(self, name, layer_sizes, rewrite=False):
        self.db = sqlite3.connect(name + '.db')
        self.layers = layer_sizes  # Ignore the input layer.
        cursor = self.db.cursor()

        if not rewrite:
            return

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
                    INSERT INTO neurons (layer, net_input, output, delta)
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
                        VALUES(?, ?, ?, ?, ?, 0)
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

        # self.db.commit()

    def __calculate_deltas(self, ideal_values):
        # TODO(gctrindade): Include biases as well?
        cursor = self.db.cursor()
        output_layer = len(self.layers)

        # First, calculate deltas for the output layer.
        output_neurons = [x for x in cursor.execute('''
            SELECT id, output
            FROM neurons
            WHERE layer = ?
        ;''', (str(output_layer),))]

        for idx in range(len(output_neurons)):
            # [idx][0] is neuron id.
            # [idx][1] is neuron output.

            deriv1 = output_neurons[idx][1] * (1 - output_neurons[idx][1])
            deriv2 = output_neurons[idx][1] - ideal_values[idx]

            cursor.execute('''
                UPDATE neurons
                SET delta = ?
                WHERE id = ?
            ''', (str(deriv1 * deriv2), str(output_neurons[idx][0])))

        # Then, calculate deltas for every other layer, backwards, ignoring the
        # input and output layers.
        for layer in range(len(self.layers) - 1, 1, -1):
            hidden_neurons = [x for x in cursor.execute('''
                SELECT id, output
                FROM neurons
                WHERE layer = ?
            ;''', (str(layer),))]

            prev_deltas = [x[0] for x in cursor.execute('''
                SELECT delta
                FROM neurons
                WHERE layer = ?
            ;''', (str(layer + 1),))]
            # prev_deltas[0] is delta

            for neuron in hidden_neurons:
                # neuron[0] is id
                # neuron[1] is output

                weights = [x[0] for x in cursor.execute('''
                    SELECT value
                    FROM weights
                    WHERE from_id = ?
                    AND to_layer = ?
                ;''', (str(neuron[0]), str(layer + 1)))]

                # Sanity check.
                if len(weights) != len(prev_deltas):
                    print("deltas) Sanity check failed!")
                    print("\tweights", weights)
                    print("\tdeltas", prev_deltas)
                    return

                deriv1 = neuron[1] * (1 - neuron[1])
                deriv2 = 0
                for idx in range(len(weights)):
                    deriv2 += prev_deltas[idx] * weights[idx]

                cursor.execute('''
                    UPDATE neurons
                    SET delta = ?
                    WHERE id = ?
                ;''', (str(deriv1 * deriv2), str(neuron[0])))

        # self.db.commit()

    def __calculate_gradients(self):
        cursor = self.db.cursor()

        for layer in range(2, len(self.layers) + 1):
            outputs = [x for x in cursor.execute('''
                SELECT id, output
                FROM neurons
                WHERE layer= ?
            ;''', (str(layer - 1),))]

            deltas = [x for x in cursor.execute('''
                SELECT id, delta
                FROM neurons
                WHERE layer = ?
            ;''', (str(layer),))]

            for idx in range(len(deltas)):
                for idx2 in range(len(outputs)):
                    cursor.execute('''
                        UPDATE weights
                        SET gradient = ?
                        WHERE from_id = ?
                        AND to_id = ?
                    ;''', (str(deltas[idx][1] * outputs[idx2][1]), str(outputs[idx2][0]), str(deltas[idx][0])))

        # self.db.commit()

    def __correct_weights(self, learning_rate=1):
        cursor = self.db.cursor()

        cursor.execute('''
            UPDATE weights
            SET value = value - (gradient * ?)
        ;''', (str(learning_rate),))

        # self.db.commit()

    def feedforward(self, input_values):
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
            # print("from_outputs", from_outputs)

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
                # print("weights", weights)

                net_input = biases[to_layer - 1]
                # print("net_input", net_input)
                for i in range(len(weights)):  # Same len as len(from_outputs)
                    # print("from", from_outputs[i], "weight", weights[i])
                    net_input += from_outputs[i] * weights[i]

                output = 1 / (1 + exp(- net_input))
                # print("output", output)

                cursor.execute('''
                    UPDATE neurons
                    SET net_input = ?, output = ?
                    WHERE id = ?
                ;''', (str(net_input), str(output), str(idx)))

        outputs = [x[0] for x in cursor.execute('''
            SELECT output
            FROM neurons
            WHERE layer = ?
        ;''', (len(self.layers),))]

        # self.db.commit()

        return outputs

    def get_total_error(self, ideal_values):
        cursor = self.db.cursor()

        output_values = [x[0] for x in cursor.execute('''
            SELECT output
            FROM neurons
            WHERE layer = ?
        ;''', (len(self.layers),))]

        total_error = 0
        for idx in range(len(output_values)):
            total_error += ((output_values[idx] -
                             ideal_values[idx]) ** 2) * 0.5

        return total_error

    def backpropagate(self, input_values, ideal_values, learning_rate=1):
        self.feedforward(input_values)
        self.__calculate_deltas(ideal_values)
        self.__calculate_gradients()
        self.__correct_weights(learning_rate)

        # self.db.commit()

    def clear_and_save(self):
        cursor = self.db.cursor()
        cursor.execute('''
            UPDATE neurons
            SET output = 0, net_input = 0, delta = 0
        ;''')

        cursor.execute('''
            UPDATE weights
            SET gradient = 0
        ;''')

        print("\nSaving db...")

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

    def train(self, training_data, learning_rate=1, max_error=0.01):
        print("Training network with rate set to {0}, until error is less than {1} .".format(learning_rate, max_error))
        print("Press Ctrl-C to stop.")
        try:
            saved_db = False
            error = 1
            while error > max_error:
                for data in training_data:
                    self.backpropagate(data[0], data[1], 1)

                error = 0
                for data in training_data:
                    self.feedforward(data[0])
                    error += self.get_total_error(data[1])

                print("Error:", error, end='\r')

        except KeyboardInterrupt:
            self.clear_and_save()
            saved_db = True
        finally:
            if not saved_db:
                self.clear_and_save()
