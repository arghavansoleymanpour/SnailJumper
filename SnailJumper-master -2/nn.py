import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.neuron_number_1 = layer_sizes[0]
        self.neuron_number_2 = layer_sizes[1]
        self.neuron_number_3 = layer_sizes[2]
        self.w1 = np.random.normal(size=(self.neuron_number_2, self.neuron_number_1))
        self.w2 = np.random.normal(size=(self.neuron_number_3, self.neuron_number_2))
        self.b2 = np.zeros((self.neuron_number_2, 1))
        self.b3 = np.zeros((self.neuron_number_3, 1))


    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        x = x.astype(float)
        return 1/(1+np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        a1 = x
        a2 = self.activation(self.w1 @ a1 + self.b2)
        a3 = self.activation(self.w2 @ a2 + self.b3)
        return a3
        pass
