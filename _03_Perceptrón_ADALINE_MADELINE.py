# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

# Implementación simple del perceptrón
class Perceptron:
    def __init__(self, num_inputs):
        self.weights = [0] * num_inputs
        self.bias = 0

    def predict(self, input_data):
        activation = sum(input_data[i] * self.weights[i] for i in range(len(input_data)) + self.bias)
        return 1 if activation > 0 else 0

# Implementación simple de ADALINE
class Adaline:
    def __init__(self, num_inputs):
        self.weights = [0] * num_inputs
        self.bias = 0
        self.learning_rate = 0.1

    def predict(self, input_data):
        activation = sum(input_data[i] * self.weights[i] for i in range(len(input_data)) + self.bias)
        return 1 if activation > 0 else 0

# Implementación simple de MADALINE
class Madaline:
    def __init__(self, num_inputs):
        self.weights = [0] * num_inputs
        self.bias = 0
        self.learning_rate = 0.1

    def predict(self, input_data):
        activation = sum(input_data[i] * self.weights[i] for i in range(len(input_data)) + self.bias)
        return 1 if activation > 0 else 0
