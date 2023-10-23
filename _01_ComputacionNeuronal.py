# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

import numpy as np

# Definir una neurona simple
def neurona(input_data, weights):
    return np.dot(input_data, weights)

# Datos de entrada y pesos
input_data = np.array([2, 3, 1])
weights = np.array([0.5, 0.8, -0.2])

# Calcular la salida de la neurona
output = neurona(input_data, weights)
print("Salida de la neurona:", output)
