from nn.layer import  Dense
import numpy as np

# Example usage of Dense layer
dense_layer = Dense(units=10)
input_tensor = np.random.randn(5, 10)
output = dense_layer(input_tensor)
print("Output:", output)