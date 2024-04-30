import numpy as np
import random

class Perceptron:
    def __init__(self):
        self.weights = [random.random(), random.random()]
        self.bias = random.random()
        self.learning_rate = 0.1
        
    def predict(self, inputs):
        sum = np.dot(inputs, self.weights) + self.bias
        return 1 if sum > 0 else 0

    def train(self, x_inputs, y_outputs, epochs):
        for _ in range(epochs):
            for inputs, outputs in zip(x_inputs, y_outputs):
                error = outputs - self.predict(inputs)
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

x_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_outputs = np.array([0, 0, 0, 1])

perceptron = Perceptron()
perceptron.train(x_inputs, y_outputs, 10)

print(f'===========================')
print(f'🚀 minimum_ai_perceptron 🚀')
print(f'===========================')

for data in np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1], [1, 0]]):
	prediction = perceptron.predict(data)
	print(f'📖input: {data}, 🤔prediction: {prediction}')

print(f'========================')
print(f'🎊🥳🎉 Congrats!  🎊🥳🎉')
print(f'========================')
