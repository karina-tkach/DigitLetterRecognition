import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, layer_sizes, learn_rate=0.15, epochs=100, batch_size=254):
        self.layer_sizes = layer_sizes
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights, self.biases = self.initialize_network()

    def initialize_network(self):
        weights = [np.random.uniform(-0.5, 0.5, (self.layer_sizes[i], self.layer_sizes[i - 1])) for i in range(1, len(self.layer_sizes))]
        biases = [np.zeros((self.layer_sizes[i], 1)) for i in range(1, len(self.layer_sizes))]
        return weights, biases

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    def forward_propagation(self, x):
        activations = [x]  # Store outputs of the neurons at each layer
        for w, b in zip(self.weights, self.biases):
            z = w @ activations[-1] + b
            a = self.sigmoid(z)
            activations.append(a)
        return activations[-1], activations

    def mini_batch_gradient_descent(self, images, labels):
        num_samples = images.shape[0]
        for epoch in range(self.epochs):
            permutation = np.random.permutation(num_samples)
            images, labels = images[permutation], labels[permutation]

            nr_correct = 0
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_images = images[start_idx:end_idx].T
                batch_labels = labels[start_idx:end_idx].T

                # Forward pass
                predictions, activations = self.forward_propagation(batch_images)

                # Backward pass
                for i in range(batch_labels.shape[1]):  # Loop over the batch dimension
                    nr_correct += int(np.argmax(predictions[:, i]) == np.argmax(batch_labels[:, i]))

                deltas = [predictions - batch_labels]
                for layer in range(len(self.weights) - 1, 0, -1):
                    delta = (self.weights[layer].T @ deltas[-1]) * self.sigmoid_derivative(activations[layer])
                    deltas.append(delta)
                deltas.reverse()

                # Update weights and biases
                for layer in range(len(self.weights)):
                    grad_w = deltas[layer] @ activations[layer].T / self.batch_size
                    grad_b = np.sum(deltas[layer], axis=1, keepdims=True) / self.batch_size
                    self.weights[layer] -= self.learn_rate * grad_w
                    self.biases[layer] -= self.learn_rate * grad_b

            accuracy = (nr_correct / num_samples) * 100  # Total correct / Total samples
            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy:.2f}%")

    def save_network_state(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases, 'learn_rate': self.learn_rate, 'epochs': self.epochs}, f)

    @staticmethod
    def load_network_state(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        return state['weights'], state['biases'], state['learn_rate'], state['epochs']

    @staticmethod
    def class_to_char(class_index):
        if class_index < 10:
            return str(class_index)  # Digits 0–9
        elif 10 <= class_index < 36:
            return chr(ord('A') + class_index - 10)  # Uppercase letters
        elif 36 <= class_index < 62:
            return chr(ord('a') + class_index - 36)  # Lowercase letters
        else:
            return '?'  # Unknown class (shouldn't occur)
