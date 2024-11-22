import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from emnist_data_loader import EMNISTDataLoader
from neural_network import NeuralNetwork
from draw_canvas import DrawCanvas
import numpy as np
from PIL import Image
import os


class DigitRecognitionApp:
    def __init__(self, root, model_filename):
        self.root = root
        self.root.title("Digit and Letter Recognition")

        # Initialize network and canvas
        self.model_filename = model_filename
        self.layer_sizes = [784, 500, 300, 100, 62]
        self.learn_rate = 0.15
        self.epochs = 100
        self.batch_size = 254

        # Load or train the model
        self.network = NeuralNetwork(self.layer_sizes, self.learn_rate, self.epochs, self.batch_size)
        if os.path.exists(self.model_filename):
            self.network.weights, self.network.biases, self.learn_rate, self.epochs = NeuralNetwork.load_network_state(self.model_filename)
        else:
            images, labels = EMNISTDataLoader.get_emnist()
            self.network.mini_batch_gradient_descent(images, labels)
            self.network.save_network_state(self.model_filename)

        # Setup GUI elements
        self.canvas = DrawCanvas(self.root, width=280, height=280, bg="black")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.result_label = tk.Label(self.root, text="Predicted Character: None", font=("Helvetica", 16))
        self.result_label.grid(row=1, column=0, pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=2, column=0, pady=10)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.canvas.clear)
        self.clear_button.grid(row=3, column=0, pady=10)

        self.load_image_button = tk.Button(self.root, text="Load and Predict Image", command=self.load_and_predict_image)
        self.load_image_button.grid(row=4, column=0, pady=10)

    def predict_digit(self):
        img = self.canvas.preprocess_canvas(self.canvas.image.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
        prediction, _ = self.network.forward_propagation(img)
        predicted_class = np.argmax(prediction)
        predicted_char = NeuralNetwork.class_to_char(predicted_class)
        self.result_label.config(text=f"Predicted Character: {predicted_char}")

    def load_and_predict_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            img = Image.open(file_path)
            processed_img = self.canvas.preprocess_canvas(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
            prediction, _ = self.network.forward_propagation(processed_img)
            predicted_class = np.argmax(prediction)
            predicted_char = NeuralNetwork.class_to_char(predicted_class)

            # Display the image and prediction result
            plt.imshow(np.array(img.convert('L').resize((28, 28))), cmap="Greys")#.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM).convert('L').resize((28, 28))), cmap="Greys")
            #plt.imshow(np.array(img.convert('L').resize((28, 28))), cmap="Greys")
            plt.title(f"Predicted Character: {predicted_char}")
            plt.axis("off")
            plt.show()
