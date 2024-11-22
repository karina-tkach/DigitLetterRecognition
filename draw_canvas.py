import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np


class DrawCanvas(tk.Canvas):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.image = Image.new('L', (280, 280), color='black')
        self.draw = ImageDraw.Draw(self.image)
        self.bind("<B1-Motion>", self.paint)
        self.bind("<ButtonRelease-1>", self.reset_coords)
        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.create_line(self.last_x, self.last_y, x, y, width=10, fill='white', capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill='white', width=10)
        self.last_x, self.last_y = x, y

    def reset_coords(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.delete("all")
        self.image = Image.new('L', (280, 280), color='black')
        self.draw = ImageDraw.Draw(self.image)

    def preprocess_canvas(self, image):
        img = image.convert('L').resize((28, 28))  # Convert to grayscale - brightness
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        img = img.flatten().reshape(-1, 1)  # Reshape to column vector (784, 1)
        return img
