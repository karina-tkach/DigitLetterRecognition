import tkinter as tk
from digit_recognition_app import DigitRecognitionApp

if __name__ == "__main__":
    root = tk.Tk()
    model_filename = "mini_batch_trained_network500_300_100.pkl"
    app = DigitRecognitionApp(root, model_filename)
    root.mainloop()
