import numpy as np
import pathlib


class EMNISTDataLoader:
    @staticmethod
    def get_emnist():
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/emnist_byclass_train.npz") as f:
            images, labels = f["images"], f["labels"]

        images = images.astype("float32") / 255.0
        images = np.reshape(images, (images.shape[0], -1))  # Flatten images
        labels = np.eye(62)[labels]  # One-hot encoding true label
        return images, labels
