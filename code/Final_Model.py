from CNN_1 import *
import cv2
import numpy as np

# Final output model for inference


class Model:
    def __init__(self):
        self.model = CNN_1(input_shape=(224, 224, 3), num_classes=21)
        self.model.model.load_weights("weights/cnn_1_finetuning2.weights.h5")
        self.class_names = [
            "agricultural",
            "airplane",
            "baseballdiamond",
            "beach",
            "buildings",
            "chaparral",
            "denseresidential",
            "forest",
            "freeway",
            "golfcourse",
            "harbor",
            "intersection",
            "mediumresidential",
            "mobilehomepark",
            "overpass",
            "parkinglot",
            "river",
            "runway",
            "sparseresidential",
            "storagetanks",
            "tenniscourt",
        ]

    def center_crop(self, img):
        height, width = img.shape[:2]
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        img_cropped = img[start_y : start_y + min_dim, start_x : start_x + min_dim]
        return img_cropped

    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return "Error: image not found."
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.center_crop(img)
        img = cv2.resize(img, (224, 224))
        img_batch = np.array([img])
        prediction = self.model.predict(img_batch)
        return f"Img: {image_path}\nClass: {self.class_names[prediction[0]]}"

    def predict_proba(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return "Error: image not found."
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.center_crop(img)
        img = cv2.resize(img, (224, 224))
        img_batch = np.array([img])
        return self.model.predict_proba(img_batch)


if __name__ == "__main__":
    model = Model()
    print(model.predict("images/river.jpg"))
