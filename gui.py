import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMG_SIZE = 128
model = load_model("deepfake_model.h5")

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "Real" if pred < 0.5 else "Fake"

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}")

app = tk.Tk()
app.title("DeepFake Detector")

tk.Button(app, text="Select Image", command=browse_image).pack(pady=20)
result_label = tk.Label(app, text="Prediction: ", font=("Arial", 16))
result_label.pack()

app.mainloop()
