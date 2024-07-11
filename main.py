from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://tomato-disease-detection-frontend-nu.vercel.app/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PATH = os.path.join(os.path.join(os.path.dirname(__file__), 'models'), 'model.tflite')

# Memuat model TensorFlow Lite
interpreter = tflite.Interpreter(model_path=PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy",
]

@app.get("/ping")
async def ping():
    return "Hello Bro"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0).astype(np.float32)
    
    # Menyiapkan input tensor
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    
    # Melakukan inferensi
    interpreter.invoke()
    
    # Mengambil output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
