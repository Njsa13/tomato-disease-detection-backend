from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import os
from os import environ as env
from fastapi import HTTPException, status

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

PATH = os.path.join(os.path.join(os.path.dirname(__file__), 'models'), 'model1.h5')
MODEL = load_model(PATH)
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
    return f"Check ping success, secret = {env['MY_VARIABLE']}"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        prediction = MODEL.predict(img_batch)
        prediction_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        return {
            'class': prediction_class,
            'confidence': float(confidence)
        }
    except (UnidentifiedImageError, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process the image. Make sure it is a valid image file.")
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server Error.")
