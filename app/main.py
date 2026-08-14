from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from app.model import predict_image


app = FastAPI(
    title="ML Inference Optimization & Serving Platform",
    description="ResNet18 inference API for ML systems performance experiments.",
)

#Return basic API info
@app.get("/")
def root():
    return {
        "message": "ML Inference Optimization & Serving Platform"
    }

#Check the status of the model service to see if it is running
@app.get("/health") 
def health():
    return {
        "status": "healthy"
    }

#Take an image file as input and return the top 3 prediction from the resnet18 model
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Error: Uploaded file must be an image (eg. image/1.jpg, image/2.jpeg)."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        predictions = predict_image(image)

        return {
            "filename": file.filename,
            "predictions": predictions
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error: Image file could not be processed: {error}"
        )