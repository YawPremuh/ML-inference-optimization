from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from app.model import predict_image


app = FastAPI(
    title="ML Inference Optimization & Serving Platform",
    description="ResNet18 inference API for ML systems performance experiments.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "ML Inference Optimization & Serving Platform"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Error: Uploaded an image file."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        predictions = predict_image(image)

        return {
            "Filename": file.filename,
            "Predictions": predictions
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"An error occured and image was unable to be processed: {error}"
        )