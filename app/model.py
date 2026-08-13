import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

#set up ResNet18 as model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]

#function to run inference
def predict_image(image: Image.Image, top_preds: int = 3):

    image = image.convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.inference_mode():
        output = model(input_batch)

    probabilities = torch.softmax(output[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_preds)

    predictions = []

    for probability, class_index in zip(top_probs, top_indices):
        predictions.append({
            "Class": categories[class_index.item()],
            "Confidence": round(probability.item() * 100, 2)
            })

    return predictions