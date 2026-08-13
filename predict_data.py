import torch
import sys
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

#check if user entered the right command else throw in an error message
if len(sys.argv) != 2:
    print("Error: \nEnter cmd in this format -> python predict.py <image_path>")
    sys.exit(1)

#load up ResNet18 as our model 
weights = ResNet18_Weights.DEFAULT #pretrained ResNet18 weights as our weights
model = resnet18(weights=weights)
model.eval() 
preprocess = weights.transforms() 

image = Image.open(f"{sys.argv[1]}").convert("RGB")
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

#run inference
with torch.inference_mode():
    output = model(input_batch)

probabilities = torch.softmax(output[0], dim=0) #converting model scores to probabilites
top_probs, top_indices = torch.topk(probabilities, 3) #Get top 3 preds

categories = weights.meta["categories"]

#output
print(f"\nImage: {sys.argv[1]}")
print(f"Input shape: {input_batch.shape}")
print("\nTop 3 predictions: \n")

for prob, index in zip(top_probs, top_indices):
    print(categories[index.item()], f"-> {prob.item() * 100:.2f}%")

