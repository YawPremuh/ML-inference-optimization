from pathlib import Path
import sys
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


onnx_path = Path("models/resnet18.onnx")
image_path = Path(f"{sys.argv[1]}")


#Load up the shared resnet18 metadata
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
categories = weights.meta["categories"]


def get_top_preds(output_tensor, top_k=3):
    probs = torch.softmax(output_tensor, dim=0)

    top_probs, top_indxs = torch.topk(probs, top_k)

    preds = []

    for probability, index in zip(top_probs, top_indxs):
        preds.append((
            categories[index.item()], 
            probability.item())
        )
    return preds

def main():

    #Prepare the image, preprocess and batch size
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    #Run pytorch inference
    pytorch_model = resnet18(weights=weights)
    pytorch_model.eval()

    with torch.inference_mode():
        pytorch_output = pytorch_model(input_batch)[0]

    pytorch_preds = get_top_preds(pytorch_output)

    #Then run onnx runtime inference
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    #Find onnx model input name
    input_name = ort_session.get_inputs()[0].name

    #Convert pytorch tensor -> numpy
    onnx_input = input_batch.numpy()

    #Run onnx runtime
    onnx_output = ort_session.run(None, {input_name: onnx_input})[0]

    #Convert onnx output -> pytorch tensor
    onnx_output_tensor = torch.from_numpy(onnx_output[0])

    onnx_preds = get_top_preds(onnx_output_tensor)

    #Compare the outputs from pytorch and onnx runtime inference
    print("\nPyTorch top 3 predictions:\n")

    for class_name, confidence in pytorch_preds:
        print(f"Class: {class_name:<25}  Confidence: {confidence * 100:.4f}%")

    print("\nONNX Runtime top 3 predictions:\n")

    for class_name, confidence in onnx_preds:
        print(f"Class: {class_name:<25}  Confidence: {confidence * 100:.4f}%")

    #Compare raw model outputs to validate equivalence between the pytorch and onnx runtime
    max_diff = torch.max(torch.abs(pytorch_output - onnx_output_tensor)).item()

    print(f"\nMaximum output difference: {max_diff:.8f}")


if __name__ == "__main__":
    main()