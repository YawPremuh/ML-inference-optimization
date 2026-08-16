import sys
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


onnx_path = ("models/resnet18.onnx")

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
categories = weights.meta["categories"]

#Using batch sizes 1, 4 and 8
batch_sizes = [1, 4, 8]

def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")

    return preprocess(image)

#function to create the batches using torch.stack()
def create_batch(input_tensor, size):
    return torch.stack([input_tensor for _ in range(size)])

def get_top_class(output):
    probs = torch.softmax(output, dim=0)
    top_indx = torch.argmax(probs).item()
    confidence = probs[top_indx].item()

    return categories[top_indx], confidence


def main(image_path): 
    input_tensor = prepare_image(image_path)

    #Load up the pytorch model
    pytorch_model = resnet18(weights=weights)
    pytorch_model.eval()

    #Load up the onnx runtime model
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
        )

    input_name = (ort_session.get_inputs()[0].name)

    for size in batch_sizes:

        print("\nTesting batch size: ", size)

        batch = create_batch(input_tensor, size)

        print("Input shape:", batch.shape)

        with torch.inference_mode():
            pytorch_output = pytorch_model(batch)

        onnx_output = ort_session.run(
                None,
                {input_name: batch.numpy()}
            )[0]
        
        print("PyTorch output shape:", pytorch_output.shape)
        print("ONNX output shape:", onnx_output.shape)

        pytorch_class, pytorch_conf = get_top_class(pytorch_output[0])
        onnx_class, onnx_conf = get_top_class(torch.from_numpy(onnx_output[0]))
        max_diff = max_diff = abs(pytorch_conf - onnx_conf)

        print("PyTorch runtime class prediction", pytorch_class, 
            f"-> Confidence: {pytorch_conf * 100:.4f}%"
        )

        print("ONNX runtime class prediction:", onnx_class,
            f"-> Confidence: {onnx_conf * 100:.4f}%")
        
        print(f"Max output difference: {max_diff:.8f}")
        


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Error: \nEnter cmd in this format -> python scripts/verify_batches.py <image_path>")
        sys.exit(1)

    main(sys.argv[1])