from pathlib import Path
import onnx
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch.export import Dim


onnx_path = Path("models/resnet18.onnx") #Save exported onnx model into the models folder

def export_model():
    #Load pretrained ResNet18, put model in evaluation/inference mode
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    #Example input with batch=1, RGB=3, height=224, width=224
    example_input = torch.randn(1, 3, 224, 224)

    #Dynamic batch size
    batch_size = Dim("batch", min=1, max=64)

    print("Exporting ResNet18 to ONNX...")

    # Export using PyTorch's recommended ONNX exporter
    onnx_program = torch.onnx.export(
        model, 
        (example_input,), 
        dynamo=True,
        dynamic_shapes={
            "x":{
                0: batch_size
            }
        }
    )

    onnx_program.save(onnx_path) #Save the onnx model
    print(f"The ONNX model saved to: {onnx_path}")

    #Export model validation
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("The ONNX model validation successful.")

if __name__ == "__main__":
    export_model()