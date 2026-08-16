import sys
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.models import ResNet18_Weights

output_path = Path("benchmarks/input.npy")

def prepare_input(image_path):
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    image = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    input_array = input_batch.numpy()

    np.save(output_path, input_array)

    print("Benchmark input saved to:", output_path)
    print("Input shape:", input_array.shape)
    print("Input dtype:", input_array.dtype)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: \nEnter cmd in this format -> python benchmarks/prepare_input.py " "<image_path>")
        sys.exit(1)

    prepare_input(sys.argv[1])