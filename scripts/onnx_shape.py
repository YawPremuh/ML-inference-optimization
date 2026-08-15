import onnxruntime as ort

session = ort.InferenceSession(
    "models/resnet18.onnx",
    providers=["CPUExecutionProvider"]
)

model_input = session.get_inputs()[0]

#Print shape data
print("Input name:", model_input.name)
print("Input shape:", model_input.shape)
print("Input type:", model_input.type)