# ML Inference Optimization & Serving Platform

This is an ML systems project which was done to explore how inference runtime, hardware, batch size and request load affect the performance when serving a model.

The purpose/goal of this project is to explore, learn and understand how to make machine learning inference faster and more efficient while measuring metrics such as latency, thoughput, and memory usage.


## Project Goals

This project will compare:

* PyTorch vs ONNX Runtime
* CPU vs hardware acceleration
* Batch sizes 1, 4, and 8
* p50, p95, and p99 inference latency
* Throughput
* Memory usage
* Performance under concurrent request load

## Progress tracking

* [x] Set up Python virtual environment
* [x] Load pretrained ResNet18 with PyTorch
* [x] Preprocess local images for inference
* [x] Run image classification inference
* [x] Return top model predictions and confidence scores
* [x] Serve the model using FastAPI
* [x] Export ResNet18 to ONNX
* [x] Run inference with ONNX Runtime
* [x] Benchmark PyTorch vs ONNX Runtime
* [x] Test batch sizes 1, 4, and 8
* [ ] Compare CPU and available hardware acceleration
* [ ] Measure p50, p95, and p99 latency
* [ ] Measure throughput and memory usage
* [ ] Load test the inference API
* [ ] Containerize the application with Docker

## Step 1 — PyTorch Inference

My first step was to use a pretrained ResNet model, specifically ResNet18, from TorchVision. I used this model because the purpose of my project is not to build a new AI model from scratch, but to test how fast and efficiently a model can run. ResNet18 is small enough to run on my computer and complex enough to give me valuable performance results.

The current inference pipeline is:

```text
Image (input)
  ↓
Image preprocessing
  ↓
PyTorch tensor
  ↓
ResNet18
  ↓
Model output
  ↓
Softmax probabilities
  ↓
Top 3 predictions 
```

Example output:

```text
Top 3 predictions:

soccer ball 44.67%
rugby ball 43.78%
baseball 3.25%
```

## Step 2 — FastAPI Model Serving

The pretrained ResNet18 model is linked to a FastAPI server so that the predictions made can be requested through HTTP instead of running the inference script manually.

### Serving Pipeline

```text
Image
  ↓
POST /predict
  ↓
FastAPI
  ↓
Image preprocessing
  ↓
ResNet18
  ↓
Top predictions
  ↓
JSON response
```

Example response:

```json
{
  "Filename": "dog_bluejay.jpeg",
  "Predictions": [
    {
      "Class": "Tibetan terrier",
      "Confidence": 0.5854
    },
    {
      "Class": "Maltese dog",
      "Confidence": 0.0931
    },
    {
      "Class": "Lhasa",
      "Confidence": 0.0853
    }
  ]
}
```

## Step 3 — ONNX Export and Validation

In this step, I successfully exported the same ResNet18 from PyTorch to the ONNX format and run ONNX Runtime inference. Then, I validated the exported model using the ONNX model checker. I wrote a script to run both PyTorch and ONNX Runtime inference and, compared the results using the same preprocessed inputs to verify that the exported model actually gives me the same predictions and, also to later compare the runtime. And now that step 3 is successful, I ask myself, which inference/execution engine can handle the same workload better than the other? At the end of this experiment/project I should be able to answer that, alongside a question about whether the performance/deployement benefit is worth all this additional complexity from exporting.

```text
PyTorch ResNet18
       ↓
    Export
       ↓
     ONNX
       ↓
 ONNX Runtime
       ↓
Prediction validation
```

## Step 4 — Dynamic Batch Size Support

I updated the ONNX export to support dynamic batch sizes while I kept the image dimensions fixed at 224×224 to make the exported graph and experimental setup simple.

The model was verified with batch sizes:

* `1`
* `4`
* `8`

for both PyTorch and ONNX Runtime.

```text
[batch, 3, 224, 224]
        ↓
     ResNet18
        ↓
[batch, 1000]
```

This step now prepares the project for benchmarking how batch size affects latency, throughput, and memory usage.



## Technologies/Tools

* Python
* PyTorch
* TorchVision
* Pillow
* Git / GitHub
* FastAPI
* Uvicorn
* ONNX Runtime

Additional technologies will be introduced as the project develops, including ONNX Runtime, Locust, and Docker.

## Project file structure

```text
ML_inf_and_serving/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └──model.py
│
├── images/
│   ├── ball.jpg
│   ├── car.jpeg
│   ├── dog.jpg
│   └── dog2.jpg
│
├── models/
│   └── .gitkeep
│
├── scripts/
│   ├── export_to_onnx.py
│   └── verify_onnx.py
│
├── predict_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/YawPremuh/ML-inference-optimization.git
cd ML-inference-optimization
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Inference

Run the inference script with a local image:

```bash
python predict_data.py {image_path}
```

The predict.py script loads the ResNet18 model, preprocesses the selected image, runs inference, and prints the top 3 highest-confidence predictions.

## Planned Architecture

```text
                    Image Request
                          │
                          ▼
                       FastAPI
                          │
                     Preprocessing
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           PyTorch               ONNX Runtime
              │                       │
              └───────────┬───────────┘
                          ▼
                      Prediction
                          │
                          ▼
                  Performance Metrics
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       Latency        Throughput        Memory
```

## Final Objective

The final project will experimentally answer:

> How can ML inference be made faster and more efficient under real request load? In other words, given the same trained ML model, how does the system around the model affect performance?

The conclusions to this question will be based on measured benchmark results rather than assumed performance improvements.
