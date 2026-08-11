# ML-inference-optimization
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
* [ ] Export ResNet18 to ONNX
* [ ] Run inference with ONNX Runtime
* [ ] Benchmark PyTorch vs ONNX Runtime
* [ ] Test batch sizes 1, 4, and 8
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

## Technologies/Tools

* Python
* PyTorch
* TorchVision
* Pillow
* Git / GitHub

Additional technologies will be introduced as the project develops, including ONNX Runtime, FastAPI, Locust, and Docker.

## Project file structure

```text
ML_inf_and_serving/
├── images/
│   ├── ball.jpg
│   ├── car.jpeg
│   ├── dog.jpg
│   └── dog2.jpg
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

> How can ML inference be made faster and more efficient under real request load?

The conclusions to this question will be based on measured benchmark results rather than assumed performance improvements.
