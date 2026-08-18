import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

input_path = Path("benchmarks/input.npy")

def calculate_metrics(latencies_ms, batch_size):
    latencies = np.array(latencies_ms)
    total_seconds = (latencies.sum() / 1000)
    total_images = (len(latencies) * batch_size)
    throughput = (total_images / total_seconds)

    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput": float(throughput)
    }


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()

def benchmark(device_name, batch_size, warmup_runs, measured_runs):
    device = torch.device(device_name)
    base_input = np.load(input_path)
    batch_array = np.repeat(base_input, batch_size,axis=0)
    input_tensor = torch.from_numpy(batch_array)

    weights = (ResNet18_Weights.DEFAULT)
    model = resnet18(weights=weights)
    model.eval()
    model = model.to(device)

    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(input_tensor)

        synchronize(device)

    latencies = []

    with torch.inference_mode():
        for _ in range(measured_runs):
            synchronize(device)

            start = (time.perf_counter_ns())
            model(input_tensor)

            synchronize(device)

            end = (time.perf_counter_ns())

            latency_ms = ((end - start) / 1_000_000)
            latencies.append(latency_ms)

    return calculate_metrics(latencies, batch_size)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[1, 4, 8, 32],
        required=True
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=300
    )

    args = parser.parse_args()

    if (args.device == "mps" and not torch.backends.mps.is_available()):
        raise RuntimeError("Error: MPS is not available on this system.")

    print(f"\nDevice: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Measured runs: {args.runs}")

    metrics = benchmark(
        args.device,
        args.batch_size,
        args.warmup,
        args.runs,
    )

    print("\nResults:")
    print(f"mean_ms: {metrics['mean_ms']:.4f}")
    print(f"p50_ms: {metrics['p50_ms']:.4f}")
    print(f"p95_ms: {metrics['p95_ms']:.4f}")
    print(f"p99_ms: {metrics['p99_ms']:.4f}")
    print(f"throughput_images_sec: {metrics['throughput']:.2f}")

if __name__ == "__main__":
    main()