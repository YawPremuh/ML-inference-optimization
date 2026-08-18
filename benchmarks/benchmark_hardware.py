import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
import csv

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

def save_result(output_path, result):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

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

    parser.add_argument(
        "--test",
        type=int,
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True
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

    result = {
    "test": args.test,
    "device": args.device,
    "batch_size": args.batch_size,
    "warmup_runs": args.warmup,
    "measured_runs": args.runs,
    "mean_ms": round(metrics["mean_ms"], 4),
    "p50_ms": round(metrics["p50_ms"], 4),
    "p95_ms": round(metrics["p95_ms"], 4),
    "p99_ms": round(metrics["p99_ms"], 4),
    "throughput_images_sec": round(metrics["throughput"], 2)
}

    print("\nResults:")
    print(f"mean_ms: {metrics['mean_ms']:.4f}")
    print(f"p50_ms: {metrics['p50_ms']:.4f}")
    print(f"p95_ms: {metrics['p95_ms']:.4f}")
    print(f"p99_ms: {metrics['p99_ms']:.4f}")
    print(f"throughput_images_sec: {metrics['throughput']:.2f}")

    save_result(args.output, result)

    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()