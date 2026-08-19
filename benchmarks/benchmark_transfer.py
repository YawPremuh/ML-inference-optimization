import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

input_path = Path("benchmarks/input.npy")

def calculate_metrics(latencies_ms, batch_size):
    latencies = np.array(latencies_ms)
    total_seconds = latencies.sum() / 1000
    total_images = len(latencies) * batch_size
    throughput = (total_images / total_seconds)

    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_images_sec": float(throughput),
    }


def benchmark(batch_size, warmup_runs, measured_runs, include_transfer):
    device = torch.device("mps")
    base_input = np.load(input_path)
    batch_array = np.repeat(base_input, batch_size, axis=0)

    cpu_input = torch.from_numpy(batch_array)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    model = model.to(device)
    mps_input = cpu_input.to(device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            if include_transfer:
                current_input = cpu_input.to(device)
            else:
                current_input = mps_input

            model(current_input)

        torch.mps.synchronize()

    latencies = []

    with torch.inference_mode():
        for _ in range(measured_runs):
            torch.mps.synchronize()
            start = time.perf_counter_ns()

            if include_transfer:
                current_input = cpu_input.to(device)
            else:
                current_input = mps_input

            model(current_input)
            torch.mps.synchronize()
            end = time.perf_counter_ns()
            latency_ms = (end - start) / 1_000_000
            latencies.append(latency_ms)

    return calculate_metrics(latencies, batch_size)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[1, 4, 8, 32],
        required=True
    )

    parser.add_argument(
        "--mode",
        choices=[
            "inference",
            "transfer"
        ],
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

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")

    include_transfer = (args.mode == "transfer")

    metrics = benchmark(
        args.batch_size,
        args.warmup,
        args.runs,
        include_transfer
    )

    print(f"\nMode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mean: {metrics['mean_ms']:.4f} ms")
    print(f"p50: {metrics['p50_ms']:.4f} ms")
    print(f"p95: {metrics['p95_ms']:.4f} ms")
    print(f"p99: {metrics['p99_ms']:.4f} ms")
    print(f"Throughput: {metrics['throughput_images_sec']:.2f} images/sec")

if __name__ == "__main__":
    main()