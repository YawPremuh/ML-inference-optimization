import argparse
import csv
import time
from pathlib import Path
import numpy as np
import psutil


input_path = Path("benchmarks/input.npy")
onnx_path = Path("models/resnet18.onnx")
results_path = Path("benchmarks/results/results.csv")


def memory_mb():
    
    #Return current process resident memory in MB
    process = psutil.Process()
    return (process.memory_info().rss/ (1024 ** 2))


def calculate_metrics(
    latencies_ms,
    batch_size
):
    latencies = np.array(
        latencies_ms
    )

    total_seconds = (
        latencies.sum()
        / 1000
    )

    total_images = (
        len(latencies)
        * batch_size
    )

    throughput = (
        total_images
        / total_seconds
    )

    return {
        "mean_ms":
            float(np.mean(latencies)),

        "p50_ms":
            float(np.percentile(
                latencies,
                50
            )),

        "p95_ms":
            float(np.percentile(
                latencies,
                95
            )),

        "p99_ms":
            float(np.percentile(
                latencies,
                99
            )),

        "throughput_images_sec":
            float(throughput)
    }


def benchmark_pytorch(
    batch,
    warmup_runs,
    measured_runs
):
    import torch

    from torchvision.models import (
        resnet18,
        ResNet18_Weights
    )

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    model.eval()
    model.to("cpu")

    input_tensor = torch.from_numpy(
        batch
    )

    memory_after_load = memory_mb()

    # Warmup
    with torch.inference_mode():

        for _ in range(warmup_runs):
            model(
                input_tensor
            )

    latencies = []

    # Measured runs
    with torch.inference_mode():

        for _ in range(measured_runs):

            start = (
                time.perf_counter_ns()
            )

            model(
                input_tensor
            )

            end = (
                time.perf_counter_ns()
            )

            latency_ms = (
                (end - start)
                / 1_000_000
            )

            latencies.append(
                latency_ms
            )

    memory_after_benchmark = (
        memory_mb()
    )

    return (
        latencies,
        memory_after_load,
        memory_after_benchmark
    )


def benchmark_onnx(
    batch,
    warmup_runs,
    measured_runs
):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path),
        providers=[
            "CPUExecutionProvider"
        ]
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    memory_after_load = memory_mb()

    # Warmup
    for _ in range(warmup_runs):

        session.run(
            None,
            {
                input_name: batch
            }
        )

    latencies = []

    # Measured runs
    for _ in range(measured_runs):

        start = (
            time.perf_counter_ns()
        )

        session.run(
            None,
            {
                input_name: batch
            }
        )

        end = (
            time.perf_counter_ns()
        )

        latency_ms = (
            (end - start)
            / 1_000_000
        )

        latencies.append(
            latency_ms
        )

    memory_after_benchmark = (
        memory_mb()
    )

    return (
        latencies,
        memory_after_load,
        memory_after_benchmark
    )


def save_result(
    result
):
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        results_path.exists()
    )

    with open(
        results_path,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            result
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runtime",
        choices=[
            "pytorch",
            "onnx"
        ],
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[
            1,
            4,
            8
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

    # Load the exact same benchmark input.
    base_input = np.load(
        input_path
    )

    # Repeat the same image to construct
    # the requested batch.
    batch = np.repeat(
        base_input,
        args.batch_size,
        axis=0
    )

    print(
        f"\nRuntime: {args.runtime}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Warmup runs: {args.warmup}"
    )

    print(
        f"Measured runs: {args.runs}"
    )

    print(
        f"Input shape: {batch.shape}\n"
    )

    memory_before_runtime = (
        memory_mb()
    )

    if args.runtime == "pytorch":

        (
            latencies,
            memory_after_load,
            memory_after_benchmark
        ) = benchmark_pytorch(
            batch,
            args.warmup,
            args.runs
        )

    else:

        (
            latencies,
            memory_after_load,
            memory_after_benchmark
        ) = benchmark_onnx(
            batch,
            args.warmup,
            args.runs
        )

    metrics = calculate_metrics(
        latencies,
        args.batch_size
    )

    runtime_memory_delta = (
        memory_after_load
        - memory_before_runtime
    )

    result = {
        "runtime":
            args.runtime,

        "device":
            "cpu",

        "batch_size":
            args.batch_size,

        "warmup_runs":
            args.warmup,

        "measured_runs":
            args.runs,

        "mean_ms":
            round(
                metrics["mean_ms"],
                4
            ),

        "p50_ms":
            round(
                metrics["p50_ms"],
                4
            ),

        "p95_ms":
            round(
                metrics["p95_ms"],
                4
            ),

        "p99_ms":
            round(
                metrics["p99_ms"],
                4
            ),

        "throughput_images_sec":
            round(
                metrics[
                    "throughput_images_sec"
                ],
                2
            ),

        "runtime_memory_mb":
            round(
                runtime_memory_delta,
                2
            ),

        "process_memory_after_mb":
            round(
                memory_after_benchmark,
                2
            )
    }

    print(
        "Results:"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    save_result(
        result
    )

    print(
        f"\nSaved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()