import csv
import statistics
from pathlib import Path


results_dir = Path("benchmarks/results")
output_file = results_dir / "summary.csv"

Metrics = [
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput_images_sec",
    "runtime_memory_mb",
    "process_memory_after_mb",
]


def main():
    test_files = sorted(results_dir.glob("results_test*.csv"))

    if not test_files:
        raise RuntimeError("No test result files were found.")

    grouped_results = {}

    for test_file in test_files:

        with open(test_file, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                key = (row["runtime"], int(row["batch_size"]))

                if key not in grouped_results:
                    grouped_results[key] = {metric: [] for metric in Metrics}

                for metric in Metrics:
                    grouped_results[key][metric].append(float(row[metric]))

    summary_rows = []

    for (runtime, batch_size), metrics in sorted(
        grouped_results.items(),
        key=lambda item: (
            item[0][0],
            item[0][1]
        )
    ):

        mean_latencies = metrics["mean_ms"]
        throughputs = metrics[
            "throughput_images_sec"
        ]

        row = {
            "runtime": runtime,
            "batch_size": batch_size,
            "trials": len(mean_latencies),

            "avg_mean_ms": round(statistics.mean(mean_latencies), 4),

            "std_mean_ms": round(
                statistics.stdev(mean_latencies)
                if len(mean_latencies) > 1
                else 0,
                4
            ),

            "avg_p50_ms": round(statistics.mean(metrics["p50_ms"]), 4),

            "avg_p95_ms": round(statistics.mean(metrics["p95_ms"]), 4),

            "avg_p99_ms": round(
                statistics.mean(metrics["p99_ms"]), 4),

            "avg_throughput": round(statistics.mean(throughputs), 2),

            "std_throughput": round(
                statistics.stdev(throughputs)
                if len(throughputs) > 1
                else 0,
                2
            ),

            "avg_runtime_memory_mb": round(
                statistics.mean(metrics["runtime_memory_mb"]), 2),

            "avg_process_memory_mb": round(
                statistics.mean(metrics["process_memory_after_mb"]), 2),
        }

        summary_rows.append(row)

    fieldnames = summary_rows[0].keys()

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary saved to: {output_file}\n")

    for row in summary_rows:

        print(
            f"{row['runtime']:<8} "
            f"batch={row['batch_size']:<2} | "
            f"mean={row['avg_mean_ms']:>8.2f} ms | "
            f"p95={row['avg_p95_ms']:>8.2f} ms | "
            f"p99={row['avg_p99_ms']:>8.2f} ms | "
            f"throughput={row['avg_throughput']:>7.2f} img/s"
        )


if __name__ == "__main__":
    main()
