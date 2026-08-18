import csv
import statistics
from pathlib import Path


results_dir = Path("benchmarks/results")
output_file = results_dir / "hardware_summary.csv"

METRICS = [
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput_images_sec",
]


def main():

    test_files = sorted( results_dir.glob("hardware_test*.csv"))

    if not test_files:
        raise RuntimeError("Error: No hardware benchmark files found.")

    grouped_results = {}

    for test_file in test_files:

        with open(test_file, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:

                key = (row["device"], int(row["batch_size"]))

                if key not in grouped_results:

                    grouped_results[key] = {
                        metric: []
                        for metric in METRICS
                    }

                for metric in METRICS:

                    grouped_results[key][metric].append(float(row[metric]))

    summary_rows = []

    for (
        device,
        batch_size
    ), metric_values in sorted(
        grouped_results.items(),
        key=lambda item: (
            item[0][0],
            item[0][1]
        )
    ):

        mean_values = (metric_values["mean_ms"])

        throughput_values = (metric_values["throughput_images_sec"])

        row = {
            "device": device,
            "batch_size": batch_size,
            "tests": len(mean_values),
            "avg_mean_ms": round(statistics.mean(mean_values),4),
            "std_mean_ms": round(statistics.stdev(mean_values) if len(mean_values) > 1 else 0, 4),
            "avg_p50_ms": round(statistics.mean(metric_values["p50_ms"]), 4),
            "avg_p95_ms": round(statistics.mean(metric_values["p95_ms"]), 4),
            "avg_p99_ms": round(statistics.mean(metric_values["p99_ms"]), 4),
            "avg_throughput": round(statistics.mean(throughput_values),2),
            "std_throughput": round(statistics.stdev(throughput_values) if len(throughput_values) > 1 else 0, 2)
        }

        summary_rows.append(row)

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nHardware benchmark summary saved to: {output_file}\n")

    for row in summary_rows:

        print(
            f"{row['device']:<4} "
            f"batch={row['batch_size']:<2} | "
            f"mean={row['avg_mean_ms']:>8.2f} ms | "
            f"p95={row['avg_p95_ms']:>8.2f} ms | "
            f"p99={row['avg_p99_ms']:>8.2f} ms | "
            f"throughput="
            f"{row['avg_throughput']:>7.2f} img/s"
        )


if __name__ == "__main__":
    main()