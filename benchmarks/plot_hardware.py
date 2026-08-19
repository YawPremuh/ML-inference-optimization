from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

summary_path = Path("benchmarks/results/hardware_summary.csv")
plots_dir = Path("benchmarks/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(summary_path)

    plt.figure()

    for device in df["device"].unique():
        device_data = df[df["device"] == device].sort_values("batch_size")

        plt.plot(
            device_data["batch_size"],
            device_data["avg_throughput"],
            marker="o",
            label=device.upper()
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (images/sec)")
    plt.title("PyTorch CPU vs MPS Inference Throughput")
    plt.legend()
    plt.grid(True)

    throughput_path = (plots_dir / "cpu_vs_mps_throughput.png")

    plt.savefig(throughput_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {throughput_path}")

    plt.figure()

    for device in df["device"].unique():
        device_data = df[df["device"] == device].sort_values("batch_size")

        plt.plot(
            device_data["batch_size"],
            device_data["avg_mean_ms"],
            marker="o",
            label=device.upper()
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Mean Batch Latency (ms)")
    plt.title("PyTorch CPU vs MPS Inference Latency")
    plt.legend()
    plt.grid(True)

    latency_path = (plots_dir / "cpu_vs_mps_latency.png")

    plt.savefig(latency_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {latency_path}")


if __name__ == "__main__":
    main()