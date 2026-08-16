from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


summary_path = Path("benchmarks/results/summary.csv")
plot_dir = Path("benchmarks/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

def load_results():
    return pd.read_csv(summary_path)


def plot_throughput(df):
    plt.figure()

    for runtime in df["runtime"].unique():

        runtime_data = df[df["runtime"] == runtime].sort_values("batch_size")

        plt.plot(
            runtime_data["batch_size"],
            runtime_data["avg_throughput"],
            marker="o",
            label=runtime
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (images/sec)")
    plt.title("CPU Inference Throughput vs Batch Size")
    plt.legend()
    plt.grid(True)

    output_path = (plot_dir / "throughput_vs_batch.png")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_latency(df):
    plt.figure()

    for runtime in df["runtime"].unique():

        runtime_data = df[df["runtime"] == runtime].sort_values("batch_size")

        plt.plot(
            runtime_data["batch_size"],
            runtime_data["avg_mean_ms"],
            marker="o",
            label=f"{runtime} mean"
        )

        plt.plot(
            runtime_data["batch_size"],
            runtime_data["avg_p99_ms"],
            marker="o",
            linestyle="--",
            label=f"{runtime} p99"
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("CPU Inference Latency vs Batch Size")
    plt.legend()
    plt.grid(True)

    output_path = (plot_dir / "latency_vs_batch.png")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_memory(df):
    plt.figure()

    for runtime in df["runtime"].unique():

        runtime_data = df[df["runtime"] == runtime].sort_values("batch_size")

        plt.plot(
            runtime_data["batch_size"],
            runtime_data["avg_process_memory_mb"],
            marker="o",
            label=runtime
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Process Resident Memory (MB)")
    plt.title("Observed Process Memory vs Batch Size")
    plt.legend()
    plt.grid(True)

    output_path = (plot_dir / "memory_vs_batch.png")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    df = load_results()

    plot_throughput(df)
    plot_latency(df)
    plot_memory(df)


if __name__ == "__main__":
    main()