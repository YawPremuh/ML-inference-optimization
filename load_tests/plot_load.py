from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

summary_path = Path("load_tests/results/load_summary.csv")
plots_dir = Path("load_tests/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

def main():

    df = pd.read_csv(summary_path)

    plt.figure()
    plt.plot(df["users"], df["requests_per_sec"], marker="o")
    plt.xlabel("Concurrent Users")
    plt.ylabel("Throughput (requests/sec)")
    plt.title("API Throughput vs Concurrent Users")
    plt.xticks(df["users"])
    plt.grid(True)

    throughput_path = (plots_dir / "throughput_vs_users.png")

    plt.savefig(throughput_path, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {throughput_path}")

    #latency plot
    plt.figure()
    plt.plot(df["users"], df["avg_ms"], marker="o", label="Average")
    plt.plot(df["users"], df["p95_ms"], marker="o", label="p95")
    plt.plot(df["users"],df["p99_ms"],marker="o",label="p99")
    plt.xlabel("Concurrent Users")
    plt.ylabel("Response Time (ms)")
    plt.title("API Latency vs Concurrent Users")
    plt.xticks(df["users"])
    plt.legend()
    plt.grid(True)

    latency_path = (plots_dir / "latency_vs_users.png")

    plt.savefig(latency_path, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {latency_path}")


if __name__ == "__main__":
    main()