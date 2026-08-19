import csv
import re
from pathlib import Path

results_dir = Path("load_tests/results")
output_file = results_dir / "load_summary.csv"

def get_user_count(file_path):
    match = re.search(r"users_(\d+)_stats\.csv", file_path.name)

    if not match:
        raise ValueError(f"Could not read user count from {file_path}")

    return int(match.group(1))


def main():

    stats_files = sorted(results_dir.glob("users_*_stats.csv"), key=get_user_count)

    if not stats_files:
        raise RuntimeError("No Locust stats files found.")

    summary_rows = []

    for stats_file in stats_files:
        users = get_user_count(stats_file)

        with open(
            stats_file,
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                if row["Name"] == "Aggregated":
                    summary_rows.append({
                        "users": users,
                        "requests": int(row["Request Count"]),
                        "failures": int(row["Failure Count"]),
                        "avg_ms": round(float(row["Average Response Time"]), 2),
                        "p50_ms": float(row["50%"]),
                        "p95_ms": float(row["95%"]),
                        "p99_ms": float(row["99%"]),
                        "requests_per_sec": round(float(row["Requests/s"]), 2),
                    })

    if not summary_rows:
        raise RuntimeError("No aggregated Locust rows were found.")

    fieldnames = summary_rows[0].keys()

    with open(
        output_file,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nLoad test summary saved to: {output_file}\n")

    for row in summary_rows:
        print(
            f"users={row['users']:<2} | "
            f"avg={row['avg_ms']:>7.2f} ms | "
            f"p95={row['p95_ms']:>7.2f} ms | "
            f"p99={row['p99_ms']:>7.2f} ms | "
            f"throughput={row['requests_per_sec']:>6.2f} req/s | "
            f"failures={row['failures']}"
        )


if __name__ == "__main__":
    main()