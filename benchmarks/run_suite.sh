#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
test_number="${1:-1}"

if [[ ! "$test_number" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: test number must be a positive integer." >&2
    exit 2
fi

cd "$repo_root"

results_dir="$script_dir/results"
temp_results="$results_dir/results.csv"
final_results="$results_dir/results_test${test_number}.csv"

mkdir -p "$results_dir"

if [[ -e "$final_results" ]]; then
    echo "Error: Test $final_results already exists; choose another test number." >&2
    exit 1
fi

echo ""
echo "====================================="
echo " ML Inference Benchmark - Test $test_number"
echo "====================================="
echo ""

rm -f "$temp_results"

for runtime in pytorch onnx; do
    for batch in 1 4 8 32; do
        echo ""
        echo "Runtime: $runtime | Batch: $batch"
        echo "-------------------------------------"

        "$python_bin" "$script_dir/benchmark.py" \
            --runtime "$runtime" \
            --batch-size "$batch"
    done
done

mv "$temp_results" "$final_results"
"$python_bin" "$script_dir/summarize_results.py"

echo ""
echo "Test $test_number complete."
echo "Results saved to:"
echo "$final_results"
echo "Summary saved to:"
echo "$results_dir/summary.csv"
