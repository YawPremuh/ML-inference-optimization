#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
test_num="${1:-}"

if [[ ! "$test_num" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Enter command in this format -> ./benchmarks/run_hardware_suite.sh <test_num>"
    exit 2
fi

cd "$repo_root"

results_dir="$script_dir/results"
output="$results_dir/hardware_test${test_num}.csv"

mkdir -p "$results_dir"

rm -f "$output"

echo ""
echo "====================================="
echo " Hardware Benchmark - Test $test_num"
echo "====================================="
echo ""

if [ $((test_num % 2)) -eq 0 ]; then
    devices=("mps" "cpu")
    batches=(32 8 4 1)
else
    devices=("cpu" "mps")
    batches=(1 4 8 32)
fi


for batch in "${batches[@]}"
do
    for device in "${devices[@]}"
    do

        echo ""
        echo "-------------------------------------"
        echo "Device: $device | Batch: $batch"
        echo "-------------------------------------"

        "$python_bin" "$script_dir/benchmark_hardware.py" \
            --device "$device" \
            --batch-size "$batch" \
            --test "$test_num" \
            --output "$output"

    done
done


echo ""
echo "Test $test_num complete."
echo "Results saved to:"
echo "$output"