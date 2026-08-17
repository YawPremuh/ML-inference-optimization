#!/bin/bash

set -e

test=${1:-1}

results_dir="benchmarks/results"
temp_results="$results_dir/results.csv"
final_results="$results_dir/results_test${test}.csv"

echo ""
echo "====================================="
echo " ML Inference Benchmark - Test $test"
echo "====================================="
echo ""

rm -f "$temp_results"

for runtime in pytorch onnx
do
    for batch in 1 4 8 32
    do
        echo ""
        echo "Runtime: $runtime | Batch: $batch"
        echo "-------------------------------------"

        python benchmarks/benchmark.py \
            --runtime "$runtime" \
            --batch-size "$batch"
    done
done

mv "$temp_results" "$final_results"

echo ""
echo "Test $test complete."
echo "Results saved to:"
echo "$final_results"