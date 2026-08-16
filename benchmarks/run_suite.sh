#!/bin/bash

set -e

TEST=${1:-1}

RESULTS_DIR="benchmarks/results"
TEMP_RESULTS="$RESULTS_DIR/results.csv"
FINAL_RESULTS="$RESULTS_DIR/results_test${TEST}.csv"

echo ""
echo "====================================="
echo " ML Inference Benchmark - Test $TEST"
echo "====================================="
echo ""

rm -f "$TEMP_RESULTS"

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

mv "$TEMP_RESULTS" "$FINAL_RESULTS"

echo ""
echo "Test $TEST complete."
echo "Results saved to:"
echo "$FINAL_RESULTS"