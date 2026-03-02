#!/bin/bash

set -e

PORT=42341

echo "========================================"
echo "Running ZKP Simulation"
echo "========================================"
echo "Port: $PORT"
echo ""

# Check if sim.exe exists
if [ ! -f "./sim.exe" ]; then
    echo "Error: sim.exe not found. Please compile first with: ./compile.sh"
    exit 1
fi

# Clean up any previous log files
rm -f prover.log verifier.log

echo "Starting verifier (party 2) in background..."
./sim.exe 2 $PORT 2>&1 | tee verifier.log &
VERIFIER_PID=$!

echo "Verifier PID: $VERIFIER_PID"
echo "Waiting 2 seconds for verifier to initialize..."
sleep 2

echo ""
echo "Starting prover (party 1)..."
./sim.exe 1 $PORT 2>&1 | tee prover.log &
PROVER_PID=$!

echo "Prover PID: $PROVER_PID"
echo ""

echo "Waiting for processes to complete..."
echo ""

# Wait for both processes
wait $PROVER_PID 2>/dev/null || true
wait $VERIFIER_PID 2>/dev/null || true

echo ""
echo "========================================"
echo "Simulation completed"
echo "========================================"
