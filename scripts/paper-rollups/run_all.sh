#!/bin/bash
# Run all three parts of the rollup fidelity study

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON_INTERPRETER:-python3}"
SCRIPTS_DIR="$REPO_ROOT/scripts/paper-rollups"

echo "=== Rollup Fidelity Study ==="
echo ""

echo "Part 1: Structural Loss Accounting"
echo "-----------------------------------"
$PYTHON "$SCRIPTS_DIR/part1_structural_loss.py"
echo ""

echo "Part 2: Question-Class Retention"
echo "---------------------------------"
$PYTHON "$SCRIPTS_DIR/part2_question_retention.py"
echo ""

echo "Part 3: Dead End Verification"
echo "------------------------------"
$PYTHON "$SCRIPTS_DIR/part3_dead_end_verification.py"
echo ""

echo "=== Study Complete ==="
echo "Results in: $REPO_ROOT/data/paper-rollups/"
