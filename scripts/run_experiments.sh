#!/bin/bash

echo "Starting 21 DVC Experiments"

# Baseline
echo "[1/21] Running baseline experiment..."
dvc exp run -n "baseline"

# Vary n_estimators
echo "[2/21] Running n_estimators=50..."
dvc exp run -n "n_est_50" --set-param train.n_estimators=50

echo "[3/21] Running n_estimators=150..."
dvc exp run -n "n_est_150" --set-param train.n_estimators=150

echo "[4/21] Running n_estimators=200..."
dvc exp run -n "n_est_200" --set-param train.n_estimators=200

echo "[5/21] Running n_estimators=300..."
dvc exp run -n "n_est_300" --set-param train.n_estimators=300

# Vary max_depth
echo "[6/21] Running max_depth=5..."
dvc exp run -n "depth_5" --set-param train.max_depth=5

echo "[7/21] Running max_depth=15..."
dvc exp run -n "depth_15" --set-param train.max_depth=15

echo "[8/21] Running max_depth=20..."
dvc exp run -n "depth_20" --set-param train.max_depth=20

# Vary min_samples_split
echo "[9/21] Running min_samples_split=2..."
dvc exp run -n "min_split_2" --set-param train.min_samples_split=2

echo "[10/21] Running min_samples_split=10..."
dvc exp run -n "min_split_10" --set-param train.min_samples_split=10

echo "[11/21] Running min_samples_split=20..."
dvc exp run -n "min_split_20" --set-param train.min_samples_split=20

# Vary min_samples_leaf
echo "[12/21] Running min_samples_leaf=2..."
dvc exp run -n "min_leaf_2" --set-param train.min_samples_leaf=2

echo "[13/21] Running min_samples_leaf=4..."
dvc exp run -n "min_leaf_4" --set-param train.min_samples_leaf=4

# Vary max_features
echo "[14/21] Running max_features=log2..."
dvc exp run -n "max_feat_log2" --set-param train.max_features=log2

# Combination experiments
echo "[15/21] Running aggressive combination..."
dvc exp run -n "aggressive" \
  --set-param train.n_estimators=200 \
  --set-param train.max_depth=20

echo "[16/21] Running conservative combination..."
dvc exp run -n "conservative" \
  --set-param train.n_estimators=50 \
  --set-param train.max_depth=5

echo "[17/21] Running balanced_high combination..."
dvc exp run -n "balanced_high" \
  --set-param train.n_estimators=150 \
  --set-param train.max_depth=15 \
  --set-param train.min_samples_split=10

echo "[18/21] Running optimal_1 combination..."
dvc exp run -n "optimal_1" \
  --set-param train.n_estimators=200 \
  --set-param train.max_depth=15 \
  --set-param train.min_samples_split=5 \
  --set-param train.min_samples_leaf=2

echo "[19/21] Running optimal_2 combination..."
dvc exp run -n "optimal_2" \
  --set-param train.n_estimators=250 \
  --set-param train.max_depth=12 \
  --set-param train.min_samples_split=8

echo "[20/21] Running optimal_3 combination..."
dvc exp run -n "optimal_3" \
  --set-param train.n_estimators=180 \
  --set-param train.max_depth=18 \
  --set-param train.min_samples_split=6

echo "[21/21] Running deep_forest combination..."
dvc exp run -n "deep_forest" \
  --set-param train.n_estimators=300 \
  --set-param train.max_depth=25

echo "experiements completed"


