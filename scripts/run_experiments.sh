#!/bin/bash

echo "Queuing 50 DVC Experiments"

# Baseline
echo "[1/50] Running baseline experiment..."
dvc exp run --queue -n "baseline"

# Single parameter variations (10 experiments - reduced from 41)
echo "[2/50] Running n_estimators=50..."
dvc exp run --queue -n "n_est_50" --set-param train.n_estimators=50
echo "[3/50] Running n_estimators=150..."
dvc exp run --queue -n "n_est_150" --set-param train.n_estimators=150
echo "[4/50] Running n_estimators=300..."
dvc exp run --queue -n "n_est_300" --set-param train.n_estimators=300

echo "[5/50] Running max_depth=5..."
dvc exp run --queue -n "depth_5" --set-param train.max_depth=5
echo "[6/50] Running max_depth=15..."
dvc exp run --queue -n "depth_15" --set-param train.max_depth=15
echo "[7/50] Running max_depth=25..."
dvc exp run --queue -n "depth_25" --set-param train.max_depth=25

echo "[8/50] Running min_samples_split=5..."
dvc exp run --queue -n "min_split_5" --set-param train.min_samples_split=5
echo "[9/50] Running min_samples_split=15..."
dvc exp run --queue -n "min_split_15" --set-param train.min_samples_split=15

echo "[10/50] Running min_samples_leaf=2..."
dvc exp run --queue -n "min_leaf_2" --set-param train.min_samples_leaf=2
echo "[11/50] Running min_samples_leaf=8..."
dvc exp run --queue -n "min_leaf_8" --set-param train.min_samples_leaf=8

# Two-parameter combinations (15 experiments)
echo "[12/50] Running low_complexity..."
dvc exp run --queue -n "low_complexity" \
  --set-param train.n_estimators=50 \
  --set-param train.max_depth=5

echo "[13/50] Running high_complexity..."
dvc exp run --queue -n "high_complexity" \
  --set-param train.n_estimators=300 \
  --set-param train.max_depth=25

echo "[14/50] Running many_shallow..."
dvc exp run --queue -n "many_shallow" \
  --set-param train.n_estimators=250 \
  --set-param train.max_depth=8

echo "[15/50] Running few_deep..."
dvc exp run --queue -n "few_deep" \
  --set-param train.n_estimators=80 \
  --set-param train.max_depth=20

echo "[16/50] Running split_control_1..."
dvc exp run --queue -n "split_control_1" \
  --set-param train.min_samples_split=10 \
  --set-param train.min_samples_leaf=5

echo "[17/50] Running split_control_2..."
dvc exp run --queue -n "split_control_2" \
  --set-param train.min_samples_split=20 \
  --set-param train.min_samples_leaf=10

echo "[18/50] Running depth_split_balance..."
dvc exp run --queue -n "depth_split_balance" \
  --set-param train.max_depth=12 \
  --set-param train.min_samples_split=8

echo "[19/50] Running estimators_leaf..."
dvc exp run --queue -n "estimators_leaf" \
  --set-param train.n_estimators=180 \
  --set-param train.min_samples_leaf=4

echo "[20/50] Running depth_leaf_control..."
dvc exp run --queue -n "depth_leaf_control" \
  --set-param train.max_depth=18 \
  --set-param train.min_samples_leaf=6

echo "[21/50] Running moderate_trees_strict..."
dvc exp run --queue -n "moderate_trees_strict" \
  --set-param train.n_estimators=120 \
  --set-param train.min_samples_split=12

echo "[22/50] Running wide_shallow..."
dvc exp run --queue -n "wide_shallow" \
  --set-param train.max_features=0.8 \
  --set-param train.max_depth=6

echo "[23/50] Running narrow_deep..."
dvc exp run --queue -n "narrow_deep" \
  --set-param train.max_features=sqrt \
  --set-param train.max_depth=22

echo "[24/50] Running feature_est_balance..."
dvc exp run --queue -n "feature_est_balance" \
  --set-param train.max_features=log2 \
  --set-param train.n_estimators=200

echo "[25/50] Running controlled_growth..."
dvc exp run --queue -n "controlled_growth" \
  --set-param train.max_depth=10 \
  --set-param train.min_samples_split=6

echo "[26/50] Running sparse_features..."
dvc exp run --queue -n "sparse_features" \
  --set-param train.max_features=0.3 \
  --set-param train.n_estimators=220

# Three-parameter combinations (12 experiments)
echo "[27/50] Running balanced_mid..."
dvc exp run --queue -n "balanced_mid" \
  --set-param train.n_estimators=150 \
  --set-param train.max_depth=12 \
  --set-param train.min_samples_split=8

echo "[28/50] Running conservative_stable..."
dvc exp run --queue -n "conservative_stable" \
  --set-param train.n_estimators=100 \
  --set-param train.max_depth=8 \
  --set-param train.min_samples_leaf=4

echo "[29/50] Running aggressive_controlled..."
dvc exp run --queue -n "aggressive_controlled" \
  --set-param train.n_estimators=250 \
  --set-param train.max_depth=20 \
  --set-param train.min_samples_split=10

echo "[30/50] Running pruned_forest..."
dvc exp run --queue -n "pruned_forest" \
  --set-param train.n_estimators=180 \
  --set-param train.min_samples_split=12 \
  --set-param train.min_samples_leaf=6

echo "[31/50] Running feature_limited..."
dvc exp run --queue -n "feature_limited" \
  --set-param train.max_features=sqrt \
  --set-param train.n_estimators=200 \
  --set-param train.max_depth=15

echo "[32/50] Running regularized_1..."
dvc exp run --queue -n "regularized_1" \
  --set-param train.max_depth=10 \
  --set-param train.min_samples_split=15 \
  --set-param train.min_samples_leaf=8

echo "[33/50] Running efficient_deep..."
dvc exp run --queue -n "efficient_deep" \
  --set-param train.n_estimators=140 \
  --set-param train.max_depth=18 \
  --set-param train.max_features=0.6

echo "[34/50] Running robust_ensemble..."
dvc exp run --queue -n "robust_ensemble" \
  --set-param train.n_estimators=220 \
  --set-param train.max_depth=14 \
  --set-param train.min_samples_leaf=3

echo "[35/50] Running compact_model..."
dvc exp run --queue -n "compact_model" \
  --set-param train.n_estimators=90 \
  --set-param train.max_depth=9 \
  --set-param train.max_features=0.5

echo "[36/50] Running diversity_focus..."
dvc exp run --queue -n "diversity_focus" \
  --set-param train.n_estimators=200 \
  --set-param train.max_features=0.4 \
  --set-param train.min_samples_split=5

echo "[37/50] Running controlled_complexity..."
dvc exp run --queue -n "controlled_complexity" \
  --set-param train.max_depth=16 \
  --set-param train.min_samples_split=7 \
  --set-param train.max_features=log2

echo "[38/50] Running high_variance..."
dvc exp run --queue -n "high_variance" \
  --set-param train.n_estimators=160 \
  --set-param train.max_depth=22 \
  --set-param train.min_samples_leaf=2

# Four-parameter combinations (10 experiments)
echo "[39/50] Running optimal_v1..."
dvc exp run --queue -n "optimal_v1" \
  --set-param train.n_estimators=180 \
  --set-param train.max_depth=14 \
  --set-param train.min_samples_split=6 \
  --set-param train.min_samples_leaf=3

echo "[40/50] Running optimal_v2..."
dvc exp run --queue -n "optimal_v2" \
  --set-param train.n_estimators=200 \
  --set-param train.max_depth=16 \
  --set-param train.min_samples_split=8 \
  --set-param train.max_features=sqrt

echo "[41/50] Running optimal_v3..."
dvc exp run --queue -n "optimal_v3" \
  --set-param train.n_estimators=150 \
  --set-param train.max_depth=12 \
  --set-param train.min_samples_leaf=4 \
  --set-param train.max_features=0.6

echo "[42/50] Running tuned_ensemble..."
dvc exp run --queue -n "tuned_ensemble" \
  --set-param train.n_estimators=240 \
  --set-param train.max_depth=18 \
  --set-param train.min_samples_split=10 \
  --set-param train.min_samples_leaf=5

echo "[43/50] Running regularized_strong..."
dvc exp run --queue -n "regularized_strong" \
  --set-param train.n_estimators=120 \
  --set-param train.max_depth=10 \
  --set-param train.min_samples_split=15 \
  --set-param train.min_samples_leaf=7

echo "[44/50] Running precision_model..."
dvc exp run --queue -n "precision_model" \
  --set-param train.n_estimators=170 \
  --set-param train.max_depth=15 \
  --set-param train.min_samples_split=9 \
  --set-param train.max_features=0.7

echo "[45/50] Running balanced_premium..."
dvc exp run --queue -n "balanced_premium" \
  --set-param train.n_estimators=190 \
  --set-param train.max_depth=13 \
  --set-param train.min_samples_split=7 \
  --set-param train.max_features=log2

echo "[46/50] Running fast_accurate..."
dvc exp run --queue -n "fast_accurate" \
  --set-param train.n_estimators=130 \
  --set-param train.max_depth=11 \
  --set-param train.min_samples_leaf=3 \
  --set-param train.max_features=0.55

echo "[47/50] Running diverse_forest..."
dvc exp run --queue -n "diverse_forest" \
  --set-param train.n_estimators=210 \
  --set-param train.max_depth=17 \
  --set-param train.min_samples_split=5 \
  --set-param train.max_features=0.45

echo "[48/50] Running stable_learner..."
dvc exp run --queue -n "stable_learner" \
  --set-param train.n_estimators=160 \
  --set-param train.max_depth=14 \
  --set-param train.min_samples_split=11 \
  --set-param train.min_samples_leaf=6

# Five-parameter combinations (2 experiments)
echo "[49/50] Running ultra_optimized..."
dvc exp run --queue -n "ultra_optimized" \
  --set-param train.n_estimators=185 \
  --set-param train.max_depth=15 \
  --set-param train.min_samples_split=7 \
  --set-param train.min_samples_leaf=4 \
  --set-param train.max_features=0.65

echo "[50/50] Running production_candidate..."
dvc exp run --queue -n "production_candidate" \
  --set-param train.n_estimators=200 \
  --set-param train.max_depth=16 \
  --set-param train.min_samples_split=8 \
  --set-param train.min_samples_leaf=3 \
  --set-param train.max_features=sqrt

echo "All 50 diverse experiments queued successfully"
