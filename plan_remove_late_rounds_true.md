# Plan: Experiment with `remove_late_rounds=True` for Champion Model

## Motivation

The current champion model trains with `remove_late_rounds=False`, meaning all rounds of each season are included in training data. Late-season rows are "easy" examples where the outcome is near-certain (champion already clinched, others mathematically eliminated). This inflates metrics but may hurt early/mid-season prediction quality.

Evidence from current results:
- LogisticRegression has a ~7-point OOT drop (CV 0.996 -> OOT 0.924), likely over-relying on late-season signals like `season_fraction` or cumulative points that shift year-to-year
- XGBoost (OOT 0.988) is more robust due to non-linear interaction learning, but may still benefit from cleaner training signal

## Hypothesis

Removing the last 4 rounds per season from training will:
1. Force the model to learn from earlier, more uncertain race data
2. Improve early/mid-season prediction accuracy (the useful case for the Streamlit app)
3. Potentially reduce the train-to-OOT gap across all models
4. Make LogisticRegression more competitive by removing the late-season crutch

## Implementation

### Step 1: Change `remove_late_rounds` flag

In `ml/champion_model.py`, change line ~40:

```python
# Before
remove_late_rounds=False

# After
remove_late_rounds=True
```

This activates the existing logic in `ml/utils.py` (lines 97-110) that filters out the last 4 rounds per year from the training set.

### Step 2: Run the experiment

```bash
python -m ml.champion_model
```

### Step 3: Compare results

Compare the new run against the current baseline in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Key metrics to compare:
- `cv_auc` and `tuned_cv_auc` (cross-validation)
- `auc_train` vs `auc_test` gap (overfitting indicator)
- `auc_oot` (generalization to 2025 — the most important metric)
- Per-model ranking changes (does LogisticRegression improve? does XGBoost hold?)

### Step 4: Evaluate early-season predictions specifically

Beyond aggregate OOT AUC, check how models perform on early-season rows (races 1-10) vs mid-season (11-18) vs late (19+). This can be done by filtering the OOT set by `season_race_number` and computing AUC per segment.

## Expected Outcomes

| Scenario | Interpretation | Action |
|----------|---------------|--------|
| OOT AUC improves for most models | Late-season data was adding noise for generalization | Keep `remove_late_rounds=True` |
| OOT AUC drops slightly but early-season accuracy improves | Trade-off between aggregate and practical accuracy | Keep it if Streamlit app focuses on in-season predictions |
| OOT AUC drops significantly | Late-season data was providing useful training signal | Revert to `remove_late_rounds=False` |

## Risks

- Smaller training set (losing ~4 rows per driver per year) could hurt models that need more data
- The `remove_late_rounds` cutoff of 4 is hardcoded — may need tuning (try 3 or 5 as well)

## Follow-up Ideas

- Train separate models for early-season (races 1-10) and full-season predictions
- Add `season_fraction` interaction features that let the model explicitly learn phase-dependent patterns
- Weight training samples by inverse season progress (early races get higher weight)
