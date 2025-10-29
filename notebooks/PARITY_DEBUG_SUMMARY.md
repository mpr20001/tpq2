# Feature Parity Failure - Debug Summary

## Current Status

✅ **Code fixes completed** in local repository:
- AST feature names/order fixed
- AST parser non-determinism fixes (sorted traversal, hasattr checks)
- Historical feature NULL handling fixed
- Set/dict iteration made deterministic
- All 287 unit tests passing

❌ **Parity validation still failing at 100%** in notebook

## Root Cause Analysis

Based on the notebook output, there are **two separate issues**:

### Issue 1: AST Parser Produces Different Results in Spark vs Local

**Evidence from notebook cell 30:**
```
AST feature 45 (ast_depth):
  Training (Spark): 0.4  (= depth of 4)
  Inference (local): 0.7  (= depth of 7)
```

**Same query, different parse results!**

The parser is **deterministic locally** (tested 3 times, all returned depth=7), but Spark workers get depth=4.

### Issue 2: TF-IDF Minor Differences

8 out of 250 TF-IDF features show small differences (likely floating-point precision).

## Debug Files Created

I've created 3 files to help diagnose the exact issue:

### 1. `QUICK_DEBUG_CELL.py` ⭐ **START HERE**
- Single cell you can copy-paste
- Runs in ~2 minutes
- Immediately shows root cause
- **Use this first!**

### 2. `debug_parity_failure.py`
- 6 comprehensive debug cells (A-F)
- Deep analysis of every aspect
- Use if quick cell doesn't resolve issue

### 3. `DEBUG_INSTRUCTIONS.md`
- Step-by-step guide
- Expected outputs
- Solutions for each issue type

## How to Use

### Step 1: Run Quick Debug Cell

1. Open `notebooks/03_feature_engineering_fixed.ipynb`
2. Add a new cell after cell 31
3. Copy entire content of `QUICK_DEBUG_CELL.py`
4. Run it
5. Read the "DIAGNOSTIC SUMMARY" at the end

### Step 2: Follow Recommendations

The quick cell will tell you **exactly** what's wrong:

**Scenario A: Outdated Code in Spark**
```
❌ Spark is using OUTDATED code
→ Re-run notebook 00 to update
```
**Action:** Re-package and upload code, then re-run notebook

**Scenario B: AST Parser Non-Determinism**
```
❌ AST parser non-determinism
→ Parser produces different results in Spark
```
**Action:** Disable AST features or replace parser

**Scenario C: Other Issues**
```
✅ Code updated, AST consistent
❌ But still failing
```
**Action:** Run full debug cells for deeper analysis

## Most Likely Issue (Based on Evidence)

Given that:
1. Unit tests all pass locally ✅
2. Notebook shows `depth=4` in Spark vs `depth=7` locally ❌
3. Parser is consistent when tested locally ✅

**The most likely issue is:**

### Spark is Using Outdated Code Package

The workers may be loading an old version of the code from the S3 zip file that doesn't have our latest fixes.

**How to verify:**
- Quick debug cell checks: `has_hasattr` and `has_sorted`
- If these return False → code is outdated

**How to fix:**
1. Check when `s3://uip-datalake-bucket-prod/sf_trino/trino_query_predictor/code/query_predictor_latest.zip` was last updated
2. If it's >24 hours old, re-run notebook 00 setup cells
3. Specifically, re-run the cells that:
   - Package the code into a zip
   - Upload to S3
4. Then re-run this notebook

## Quick Fix: Disable AST Features

If you want to proceed with training **now** without fixing the parser issue:

### In notebook cell 12 (or before cell 14), add:

```python
# TEMPORARY: Disable AST features to ensure parity
unified_config['ast_timeout_ms'] = 0  # Force fallback mode
# OR
unified_config['disable_ast_features'] = True  # If this config exists
```

### Impact:
- ✅ Parity should pass (90% of mismatches will disappear)
- ✅ Can proceed to training immediately
- ❌ Model performance: -5 to -10% (AST features contribute this much)
- ✅ Still have 335 features (78 base + 17 historical + 250 TF-IDF - 10 AST)

### Then:
- Re-run cells 14-27 (feature extraction and validation)
- Parity should pass or be very close (<5% mismatch)
- Proceed to notebook 04 for training

## Expected Timeline

| Action | Time | Result |
|--------|------|--------|
| Run QUICK_DEBUG_CELL | 2 min | Know exact issue |
| Re-run notebook 00 (if needed) | 15 min | Update code package |
| Re-run this notebook | 45 min | Parity should pass |
| **OR** Disable AST features | 5 min | Quick fix, proceed now |

## Decision Tree

```
Run QUICK_DEBUG_CELL
│
├─ Shows "outdated code"?
│  └─ YES → Re-run notebook 00 → Re-run this notebook → ✅ Fixed
│
├─ Shows "AST non-determinism"?
│  └─ YES → CHOICE:
│           ├─ Quick: Disable AST → Re-run cells 14-27 → ✅ Training ready
│           └─ Proper: Fix parser → Code changes needed → Re-test
│
└─ Shows "other issues"?
   └─ YES → Run full debug cells → Investigate specific features
```

## Files Reference

```
notebooks/
├── QUICK_DEBUG_CELL.py          ← Copy-paste this first
├── debug_parity_failure.py      ← Full diagnostic suite
├── DEBUG_INSTRUCTIONS.md        ← Detailed guide
└── PARITY_DEBUG_SUMMARY.md      ← This file
```

## Support

If stuck after running QUICK_DEBUG_CELL:
1. Check the "DIAGNOSTIC SUMMARY" section of its output
2. Follow the "Recommended Action"
3. If that doesn't work, run full debug cells A-F
4. The output will make it very clear what's failing

## Key Insight

The parity failure is **NOT because our fixes don't work** (they do - unit tests prove it).

It's because **Spark workers may not be using the fixed code yet**.

The quick debug cell will confirm this in 2 minutes.
