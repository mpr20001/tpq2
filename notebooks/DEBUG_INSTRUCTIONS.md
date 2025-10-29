# Debug Instructions for Feature Parity Failure

## Quick Start

The parity validation is failing with 100% mismatch rate. This debug code will help identify the exact root cause.

### Step 1: Add Debug Cells to Notebook

Open `notebooks/03_feature_engineering_fixed.ipynb` and add new cells **after cell 31** (the masked parity test).

Copy-paste the sections from `debug_parity_failure.py` as separate cells:

```
Cell A: Verify Spark Code Version
Cell B: Create Debug UDF
Cell C: Single Query Deep Dive  ← START HERE
Cell D: Full Feature Vector Comparison
Cell E: Pattern Analysis
Cell F: Recommendations
```

### Step 2: Run Cell C First (Most Important)

**Cell C** performs a single-query deep dive that will immediately show you:
- Exact AST metrics from Spark workers vs local parsing
- Whether they match or differ
- Which specific metrics are wrong

**Expected Output:**
```
1. LOCAL PARSING
   depth: 7
   breadth: 4
   ...

2. SPARK PARSING
   depth: 4  ← Different!
   breadth: 2  ← Different!
   ...

3. COMPARISON
   ❌ AST METRICS DIFFER
```

### Step 3: Check Cell A (Code Version)

If Cell C shows mismatches, run **Cell A** to verify if Spark is using outdated code:

**Key Questions:**
1. Does `Uses hasattr() for Expression check: True`?
2. Does `Uses sorted() for deterministic traversal: True`?
3. Is S3 zip file recent (< 24 hours old)?

**If any answer is NO:**
```python
# You need to re-upload the code package
# Go to notebook 00_setup_and_config.ipynb
# Re-run the cells that package and upload code to S3
```

### Step 4: Run Full Analysis (Cells D-F)

Once you understand the issue from Cells A and C:
- **Cell D**: See which exact features mismatch
- **Cell E**: Check if pattern is consistent across queries
- **Cell F**: Get automated recommendations

## Expected Issues and Solutions

### Issue 1: Outdated Code in Spark ✅ Easy Fix

**Symptoms:**
- Cell A shows `Uses hasattr(): False` or `Uses sorted(): False`
- S3 zip file is >24 hours old

**Solution:**
1. Re-run notebook 00 cells that build code package
2. Re-run this notebook
3. Parity should pass

### Issue 2: AST Parser Non-Determinism 🔧 Harder

**Symptoms:**
- Cell A shows code IS up-to-date
- Cell C shows different AST metrics between Spark and local
- Same query produces depth=4 in Spark, depth=7 locally

**Root Cause:**
The sqlglot parser behaves differently in Spark distributed context due to:
- Different Python environment in workers
- Threading/timeout behavior differences
- Memory constraints

**Solutions (in priority order):**

#### Option A: Disable AST Features (Recommended)
```python
# Add this config before feature extraction
unified_config['disable_ast_features'] = True

# Or set AST timeout to 0 to force fallback
unified_config['ast_timeout_ms'] = 0
```
**Impact:** -5-10% model performance, but guaranteed parity

#### Option B: Use Only Regex Fallback
```python
# Force AST extractor to always use fallback
# Modify ast_extractor.py to always call _fallback_features()
```
**Impact:** Less accurate AST features, but deterministic

#### Option C: Pre-compute AST Features
```python
# Add a preprocessing step to compute AST features
# Store them in the dataset
# Use pre-computed values in both train and inference
```
**Impact:** More complex pipeline, but preserves accuracy

### Issue 3: TF-IDF Precision Differences 📊 Minor

**Symptoms:**
- AST features match (Cell C)
- But TF-IDF features mismatch (Cell D)
- Small floating-point differences (< 0.01)

**Solutions:**
1. Increase tolerance in validator: `validator.tolerance = 1e-4`
2. Use consistent TF-IDF implementation (both sklearn or both Spark ML)
3. Quantize features to lower precision

## Quick Decision Tree

```
Is Cell A showing latest code?
├─ NO → Re-run notebook 00 to update code package
└─ YES → Continue

Does Cell C show AST metrics match?
├─ YES → Problem is elsewhere (TF-IDF, historical, etc.)
│        Run Cell D to identify which features
└─ NO → AST parser non-determinism
         ├─ QUICK FIX: Disable AST features
         └─ LONG-term: Replace parser
```

## Expected Timeline

- **Cell C**: 2-3 minutes (single query test)
- **Cell A**: 30 seconds (code verification)
- **Cells D-F**: 5-10 minutes (full analysis)

## Next Steps After Debugging

Once you identify the root cause:

### If Code is Outdated:
1. Re-package code (notebook 00)
2. Re-run this notebook
3. Expect parity to pass

### If AST Parser Issue:
1. **Quick path**: Disable AST features
2. Re-run feature extraction with disabled AST
3. Parity should pass with 335 features instead of 345
4. Proceed to model training (notebook 04)

### If Other Issues:
1. Review Cell D output to see exact mismatches
2. Add targeted fixes for specific features
3. Re-test

## Support

If you're stuck, check:
1. Cell C output - this is the smoking gun
2. Cell A code verification
3. Cell F recommendations

The debug output should make it crystal clear what's failing and why.
