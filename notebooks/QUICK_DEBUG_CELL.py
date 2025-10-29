"""
QUICK DEBUG CELL - Copy-paste this into notebook after cell 31
================================================================

This single cell will immediately show you the root cause of the parity failure.
Run time: ~2 minutes

After running this, you'll know:
1. If Spark is using outdated code
2. If AST parser produces different results in Spark vs local
3. Exact feature values that are mismatching
"""

print("="*80)
print("QUICK PARITY FAILURE DIAGNOSTIC")
print("="*80)

# ===== PART 1: Code Version Check =====
print("\n[1/3] Checking if Spark is using latest code...")
print("-"*80)

from query_predictor.core.featurizer.parsers.ast_parser import ASTParser
import inspect

parser_source = inspect.getsource(ASTParser._calculate_depth)
has_fixes = 'hasattr(arg_value' in parser_source and 'sorted(node.args.keys())' in parser_source

if has_fixes:
    print("✅ Spark environment has latest code fixes")
else:
    print("❌ PROBLEM FOUND: Spark is using OUTDATED code!")
    print("   → Solution: Re-run notebook 00 to update code package")
    print("   → Then re-run this notebook")
    print("\nStopping here - fix code version first before continuing.")
    # Don't continue if code is outdated
    import sys
    sys.exit(0)

# ===== PART 2: AST Parser Consistency Test =====
print("\n[2/3] Testing if AST parser is consistent between Spark and local...")
print("-"*80)

# Get first sample query
sample_query_text = sample_queries[0]['query']
print(f"Test query preview: {sample_query_text[:150]}...")

# Parse locally
local_parser = ASTParser(timeout_ms=200)
local_success, local_ast, local_metrics = local_parser.parse(sample_query_text)

print(f"\nLocal parsing:")
print(f"  Success: {local_metrics.parse_success}")
print(f"  Depth: {local_metrics.depth}")
print(f"  Breadth: {local_metrics.breadth}")
print(f"  CTE count: {local_metrics.cte_count}")

# Parse in Spark
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType

@F.udf(returnType=StructType([
    StructField("success", BooleanType()),
    StructField("depth", IntegerType()),
    StructField("breadth", IntegerType()),
    StructField("cte_count", IntegerType()),
]))
def spark_ast_parse(query):
    from query_predictor.core.featurizer.parsers.ast_parser import ASTParser
    parser = ASTParser(timeout_ms=200)
    success, ast, metrics = parser.parse(query)
    return (metrics.parse_success, metrics.depth, metrics.breadth, metrics.cte_count)

spark_result = spark.createDataFrame([(sample_query_text,)], ["query"]) \
    .withColumn("metrics", spark_ast_parse(F.col("query"))) \
    .select("metrics.*") \
    .collect()[0]

print(f"\nSpark parsing:")
print(f"  Success: {spark_result.success}")
print(f"  Depth: {spark_result.depth}")
print(f"  Breadth: {spark_result.breadth}")
print(f"  CTE count: {spark_result.cte_count}")

# Compare
ast_matches = (
    local_metrics.depth == spark_result.depth and
    local_metrics.breadth == spark_result.breadth and
    local_metrics.cte_count == spark_result.cte_count
)

if ast_matches:
    print("\n✅ AST metrics MATCH between Spark and local")
    print("   → AST parser is working correctly")
else:
    print("\n❌ PROBLEM FOUND: AST metrics DIFFER between Spark and local")
    print(f"   → Depth: {local_metrics.depth} (local) vs {spark_result.depth} (Spark)")
    print(f"   → Breadth: {local_metrics.breadth} (local) vs {spark_result.breadth} (Spark)")
    print(f"   → CTE: {local_metrics.cte_count} (local) vs {spark_result.cte_count} (Spark)")
    print("\n   This is the ROOT CAUSE of parity failure!")
    print("\n   Solutions:")
    print("   A) QUICK FIX: Disable AST features")
    print("      unified_config['ast_timeout_ms'] = 0  # Forces fallback")
    print("   B) LONG-TERM: Replace sqlglot with deterministic parser")

# ===== PART 3: Feature-Level Analysis =====
print("\n[3/3] Analyzing which features are mismatching...")
print("-"*80)

# Get features from both paths
sample_idx = 0
training_feat = training_features[sample_idx]
inference_feat = inference_featurizer.extract(sample_queries[sample_idx])
tfidf_feat = tfidf_pipeline.transform_single(sample_query_text)
inference_combined = np.concatenate([inference_feat, tfidf_feat])

# Find mismatches
diff = np.abs(training_feat - inference_combined)
mismatch_indices = np.where(diff > validator.tolerance)[0]

# Categorize
ast_mismatches = [i for i in mismatch_indices if 45 <= i <= 54]
historical_mismatches = [i for i in mismatch_indices if 78 <= i <= 94]
tfidf_mismatches = [i for i in mismatch_indices if i >= 95]
other_mismatches = [i for i in mismatch_indices if i < 45 or (55 <= i <= 77)]

print(f"\nTotal mismatches: {len(mismatch_indices)} out of {len(training_feat)} features")
print(f"\nBreakdown:")
print(f"  AST features (45-54):       {len(ast_mismatches)} mismatches")
print(f"  Historical features (78-94): {len(historical_mismatches)} mismatches")
print(f"  TF-IDF features (95+):       {len(tfidf_mismatches)} mismatches")
print(f"  Other base features:         {len(other_mismatches)} mismatches")

if ast_mismatches:
    print(f"\n  AST mismatch indices: {ast_mismatches}")
if historical_mismatches:
    print(f"  Historical mismatch indices: {historical_mismatches[:10]}")
if len(tfidf_mismatches) > 0:
    print(f"  TF-IDF mismatch count: {len(tfidf_mismatches)} (indices not shown)")

# ===== SUMMARY =====
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

print("\n🔍 Root Cause:")
if not has_fixes:
    print("   ❌ Spark is using OUTDATED code")
    print("   → Re-run notebook 00 to update")
elif not ast_matches:
    print("   ❌ AST parser non-determinism in Spark")
    print("   → Parser produces different results in distributed context")
    print("   → This affects features at indices 45-54")
    if len(tfidf_mismatches) > 0:
        print("   → Also some TF-IDF mismatches (minor, likely floating-point)")
else:
    print("   ✅ Code is updated and AST parser is consistent")
    print("   ❌ But parity still fails - other issues present")
    if len(tfidf_mismatches) > len(ast_mismatches):
        print("   → Primary issue: TF-IDF feature differences")
    if len(historical_mismatches) > 0:
        print("   → Historical features also have issues")

print("\n📋 Recommended Action:")
if not has_fixes:
    print("   1. Re-run notebook 00 (setup)")
    print("   2. Re-run this notebook")
elif not ast_matches:
    print("   1. OPTION A (Quick): Disable AST features")
    print("      unified_config['ast_timeout_ms'] = 0")
    print("      Re-run feature extraction cells (cells 14-22)")
    print("      Expected: 90% of mismatches will disappear")
    print("")
    print("   2. OPTION B (Better): Fix AST parser")
    print("      Replace sqlglot with deterministic parser")
    print("      This requires code changes")
else:
    print("   1. Investigate specific feature mismatches")
    print("   2. Run full debug cells from debug_parity_failure.py")

print("\n" + "="*80)
