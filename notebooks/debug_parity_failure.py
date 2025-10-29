"""
Debug Code for Feature Parity Investigation
============================================

Add these cells to notebook 03_feature_engineering_fixed.ipynb after cell 31
to perform deep debugging of the parity failure.

Each section can be run as a separate cell.
"""

# ==============================================================================
# CELL A: Verify Spark is Using Latest Code
# ==============================================================================
print("="*80)
print("CELL A: VERIFY SPARK CODE VERSION")
print("="*80)

# Check what version of code is in the Spark environment
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor
from query_predictor.core.featurizer.extractors.ast_extractor import ASTFeatureExtractor
from query_predictor.core.featurizer.parsers.ast_parser import ASTParser
from query_predictor.core.types.ast_metrics import ASTMetrics
import inspect

print("\n1. Feature Extractor Version:")
print(f"   VERSION: {FeatureExtractor.VERSION}")

print("\n2. AST Extractor Feature Names:")
ast_ext = ASTFeatureExtractor({'ast_timeout_ms': 200})
print(f"   Feature names: {ast_ext.get_feature_names()}")
print(f"   Feature count: {ast_ext.feature_count}")

print("\n3. ASTMetrics Fields:")
ast_metrics_fields = [f.name for f in ASTMetrics.__dataclass_fields__.values()]
print(f"   Fields: {ast_metrics_fields}")
print(f"   Has 'breadth': {'breadth' in ast_metrics_fields}")
print(f"   Has 'with_count': {'with_count' in ast_metrics_fields}")

print("\n4. AST Parser Code Snippet (check for hasattr fixes):")
parser_source = inspect.getsource(ASTParser._calculate_depth)
has_hasattr = 'hasattr(arg_value' in parser_source
has_sorted = 'sorted(node.args.keys())' in parser_source
print(f"   Uses hasattr() for Expression check: {has_hasattr}")
print(f"   Uses sorted() for deterministic traversal: {has_sorted}")

if not has_hasattr or not has_sorted:
    print("\n   ⚠️  WARNING: Code does NOT have latest fixes!")
    print("   The Spark environment may be using an outdated version.")
    print("   Check S3 zip file and re-upload if needed.")
else:
    print("\n   ✅ Code appears to have latest fixes")

print("\n5. Check S3 Zip File Timestamp:")
import boto3
from datetime import datetime

s3_client = boto3.client('s3')
zip_key = "sf_trino/trino_query_predictor/code/query_predictor_latest.zip"
try:
    response = s3_client.head_object(Bucket='uip-datalake-bucket-prod', Key=zip_key)
    last_modified = response['LastModified']
    print(f"   S3 zip file last modified: {last_modified}")

    # Check if it's recent (within last 24 hours)
    age_hours = (datetime.now(last_modified.tzinfo) - last_modified).total_seconds() / 3600
    print(f"   Age: {age_hours:.1f} hours ago")

    if age_hours > 24:
        print(f"   ⚠️  WARNING: Zip file is {age_hours:.1f} hours old!")
        print("   You may need to re-run notebook 00 to update the code package.")
except Exception as e:
    print(f"   Error checking S3 file: {e}")

print("\n" + "="*80)


# ==============================================================================
# CELL B: Create Debug UDF to Extract Raw Metrics
# ==============================================================================
print("="*80)
print("CELL B: CREATE DEBUG UDF FOR RAW METRICS")
print("="*80)

from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, BooleanType, IntegerType, StringType, FloatType

# Define schema for debug output
debug_schema = StructType([
    StructField("parse_success", BooleanType(), False),
    StructField("parse_timeout", BooleanType(), False),
    StructField("depth", IntegerType(), False),
    StructField("breadth", IntegerType(), False),
    StructField("node_count", IntegerType(), False),
    StructField("with_count", IntegerType(), False),
    StructField("cte_count", IntegerType(), False),
    StructField("lateral_view_count", IntegerType(), False),
    StructField("window_func_count", IntegerType(), False),
    StructField("distinct_count", IntegerType(), False),
    StructField("having_count", IntegerType(), False),
    StructField("case_when_count", IntegerType(), False),
    StructField("coalesce_null_if_count", IntegerType(), False),
    StructField("error_msg", StringType(), True),
])

@udf(returnType=debug_schema)
def debug_ast_parse(query):
    """Debug UDF to extract raw AST metrics."""
    if not query:
        return None

    try:
        from query_predictor.core.featurizer.parsers.ast_parser import ASTParser

        parser = ASTParser(timeout_ms=200)
        success, ast, metrics = parser.parse(query)

        return (
            metrics.parse_success,
            metrics.parse_timeout,
            metrics.depth,
            metrics.breadth,
            metrics.node_count,
            metrics.with_count,
            metrics.cte_count,
            metrics.lateral_view_count,
            metrics.window_func_count,
            metrics.distinct_count,
            metrics.having_count,
            metrics.case_when_count,
            metrics.coalesce_null_if_count,
            "" if success else "Parse failed"
        )
    except Exception as e:
        # Return zeros with error message
        return (False, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, str(e))

print("✅ Debug UDF created successfully")
print("\nThis UDF will extract raw AST metrics directly from Spark workers")
print("to compare with local parsing results.")

print("\n" + "="*80)


# ==============================================================================
# CELL C: Extract Single Query for Deep Analysis
# ==============================================================================
print("="*80)
print("CELL C: SINGLE QUERY DEEP DIVE")
print("="*80)

# Get the first sample query that had mismatches
sample_idx = 0
sample_query_data = sample_queries[sample_idx]
sample_query_text = sample_query_data['query']

print(f"\nAnalyzing Sample Query #{sample_idx}")
print("="*40)
print(f"Query preview: {sample_query_text[:200]}...")
print(f"User: {sample_query_data['user']}")
print(f"Catalog: {sample_query_data['catalog']}")
print(f"Schema: {sample_query_data['schema']}")

# ===== 1. Parse locally =====
print("\n" + "-"*40)
print("1. LOCAL PARSING (Inference Path)")
print("-"*40)

from query_predictor.core.featurizer.parsers.ast_parser import ASTParser

local_parser = ASTParser(timeout_ms=200)
local_success, local_ast, local_metrics = local_parser.parse(sample_query_text)

print(f"Parse success: {local_metrics.parse_success}")
print(f"Parse timeout: {local_metrics.parse_timeout}")
print(f"\nRaw AST Metrics (local):")
print(f"  depth:                  {local_metrics.depth}")
print(f"  breadth:                {local_metrics.breadth}")
print(f"  node_count:             {local_metrics.node_count}")
print(f"  with_count:             {local_metrics.with_count}")
print(f"  cte_count:              {local_metrics.cte_count}")
print(f"  lateral_view_count:     {local_metrics.lateral_view_count}")
print(f"  window_func_count:      {local_metrics.window_func_count}")
print(f"  distinct_count:         {local_metrics.distinct_count}")
print(f"  having_count:           {local_metrics.having_count}")
print(f"  case_when_count:        {local_metrics.case_when_count}")
print(f"  coalesce_null_if_count: {local_metrics.coalesce_null_if_count}")

# ===== 2. Extract from Spark =====
print("\n" + "-"*40)
print("2. SPARK PARSING (Training Path)")
print("-"*40)

# Find this query in the training data
from pyspark.sql import functions as F

# Create a DataFrame with just this query
debug_df = spark.createDataFrame([(sample_query_text,)], ["query"])
debug_df = debug_df.withColumn("ast_metrics", debug_ast_parse(F.col("query")))

# Collect the result
spark_metrics = debug_df.select("ast_metrics.*").collect()[0]

print(f"Parse success: {spark_metrics.parse_success}")
print(f"Parse timeout: {spark_metrics.parse_timeout}")
if spark_metrics.error_msg:
    print(f"Error: {spark_metrics.error_msg}")

print(f"\nRaw AST Metrics (Spark):")
print(f"  depth:                  {spark_metrics.depth}")
print(f"  breadth:                {spark_metrics.breadth}")
print(f"  node_count:             {spark_metrics.node_count}")
print(f"  with_count:             {spark_metrics.with_count}")
print(f"  cte_count:              {spark_metrics.cte_count}")
print(f"  lateral_view_count:     {spark_metrics.lateral_view_count}")
print(f"  window_func_count:      {spark_metrics.window_func_count}")
print(f"  distinct_count:         {spark_metrics.distinct_count}")
print(f"  having_count:           {spark_metrics.having_count}")
print(f"  case_when_count:        {spark_metrics.case_when_count}")
print(f"  coalesce_null_if_count: {spark_metrics.coalesce_null_if_count}")

# ===== 3. Compare =====
print("\n" + "-"*40)
print("3. COMPARISON")
print("-"*40)

metrics_to_compare = [
    'depth', 'breadth', 'node_count', 'with_count', 'cte_count',
    'lateral_view_count', 'window_func_count', 'distinct_count',
    'having_count', 'case_when_count', 'coalesce_null_if_count'
]

print(f"{'Metric':<25} {'Local':<10} {'Spark':<10} {'Match':<10}")
print("-" * 60)

all_match = True
for metric in metrics_to_compare:
    local_val = getattr(local_metrics, metric)
    spark_val = getattr(spark_metrics, metric)
    match = local_val == spark_val
    if not match:
        all_match = False

    status = "✅ MATCH" if match else "❌ DIFF"
    print(f"{metric:<25} {local_val:<10} {spark_val:<10} {status}")

if all_match:
    print("\n✅ ALL AST METRICS MATCH!")
    print("The parity issue is NOT in the AST parser.")
else:
    print("\n❌ AST METRICS DIFFER BETWEEN LOCAL AND SPARK")
    print("This confirms the parser behaves differently in distributed context.")

print("\n" + "="*80)


# ==============================================================================
# CELL D: Extract Full Feature Vectors for Comparison
# ==============================================================================
print("="*80)
print("CELL D: FULL FEATURE VECTOR COMPARISON")
print("="*80)

# Get training features for this sample
train_features_sample0 = training_features[sample_idx]

# Get inference features for this sample
inference_base = inference_featurizer.extract(sample_query_data)
inference_tfidf = tfidf_pipeline.transform_single(sample_query_text)
inference_combined = np.concatenate([inference_base, inference_tfidf])

print(f"\nSample #{sample_idx} Feature Dimensions:")
print(f"  Training features:  {len(train_features_sample0)}")
print(f"  Inference base:     {len(inference_base)}")
print(f"  Inference TF-IDF:   {len(inference_tfidf)}")
print(f"  Inference combined: {len(inference_combined)}")

# Feature names
all_feature_names = (
    unified_extractor.get_feature_names() +
    [f"tfidf_{i}" for i in range(config['features']['tfidf_vocab_size'])]
)

# Create detailed comparison table
print(f"\n{'Idx':<5} {'Feature Name':<35} {'Training':<15} {'Inference':<15} {'Diff':<15} {'Status':<10}")
print("="*100)

mismatches = []
for i in range(len(train_features_sample0)):
    feat_name = all_feature_names[i] if i < len(all_feature_names) else f"feature_{i}"
    train_val = train_features_sample0[i]
    inf_val = inference_combined[i]
    diff = abs(train_val - inf_val)

    if diff > validator.tolerance:
        status = "❌ MISMATCH"
        mismatches.append((i, feat_name, train_val, inf_val, diff))
        print(f"{i:<5} {feat_name:<35} {train_val:<15.9f} {inf_val:<15.9f} {diff:<15.9f} {status}")

print(f"\n{'='*100}")
print(f"Total mismatches: {len(mismatches)}")

# Group mismatches by feature type
ast_mismatches = [m for m in mismatches if 45 <= m[0] <= 54]
historical_mismatches = [m for m in mismatches if 78 <= m[0] <= 94]
tfidf_mismatches = [m for m in mismatches if m[0] >= 95]
other_mismatches = [m for m in mismatches if m[0] < 45 or (55 <= m[0] <= 77)]

print(f"\nMismatch Breakdown:")
print(f"  AST features (45-54):       {len(ast_mismatches)}")
print(f"  Historical features (78-94): {len(historical_mismatches)}")
print(f"  TF-IDF features (95+):       {len(tfidf_mismatches)}")
print(f"  Other base features:         {len(other_mismatches)}")

if ast_mismatches:
    print(f"\n❌ AST Feature Mismatches:")
    for idx, name, train, inf, diff in ast_mismatches:
        print(f"   [{idx}] {name}: {train:.6f} (train) vs {inf:.6f} (inf), diff={diff:.6f}")

if historical_mismatches:
    print(f"\n❌ Historical Feature Mismatches:")
    for idx, name, train, inf, diff in historical_mismatches[:5]:
        print(f"   [{idx}] {name}: {train:.6f} (train) vs {inf:.6f} (inf), diff={diff:.6f}")

if tfidf_mismatches:
    print(f"\n❌ TF-IDF Feature Mismatches (first 10):")
    for idx, name, train, inf, diff in tfidf_mismatches[:10]:
        print(f"   [{idx}] {name}: {train:.6f} (train) vs {inf:.6f} (inf), diff={diff:.6f}")

print("\n" + "="*80)


# ==============================================================================
# CELL E: Test Multiple Queries to Identify Patterns
# ==============================================================================
print("="*80)
print("CELL E: PATTERN ANALYSIS ACROSS MULTIPLE QUERIES")
print("="*80)

print("\nTesting 10 sample queries to identify consistency patterns...")

# Test 10 queries
num_test_queries = min(10, len(sample_queries))
test_queries_text = [sample_queries[i]['query'] for i in range(num_test_queries)]

# Create DataFrame with test queries
test_df = spark.createDataFrame([(q,) for q in test_queries_text], ["query"])
test_df = test_df.withColumn("ast_metrics", debug_ast_parse(F.col("query")))

# Collect all Spark results
spark_results = test_df.select("ast_metrics.*").collect()

# Parse locally
local_results = []
for query_text in test_queries_text:
    parser = ASTParser(timeout_ms=200)
    success, ast, metrics = parser.parse(query_text)
    local_results.append(metrics)

# Compare patterns
print(f"\n{'Query':<8} {'Metric':<25} {'Local':<10} {'Spark':<10} {'Match':<10}")
print("="*70)

pattern_summary = {metric: {'matches': 0, 'mismatches': 0} for metric in metrics_to_compare}

for i in range(num_test_queries):
    local_m = local_results[i]
    spark_m = spark_results[i]

    # Check each metric
    for metric in ['depth', 'breadth', 'cte_count', 'window_func_count']:
        local_val = getattr(local_m, metric)
        spark_val = getattr(spark_m, metric)
        match = local_val == spark_val

        if match:
            pattern_summary[metric]['matches'] += 1
        else:
            pattern_summary[metric]['mismatches'] += 1
            status = "❌ DIFF"
            print(f"Q{i:<7} {metric:<25} {local_val:<10} {spark_val:<10} {status}")

# Summary
print(f"\n{'='*70}")
print(f"\nPattern Summary Across {num_test_queries} Queries:")
print(f"{'Metric':<25} {'Matches':<10} {'Mismatches':<12} {'Match Rate':<12}")
print("-"*70)

for metric in metrics_to_compare:
    matches = pattern_summary[metric]['matches']
    mismatches = pattern_summary[metric]['mismatches']
    match_rate = (matches / num_test_queries) * 100
    print(f"{metric:<25} {matches:<10} {mismatches:<12} {match_rate:.1f}%")

print("\n" + "="*80)


# ==============================================================================
# CELL F: Recommendations Based on Findings
# ==============================================================================
print("="*80)
print("CELL F: DIAGNOSTIC SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n📊 DIAGNOSTIC RESULTS:")
print("-"*40)

# Check if code is outdated
code_is_latest = has_hasattr and has_sorted  # from Cell A

# Check parser consistency
parser_is_consistent = all_match  # from Cell C

# Count mismatch types
total_mismatches = len(mismatches)
ast_pct = (len(ast_mismatches) / total_mismatches * 100) if total_mismatches > 0 else 0
tfidf_pct = (len(tfidf_mismatches) / total_mismatches * 100) if total_mismatches > 0 else 0

print(f"1. Code Version in Spark: {'✅ Latest' if code_is_latest else '❌ Outdated'}")
print(f"2. AST Parser Consistency: {'❌ Inconsistent' if not parser_is_consistent else '✅ Consistent'}")
print(f"3. Mismatch Distribution:")
print(f"   - AST features:        {ast_pct:.1f}%")
print(f"   - TF-IDF features:     {tfidf_pct:.1f}%")
print(f"   - Other features:      {100 - ast_pct - tfidf_pct:.1f}%")

print("\n🔍 ROOT CAUSE ANALYSIS:")
print("-"*40)

if not code_is_latest:
    print("❌ PRIMARY ISSUE: Spark is using OUTDATED code!")
    print("   → Action: Re-run notebook 00 to rebuild and upload code package")
    print("   → Then re-run this notebook with updated code")

elif not parser_is_consistent:
    print("❌ PRIMARY ISSUE: AST Parser non-determinism in Spark")
    print("   → The parser produces different results in Spark vs local context")
    print("   → This is likely due to:")
    print("      1. Different Python/sqlglot environment in workers")
    print("      2. Threading/timeout behavior differences")
    print("      3. Memory constraints affecting parse results")

    print("\n📋 RECOMMENDED SOLUTIONS (in order of preference):")
    print("   1. DISABLE AST FEATURES for now")
    print("      - Modify config: set_disable_ast_features=True")
    print("      - This ensures parity at cost of ~5-10% model performance")
    print("      - Features will be 335 instead of 345")

    print("\n   2. PRE-COMPUTE AST FEATURES in separate pass")
    print("      - Add AST features to dataset before training")
    print("      - Use same pre-computed values in inference")
    print("      - Requires pipeline changes")

    print("\n   3. REPLACE sqlglot with simpler parser")
    print("      - Use regex-based fallback features only")
    print("      - Already implemented in _fallback_features()")
    print("      - Less accurate but deterministic")

else:
    print("✅ Code is up-to-date and parser is consistent")
    print("❌ But parity still fails - investigating other causes...")

    if len(tfidf_mismatches) > len(ast_mismatches):
        print("   → Primary issue appears to be TF-IDF features")
        print("   → Check Spark ML vs sklearn differences")

print("\n" + "="*80)
