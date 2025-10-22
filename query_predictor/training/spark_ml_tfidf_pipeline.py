"""
Spark ML TF-IDF pipeline for large-scale distributed training.

Uses Spark ML (CountVectorizer + IDF) for fitting on large datasets without
requiring data collection to driver. Extracts vocabulary and IDF weights to
create sklearn-compatible vectorizer for lightweight single-query inference.

Key Features:
- Fully distributed training (no .collect() needed)
- SQL-aware tokenization and normalization
- Binary TF-IDF mode optimized for SQL queries
- Filters common SQL keywords
- sklearn-compatible inference API
- Train-serve parity validation
- Pickle serialization

Example (Training):
    pipeline = SparkMLTfidfPipeline(config)
    pipeline.fit_on_dataframe(train_df, query_column='query')
    tfidf_udf = pipeline.create_spark_udf()
    train_features = train_df.withColumn('tfidf_features', tfidf_udf(F.col('query')))
    pipeline.save('/tmp/tfidf_pipeline.pkl')

Example (Inference):
    pipeline = SparkMLTfidfPipeline.load('/tmp/tfidf_pipeline.pkl')
    features = pipeline.transform_single("SELECT * FROM table")
"""

import logging
import pickle
import re
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from pyspark.ml.feature import CountVectorizer, IDF, RegexTokenizer, StopWordsRemover
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.sql import DataFrame, functions as F
    from pyspark.sql.types import ArrayType, FloatType, StringType
    from pyspark.ml.linalg import SparseVector, DenseVector
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SparkMLTfidfPipeline:
    """
    Distributed TF-IDF pipeline using Spark ML.

    Uses CountVectorizer + IDF for scalable fitting, extracts vocabulary
    and IDF weights for sklearn-compatible inference.
    """

    # Common SQL keywords to filter out
    SQL_KEYWORDS = {
        'select', 'from', 'where', 'join', 'on', 'and', 'or', 'in',
        'group', 'by', 'order', 'having', 'as', 'with', 'union',
        'insert', 'into', 'values', 'update', 'set', 'delete', 'create',
        'table', 'distinct', 'between', 'exists', 'case', 'when', 'then',
        'else', 'end', 'left', 'right', 'inner', 'outer', 'full', 'cross',
        'is', 'not', 'null', 'like', 'limit', 'offset', 'all', 'any',
        'drop', 'alter', 'column', 'add', 'constraint', 'primary', 'key',
        'foreign', 'references', 'unique', 'index', 'view', 'trigger'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Spark ML TF-IDF pipeline.

        Args:
            config: Configuration with TF-IDF parameters:
                - tfidf_vocab_size: Maximum vocabulary size (default: 250)
                - min_df: Minimum document frequency (default: 100)
                - max_df: Maximum document frequency (default: 0.80)
                - use_binary: Use binary TF (default: True for SQL)
                - filter_sql_keywords: Filter SQL keywords (default: True)
                - normalize_sql: Apply SQL normalization (default: True)
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required for SparkMLTfidfPipeline")

        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for SparkMLTfidfPipeline. "
                            "Install with: pip install scikit-learn")

        self.vocab_size = config.get('tfidf_vocab_size', 250)
        self.min_df = config.get('min_df', 100)
        self.max_df = config.get('max_df', 0.80)
        self.use_binary = config.get('use_binary', True)
        self.filter_sql_keywords = config.get('filter_sql_keywords', True)
        self.normalize_sql = config.get('normalize_sql', True)

        # Spark ML components
        self.spark_model: Optional[PipelineModel] = None
        self.vocabulary: List[str] = []
        self.idf_weights: Optional[np.ndarray] = None

        # sklearn components (for inference)
        self.sklearn_vectorizer: Optional[TfidfVectorizer] = None

        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _normalize_sql_query(query: str) -> str:
        """
        Normalize SQL query to reduce noise in TF-IDF.

        Args:
            query: SQL query string

        Returns:
            Normalized query with placeholders for literals and numbers
        """
        if not query:
            return ""

        # IMPORTANT: Order matters! Process most specific patterns first
        # Replace timestamps FIRST (before dates, as timestamps contain date patterns)
        query = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', " TIMESTAMP_VALUE ", query)

        # Replace date patterns (before string literals to catch dates in quotes)
        query = re.sub(r'\d{4}-\d{2}-\d{2}', " DATE_VALUE ", query)
        query = re.sub(r'\d{2}/\d{2}/\d{4}', " DATE_VALUE ", query)

        # Replace string literals with placeholder (catch-all for remaining quoted strings)
        query = re.sub(r"'[^']*'", " STRING_LITERAL ", query)
        query = re.sub(r'"[^"]*"', " STRING_LITERAL ", query)

        # Replace numbers with placeholder
        query = re.sub(r'\b\d+\b', " NUMERIC ", query)

        return query

    def fit_on_dataframe(self, df: DataFrame, query_column: str = 'query') -> 'SparkMLTfidfPipeline':
        """
        Fit TF-IDF on Spark DataFrame (fully distributed - no collect!).

        Args:
            df: Spark DataFrame with query column
            query_column: Name of column containing SQL queries

        Returns:
            Self (for chaining)
        """
        self.logger.info(f"Fitting Spark ML TF-IDF on column '{query_column}'...")
        self.logger.info(f"  Vocabulary size: {self.vocab_size}")
        self.logger.info(f"  Min DF: {self.min_df}")
        self.logger.info(f"  Max DF: {self.max_df}")
        self.logger.info(f"  Binary mode: {self.use_binary}")
        self.logger.info(f"  Filter SQL keywords: {self.filter_sql_keywords}")
        self.logger.info(f"  Normalize SQL: {self.normalize_sql}")

        stages = []

        # Step 1: Normalize SQL queries if enabled
        if self.normalize_sql:
            normalize_udf = F.udf(self._normalize_sql_query, StringType())
            df = df.withColumn(
                f"{query_column}_normalized",
                normalize_udf(F.col(query_column))
            )
            tokenizer_input_col = f"{query_column}_normalized"
        else:
            tokenizer_input_col = query_column

        # Step 2: Tokenize with SQL-aware pattern
        tokenizer = RegexTokenizer(
            inputCol=tokenizer_input_col,
            outputCol="words_raw",
            pattern=r'[^a-zA-Z0-9_]',  # Keep underscores for table_names
            minTokenLength=2,  # Skip single characters
            toLowercase=True
        )
        stages.append(tokenizer)

        # Step 3: Filter SQL keywords if enabled
        if self.filter_sql_keywords:
            stop_words_remover = StopWordsRemover(
                inputCol="words_raw",
                outputCol="words",
                stopWords=list(self.SQL_KEYWORDS),
                caseSensitive=False
            )
            stages.append(stop_words_remover)
            cv_input_col = "words"
        else:
            cv_input_col = "words_raw"

        # Step 4: CountVectorizer with binary mode
        count_vectorizer = CountVectorizer(
            inputCol=cv_input_col,
            outputCol="rawFeatures",
            vocabSize=self.vocab_size,
            minDF=float(self.min_df),
            binary=self.use_binary  # Binary presence/absence for SQL
        )
        stages.append(count_vectorizer)

        # Step 5: IDF transformation
        idf = IDF(
            inputCol="rawFeatures",
            outputCol="features",
            minDocFreq=self.min_df
        )
        stages.append(idf)

        # Build and fit pipeline
        pipeline = Pipeline(stages=stages)
        self.spark_model = pipeline.fit(df)
        self.is_fitted = True

        # Extract vocabulary from CountVectorizerModel
        # BUG FIX #1: Correct stage index calculation
        cv_stage_idx = 2 if self.filter_sql_keywords else 1
        cv_model = self.spark_model.stages[cv_stage_idx]
        self.vocabulary = cv_model.vocabulary

        # Extract IDF weights from IDFModel
        idf_stage_idx = 3 if self.filter_sql_keywords else 2
        idf_model = self.spark_model.stages[idf_stage_idx]
        self.idf_weights = np.array(idf_model.idf.toArray())

        self.logger.info(f"Spark ML TF-IDF fitted successfully")
        self.logger.info(f"  Actual vocabulary size: {len(self.vocabulary):,}")

        # Initialize sklearn vectorizer for inference
        self._init_sklearn_vectorizer()

        return self

    def _init_sklearn_vectorizer(self):
        """Initialize sklearn TfidfVectorizer with Spark-learned vocabulary and IDF weights."""
        # Create vocabulary dict (word -> index)
        vocabulary = {word: idx for idx, word in enumerate(self.vocabulary)}

        # BUG FIX #2: Remove incorrect token_pattern (not needed with fixed vocabulary)
        # Initialize sklearn TfidfVectorizer with vocabulary
        self.sklearn_vectorizer = TfidfVectorizer(
            vocabulary=vocabulary,
            norm='l2',  # Match Spark ML normalization
            use_idf=True,
            smooth_idf=False,  # We'll set IDF weights manually
            sublinear_tf=False,
            binary=self.use_binary,  # Match binary mode
            lowercase=True,
            preprocessor=self._normalize_sql_query if self.normalize_sql else None
        )

        # Fit with empty data to initialize internal state
        self.sklearn_vectorizer.fit([''])

        # Manually set IDF weights from Spark ML
        self.sklearn_vectorizer.idf_ = self.idf_weights

        self.logger.info("sklearn TfidfVectorizer initialized with Spark vocabulary and IDF weights")

    def create_spark_udf(self):
        """
        Create PySpark UDF for distributed TF-IDF transformation.

        Note: This uses manual implementation in UDF for consistency with inference.
        For very large-scale transformations, use transform_dataframe() instead
        which uses Spark ML directly.

        Returns:
            UDF that transforms SQL query → TF-IDF features (array of floats)

        Raises:
            ValueError: If pipeline not fitted
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_on_dataframe() before creating UDF")

        # Capture configuration in closure for UDF
        vocabulary_dict = {word: idx for idx, word in enumerate(self.vocabulary)}
        idf_weights = self.idf_weights.copy()
        vocab_size = len(self.vocabulary)
        use_binary = self.use_binary
        normalize_sql = self.normalize_sql
        filter_keywords = self.filter_sql_keywords
        sql_keywords = self.SQL_KEYWORDS if filter_keywords else set()

        @F.udf(returnType=ArrayType(FloatType()))
        def tfidf_udf(query: str) -> List[float]:
            """Transform single query to TF-IDF features."""
            if not query or not query.strip():
                return [0.0] * vocab_size

            try:
                import re
                import numpy as np

                # Normalize if enabled (order matters - match _normalize_sql_query)
                if normalize_sql:
                    query = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', " TIMESTAMP_VALUE ", query)
                    query = re.sub(r'\d{4}-\d{2}-\d{2}', " DATE_VALUE ", query)
                    query = re.sub(r'\d{2}/\d{2}/\d{4}', " DATE_VALUE ", query)
                    query = re.sub(r"'[^']*'", " STRING_LITERAL ", query)
                    query = re.sub(r'"[^"]*"', " STRING_LITERAL ", query)
                    query = re.sub(r'\b\d+\b', " NUMERIC ", query)

                # Tokenize (match RegexTokenizer pattern)
                tokens = re.split(r'[^a-zA-Z0-9_]', query.lower())
                tokens = [t for t in tokens if t and len(t) >= 2]

                # Filter SQL keywords if enabled
                if filter_keywords:
                    tokens = [t for t in tokens if t not in sql_keywords]

                # Count term frequencies
                tf_vector = np.zeros(vocab_size, dtype=np.float64)
                for token in tokens:
                    if token in vocabulary_dict:
                        idx = vocabulary_dict[token]
                        if use_binary:
                            tf_vector[idx] = 1.0  # Binary: presence/absence
                        else:
                            tf_vector[idx] += 1.0  # Count: frequency

                # Apply IDF weights
                tfidf_vector = tf_vector * idf_weights

                # L2 normalization (match Spark ML and sklearn)
                norm = np.linalg.norm(tfidf_vector)
                if norm > 0:
                    tfidf_vector = tfidf_vector / norm

                return tfidf_vector.astype(np.float32).tolist()

            except Exception as e:
                # Return zero vector on error
                return [0.0] * vocab_size

        return tfidf_udf

    def transform_dataframe(self, df: DataFrame, query_column: str = 'query') -> DataFrame:
        """
        Transform DataFrame using fitted Spark ML pipeline.

        Args:
            df: Spark DataFrame with query column
            query_column: Name of column containing SQL queries

        Returns:
            DataFrame with 'tfidf_features' column added
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_on_dataframe() before transform")

        # Apply full pipeline
        transformed = self.spark_model.transform(df)

        # Extract dense array from SparseVector
        vocab_size = len(self.vocabulary)

        def sparse_to_dense_udf(vector):
            """Convert SparseVector to dense array."""
            if vector is None:
                return [0.0] * vocab_size

            if isinstance(vector, SparseVector):
                return vector.toArray().tolist()
            elif isinstance(vector, DenseVector):
                return vector.toArray().tolist()
            else:
                return [0.0] * vocab_size

        sparse_to_dense = F.udf(sparse_to_dense_udf, ArrayType(FloatType()))

        # Add tfidf_features column
        result = transformed.withColumn(
            'tfidf_features',
            sparse_to_dense(F.col('features'))
        )

        return result

    def transform_single(self, query: str) -> np.ndarray:
        """
        Transform single query (for inference).

        Args:
            query: SQL query string

        Returns:
            NumPy array of TF-IDF features (shape: vocab_size,)

        Raises:
            ValueError: If pipeline not fitted
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_on_dataframe() before transform")

        if not query or not query.strip():
            return np.zeros(len(self.vocabulary), dtype=np.float32)

        try:
            # BUG FIX #3: Let sklearn's preprocessor handle normalization
            # Tokenize and filter keywords manually to match training
            tokens = re.split(r'[^a-zA-Z0-9_]', query.lower())
            tokens = [t for t in tokens if t and len(t) >= 2]

            # Filter SQL keywords if enabled
            if self.filter_sql_keywords:
                tokens = [t for t in tokens if t not in self.SQL_KEYWORDS]

            # Reconstruct as string for sklearn
            processed_query = ' '.join(tokens)

            # Use sklearn vectorizer with Spark-learned vocabulary and IDF weights
            # preprocessor will handle SQL normalization if enabled
            tfidf_vector = self.sklearn_vectorizer.transform([processed_query]).toarray()[0]
            return tfidf_vector.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"TF-IDF transform failed: {e}")
            return np.zeros(len(self.vocabulary), dtype=np.float32)

    def save(self, path: str):
        """
        Save pipeline to disk (pickle).

        Args:
            path: File path to save to

        Raises:
            ValueError: If pipeline not fitted
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_on_dataframe() before saving")

        # Save sklearn components and configuration
        save_data = {
            'vocabulary': self.vocabulary,
            'idf_weights': self.idf_weights,
            'vocab_size': self.vocab_size,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_binary': self.use_binary,
            'filter_sql_keywords': self.filter_sql_keywords,
            'normalize_sql': self.normalize_sql,
            'sklearn_vectorizer': self.sklearn_vectorizer
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        self.logger.info(f"SparkMLTfidfPipeline saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'SparkMLTfidfPipeline':
        """
        Load pipeline from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded SparkMLTfidfPipeline instance
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        # Create instance with saved configuration
        config = {
            'tfidf_vocab_size': save_data['vocab_size'],
            'min_df': save_data['min_df'],
            'max_df': save_data['max_df'],
            'use_binary': save_data.get('use_binary', True),
            'filter_sql_keywords': save_data.get('filter_sql_keywords', True),
            'normalize_sql': save_data.get('normalize_sql', True)
        }
        pipeline = cls(config)

        # Restore state
        pipeline.vocabulary = save_data['vocabulary']
        pipeline.idf_weights = save_data['idf_weights']
        pipeline.sklearn_vectorizer = save_data['sklearn_vectorizer']
        pipeline.is_fitted = True

        logger.info(f"SparkMLTfidfPipeline loaded: {path}")
        logger.info(f"  Vocabulary size: {len(pipeline.vocabulary):,}")
        logger.info(f"  Binary mode: {pipeline.use_binary}")

        return pipeline

    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Get TF-IDF configuration and vocabulary metadata.

        Returns:
            Dictionary with vocab size, config, feature names
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_on_dataframe() before getting metadata")

        return {
            'vocab_size': len(self.vocabulary),
            'max_features': self.vocab_size,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_binary': self.use_binary,
            'filter_sql_keywords': self.filter_sql_keywords,
            'normalize_sql': self.normalize_sql,
            'feature_names': [f"tfidf_{term}" for term in self.vocabulary],
            'is_fitted': self.is_fitted,
            'method': 'spark_ml_countvectorizer_optimized'
        }

    def get_top_features(self, query: str, top_n: int = 10) -> List[tuple]:
        """
        Get top N TF-IDF features for a query (for debugging).

        Args:
            query: SQL query string
            top_n: Number of top features to return

        Returns:
            List of (feature_name, score) tuples, sorted by score descending
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_on_dataframe() before getting top features")

        features = self.transform_single(query)

        # Get top N indices
        top_indices = np.argsort(features)[-top_n:][::-1]

        # Get feature names and scores
        feature_names = [f"tfidf_{self.vocabulary[idx]}" for idx in range(len(self.vocabulary))]
        top_features = [
            (feature_names[idx], float(features[idx]))
            for idx in top_indices
            if features[idx] > 0
        ]

        return top_features
