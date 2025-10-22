"""
Boundary-focused sampling for class-balanced training data.

Implements continuous distance-based sampling (POC v2.1.0) that prioritizes
queries near CPU/memory decision boundaries for better model training.
"""

import logging
from typing import Dict, Any, Tuple
import random

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    DataFrame = Any

logger = logging.getLogger(__name__)


class BoundarySampler:
    """
    Continuous distance-based sampling for class balancing.

    Key innovation (from POC v2.1.0):
    - Continuous sampling probability based on distance from boundary
    - Smooth gradient instead of discrete buckets
    - Guarantee threshold for 100% sampling of close queries
    - Safety adjustment to hit exact target ratio

    Algorithm:
        distance = min(cpu_distance, memory_distance)
        if distance < guarantee_threshold:
            sampling_prob = 1.0
        else:
            sampling_prob = max_boost - (max_boost - min_multiplier) * min(distance, 1.0)

    Example:
        sampler = BoundarySampler(config)
        df_sampled = sampler.sample(df_filtered, label_column='is_heavy')
        stats = sampler.get_sampling_stats(df_filtered)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize boundary sampler.

        Args:
            config: Configuration with boundary sampling parameters:
                - balance_ratio: Target ratio small:heavy (e.g., 5.0 for 5:1)
                - boundary_sampling_max_boost: Max multiplier at boundary (e.g., 2.0)
                - boundary_sampling_min_multiplier: Min multiplier for far queries (e.g., 0.05)
                - guarantee_close_threshold: Distance for 100% sampling (e.g., 0.5)
                - enable_safety_adjustment: Auto-adjust to hit exact ratio
                - cpu_threshold_seconds: CPU threshold for distance calculation
                - memory_threshold_gb: Memory threshold for distance calculation
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required for BoundarySampler")

        self.balance_ratio = config.get('balance_ratio', 5.0)
        self.max_boost = config.get('boundary_sampling_max_boost', 2.0)
        self.min_multiplier = config.get('boundary_sampling_min_multiplier', 0.05)
        self.guarantee_threshold = config.get('guarantee_close_threshold', 0.5)
        self.enable_safety_adjustment = config.get('enable_safety_adjustment', True)

        self.cpu_threshold = config.get('cpu_threshold_seconds', 300)
        self.memory_threshold = config.get('memory_threshold_gb', 10)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BoundarySampler initialized: "
                        f"ratio={self.balance_ratio}:1, "
                        f"max_boost={self.max_boost}, "
                        f"min_mult={self.min_multiplier}, "
                        f"guarantee_thresh={self.guarantee_threshold}")

    def calculate_distance(self, df: DataFrame) -> DataFrame:
        """
        Calculate normalized distance from decision boundary.

        Distance is the minimum of:
        - CPU distance: |cpu_time - threshold| / threshold
        - Memory distance: |memory_gb - threshold| / threshold

        Args:
            df: DataFrame with 'cpu_time_seconds' and 'memory_gb' columns

        Returns:
            DataFrame with 'boundary_distance' column (0 = on boundary, higher = farther)
        """
        # Calculate CPU distance (normalized)
        df = df.withColumn(
            'cpu_distance',
            F.abs(F.col('cpu_time_seconds') - F.lit(self.cpu_threshold)) / F.lit(self.cpu_threshold)
        )

        # Calculate memory distance (normalized)
        df = df.withColumn(
            'memory_distance',
            F.abs(F.col('memory_gb') - F.lit(self.memory_threshold)) / F.lit(self.memory_threshold)
        )

        # Take minimum distance (closest to either boundary)
        df = df.withColumn(
            'boundary_distance',
            F.least(F.col('cpu_distance'), F.col('memory_distance'))
        )

        return df

    def calculate_sampling_probability(self, df: DataFrame) -> DataFrame:
        """
        Calculate sampling probability based on distance.

        Continuous function:
        - distance < guarantee_threshold: probability = 1.0 (100% sampling)
        - distance >= guarantee_threshold: smooth decay from max_boost to min_multiplier

        Args:
            df: DataFrame with 'boundary_distance' column

        Returns:
            DataFrame with 'sampling_probability' column
        """
        # Apply guarantee threshold
        df = df.withColumn(
            'sampling_probability',
            F.when(
                F.col('boundary_distance') < self.guarantee_threshold,
                1.0  # 100% sampling for close queries
            ).otherwise(
                # Smooth decay: max_boost - (max_boost - min_mult) * min(distance, 1.0)
                self.max_boost - (self.max_boost - self.min_multiplier) *
                F.least(F.col('boundary_distance'), F.lit(1.0))
            )
        )

        return df

    def sample(
        self,
        df: DataFrame,
        label_column: str = 'is_heavy'
    ) -> DataFrame:
        """
        Apply boundary-focused sampling to achieve target ratio.

        Steps:
        1. Keep ALL heavy queries (no minority class loss)
        2. Calculate distance for small queries
        3. Apply continuous sampling probability
        4. Safety adjustment to hit exact target ratio (if enabled)

        Args:
            df: Input DataFrame with labels
            label_column: Column name for class labels (0=small, 1=heavy)

        Returns:
            Sampled DataFrame with target class ratio
        """
        self.logger.info(f"Starting boundary-focused sampling (target ratio: {self.balance_ratio}:1)")

        # Separate heavy and small queries
        df_heavy = df.filter(F.col(label_column) == 1)
        df_small = df.filter(F.col(label_column) == 0)

        heavy_count = df_heavy.count()
        small_count = df_small.count()
        initial_ratio = small_count / heavy_count if heavy_count > 0 else 0

        self.logger.info(f"Initial distribution: {small_count:,} small, {heavy_count:,} heavy "
                        f"(ratio: {initial_ratio:.2f}:1)")

        # Calculate target small query count
        target_small_count = int(heavy_count * self.balance_ratio)
        self.logger.info(f"Target small count: {target_small_count:,} (for {self.balance_ratio}:1 ratio)")

        # Calculate boundary distance and sampling probability for small queries
        df_small = self.calculate_distance(df_small)
        df_small = self.calculate_sampling_probability(df_small)

        # Calculate base fraction
        base_fraction = target_small_count / small_count if small_count > 0 else 1.0

        # Apply sampling probability (multiply by base fraction)
        df_small = df_small.withColumn(
            'final_sampling_prob',
            F.col('sampling_probability') * base_fraction
        )

        # Generate random number for sampling
        df_small = df_small.withColumn('random', F.rand(seed=42))

        # Sample based on probability
        df_small_sampled = df_small.filter(F.col('random') <= F.col('final_sampling_prob'))

        sampled_small_count = df_small_sampled.count()
        self.logger.info(f"After probability sampling: {sampled_small_count:,} small queries")

        # Safety adjustment (if enabled)
        if self.enable_safety_adjustment:
            df_small_sampled = self._apply_safety_adjustment(
                df_small,
                df_small_sampled,
                target_small_count,
                sampled_small_count
            )

        # Combine heavy (all) + sampled small
        df_sampled = df_heavy.unionByName(
            df_small_sampled.drop('cpu_distance', 'memory_distance', 'boundary_distance',
                                 'sampling_probability', 'final_sampling_prob', 'random'),
            allowMissingColumns=True
        )

        final_count = df_sampled.count()
        final_small = final_count - heavy_count
        final_ratio = final_small / heavy_count if heavy_count > 0 else 0

        self.logger.info(f"Sampling complete: {final_count:,} total queries")
        self.logger.info(f"Final distribution: {final_small:,} small, {heavy_count:,} heavy "
                        f"(ratio: {final_ratio:.2f}:1)")

        return df_sampled

    def _apply_safety_adjustment(
        self,
        df_small_all: DataFrame,
        df_small_sampled: DataFrame,
        target_count: int,
        current_count: int
    ) -> DataFrame:
        """
        Apply safety adjustment to hit exact target ratio.

        If under-sampled: add random samples from remainder
        If over-sampled: randomly drop excess samples

        Args:
            df_small_all: All small queries
            df_small_sampled: Currently sampled small queries
            target_count: Target count
            current_count: Current sampled count

        Returns:
            Adjusted DataFrame
        """
        diff = target_count - current_count

        if abs(diff) < target_count * 0.05:  # Within 5% tolerance
            self.logger.info(f"Sampling within tolerance: {current_count:,} vs {target_count:,} target")
            return df_small_sampled

        if diff > 0:  # Under-sampled, need to add more
            self.logger.info(f"Under-sampled by {diff:,}, adding random samples")
            # Get remainder (queries not yet sampled)
            df_remainder = df_small_all.join(
                df_small_sampled.select('queryId'),
                on='queryId',
                how='left_anti'
            )
            # Sample additional queries
            additional_fraction = diff / df_remainder.count() if df_remainder.count() > 0 else 0
            df_additional = df_remainder.sample(fraction=min(additional_fraction, 1.0), seed=43)
            df_small_sampled = df_small_sampled.unionByName(df_additional, allowMissingColumns=True)

        elif diff < 0:  # Over-sampled, need to remove some
            self.logger.info(f"Over-sampled by {-diff:,}, removing excess")
            # Randomly drop excess
            keep_fraction = target_count / current_count
            df_small_sampled = df_small_sampled.sample(fraction=keep_fraction, seed=44)

        adjustment_pct = abs(diff) / target_count * 100
        if adjustment_pct > 20:
            self.logger.warning(f"Large safety adjustment: {adjustment_pct:.1f}% "
                              "(consider tuning sampling parameters)")

        return df_small_sampled

    def get_sampling_stats(self, df: DataFrame) -> Dict[str, Any]:
        """
        Generate detailed sampling statistics.

        Args:
            df: DataFrame with boundary distance calculations

        Returns:
            Dictionary with distance distributions and sampling efficiency
        """
        # Calculate distances if not already present
        if 'boundary_distance' not in df.columns:
            df = self.calculate_distance(df)

        # Distance statistics
        stats = df.select('boundary_distance').summary('min', 'mean', '50%', '90%', 'max').collect()

        distance_stats = {
            'min': float(stats[0]['boundary_distance']),
            'mean': float(stats[1]['boundary_distance']),
            'median': float(stats[2]['boundary_distance']),
            'p90': float(stats[3]['boundary_distance']),
            'max': float(stats[4]['boundary_distance'])
        }

        # Sampling efficiency by distance ranges
        ranges = [
            ('very_close', 0.0, 0.2),
            ('close', 0.2, 0.5),
            ('moderate', 0.5, 1.0),
            ('far', 1.0, 2.0),
            ('very_far', 2.0, float('inf'))
        ]

        efficiency = {}
        for name, lower, upper in ranges:
            count = df.filter(
                (F.col('boundary_distance') >= lower) &
                (F.col('boundary_distance') < upper)
            ).count()
            efficiency[name] = count

        return {
            'distance_distribution': distance_stats,
            'sampling_efficiency': efficiency
        }
