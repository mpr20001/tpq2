"""Unit tests for featurizer models."""

import pytest
from datetime import datetime, timezone
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.types.feature_group import FeatureGroup
from query_predictor.core.types.feature_spec import FeatureSpec
from query_predictor.core.types.ast_metrics import ASTMetrics


class TestFeatureGroup:
    """Test FeatureGroup dataclass."""

    def test_creation(self):
        group = FeatureGroup(
            name="test_group",
            feature_names=["f1", "f2", "f3"],
            enabled=True,
            description="Test group"
        )
        assert group.name == "test_group"
        assert len(group.feature_names) == 3
        assert group.enabled is True

    def test_disabled_group(self):
        group = FeatureGroup(
            name="disabled_group",
            feature_names=["f1"],
            enabled=False
        )
        assert group.enabled is False

    def test_to_dict(self):
        group = FeatureGroup(
            name="test_group",
            feature_names=["f1", "f2"],
            enabled=True
        )
        data = group.to_dict()
        assert data["name"] == "test_group"
        assert data["feature_names"] == ["f1", "f2"]
        assert data["enabled"] is True


class TestFeatureSpec:
    """Test FeatureSpec class."""

    def test_creation(self):
        spec = FeatureSpec(
            version="3.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_groups={
                "sql": FeatureGroup(
                    name="sql",
                    feature_names=["f1", "f2"],
                    enabled=True
                )
            },
            feature_importance={"f1": 0.8, "f2": 0.2},
            disabled_features=set(),
            importance_threshold=0.0
        )
        assert spec.version == "3.0.0"
        assert "sql" in spec.feature_groups
        assert spec.feature_count == 2

    def test_active_feature_names(self):
        spec = FeatureSpec(
            version="3.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_groups={
                "sql": FeatureGroup(
                    name="sql",
                    feature_names=["f1", "f2"],
                    enabled=True
                ),
                "disabled": FeatureGroup(
                    name="disabled",
                    feature_names=["f3"],
                    enabled=False
                )
            },
            feature_importance={},
            disabled_features={"f2"},
            importance_threshold=0.0
        )
        active = spec.active_feature_names
        assert "f1" in active
        assert "f2" not in active  # Explicitly disabled
        assert "f3" not in active  # Group disabled

    def test_feature_enable_disable(self):
        spec = FeatureSpec(
            version="3.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_groups={
                "sql": FeatureGroup(
                    name="sql",
                    feature_names=["f1", "f2"],
                    enabled=True
                )
            },
            feature_importance={},
            disabled_features={"f2"},
            importance_threshold=0.0
        )
        # Test that f1 is active (not in disabled_features)
        assert "f1" in spec.active_feature_names
        # Test that f2 is not active (in disabled_features)
        assert "f2" not in spec.active_feature_names
        # Test that f3 is not active (not in any group)
        assert "f3" not in spec.active_feature_names

    def test_disable_by_importance(self):
        spec = FeatureSpec(
            version="3.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_groups={
                "sql": FeatureGroup(
                    name="sql",
                    feature_names=["f1", "f2", "f3"],
                    enabled=True
                )
            },
            feature_importance={"f1": 0.9, "f2": 0.05, "f3": 0.01},
            disabled_features=set(),
            importance_threshold=0.1  # Disable features with importance < 0.1
        )
        # The importance threshold is applied in __post_init__
        assert "f1" in spec.active_feature_names
        assert "f2" not in spec.active_feature_names
        assert "f3" not in spec.active_feature_names

    def test_to_dict(self):
        spec = FeatureSpec(
            version="3.0.0",
            created_at="2024-01-01T00:00:00Z",
            feature_groups={
                "sql": FeatureGroup(
                    name="sql",
                    feature_names=["f1"],
                    enabled=True
                )
            },
            feature_importance={"f1": 0.8},
            disabled_features=set(),
            importance_threshold=0.0
        )
        data = spec.to_dict()
        assert data["version"] == "3.0.0"
        assert "feature_groups" in data
        assert "feature_importance" in data

    def test_from_dict(self):
        data = {
            "version": "3.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "feature_groups": {
                "sql": {
                    "name": "sql",
                    "feature_names": ["f1", "f2"],
                    "enabled": True,
                    "description": ""
                }
            },
            "feature_importance": {"f1": 0.8, "f2": 0.2},
            "disabled_features": [],
            "importance_threshold": 0.0
        }
        spec = FeatureSpec.from_dict(data)
        assert spec.version == "3.0.0"
        assert "sql" in spec.feature_groups
        assert spec.feature_count == 2


class TestQueryData:
    """Test QueryData dataclass."""

    def test_valid_creation(self):
        data = QueryData(
            query="SELECT * FROM table",
            user="test_user",
            catalog="hive",
            schema="default",
            hour=12
        )
        assert data.query == "SELECT * FROM table"
        assert data.user == "test_user"
        assert data.hour == 12

    def test_null_catalog_schema(self):
        data = QueryData(
            query="SELECT 1",
            user="test_user",
            catalog=None,
            schema=None,
            hour=12
        )
        assert data.catalog is None
        assert data.schema is None

    def test_empty_query_raises_error(self):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            QueryData(
                query="",
                user="test_user",
                hour=12
            )

    def test_invalid_hour_raises_error(self):
        with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
            QueryData(
                query="SELECT 1",
                user="test_user",
                hour=25
            )

    def test_to_dict(self):
        data = QueryData(
            query="SELECT 1",
            user="test_user",
            catalog="hive",
            schema="default",
            hour=12,
            client_info="cli",
            session_properties={"key": "value"}
        )
        result = data.to_dict()
        assert result["query"] == "SELECT 1"
        assert result["user"] == "test_user"
        assert result["session_properties"]["key"] == "value"


class TestASTMetrics:
    """Test ASTMetrics dataclass."""

    def test_creation(self):
        metrics = ASTMetrics(
            parse_success=True,
            parse_timeout=False,
            node_count=100,
            depth=5,
            cte_count=2,
            subquery_count=3,
            case_when_count=1,
            max_branches=4,
            union_count=0,
            union_all_count=1
        )
        assert metrics.parse_success is True
        assert metrics.node_count == 100
        assert metrics.depth == 5

    def test_failed_parse(self):
        metrics = ASTMetrics(
            parse_success=False,
            parse_timeout=False,
            node_count=0,
            depth=0,
            cte_count=0,
            subquery_count=0,
            case_when_count=0,
            max_branches=0,
            union_count=0,
            union_all_count=0
        )
        assert metrics.parse_success is False

    def test_timeout(self):
        metrics = ASTMetrics(
            parse_success=False,
            parse_timeout=True,
            node_count=0,
            depth=0,
            cte_count=0,
            subquery_count=0,
            case_when_count=0,
            max_branches=0,
            union_count=0,
            union_all_count=0
        )
        assert metrics.parse_timeout is True
