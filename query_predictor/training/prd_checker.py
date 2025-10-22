"""
PRD (Product Requirements Document) compliance checker.

Validates that trained model meets product requirements for heavy query detection.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """PRD compliance status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


@dataclass
class RequirementCheck:
    """Single requirement check result."""
    name: str
    target_value: float
    actual_value: float
    passed: bool
    status: ComplianceStatus
    message: str


@dataclass
class PRDComplianceReport:
    """Complete PRD compliance report."""
    overall_status: ComplianceStatus
    checks: List[RequirementCheck]
    summary: Dict[str, Any]
    timestamp: str


class PRDChecker:
    """
    Validate model against PRD requirements.

    Key Requirements:
    - Heavy query recall ≥98% (CRITICAL)
    - False negative rate ≤2% (CRITICAL)
    - F1 score ≥85% (TARGET)
    - ROC-AUC ≥90% (TARGET)

    Example:
        checker = PRDChecker(config)
        report = checker.check_compliance(test_metrics)
        if report.overall_status == ComplianceStatus.PASSED:
            print("Model meets PRD requirements")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PRD checker.

        Args:
            config: Configuration with prd_requirements:
                - target_heavy_recall: Minimum recall for heavy queries (e.g., 0.98)
                - target_fnr: Maximum false negative rate (e.g., 0.02)
                - target_f1: Minimum F1 score (e.g., 0.85)
                - target_roc_auc: Minimum ROC-AUC (e.g., 0.90)
        """
        self.config = config
        self.prd_reqs = config.get('prd_requirements', {})

        # Critical requirements (MUST pass)
        self.target_recall = self.prd_reqs.get('target_heavy_recall', 0.98)
        self.target_fnr = self.prd_reqs.get('target_fnr', 0.02)

        # Target requirements (SHOULD pass, but warnings only)
        self.target_f1 = self.prd_reqs.get('target_f1', 0.85)
        self.target_roc_auc = self.prd_reqs.get('target_roc_auc', 0.90)

        self.logger = logging.getLogger(__name__)

    def check_compliance(
        self,
        test_metrics: Dict[str, Any],
        strict: bool = True
    ) -> PRDComplianceReport:
        """
        Check if model meets PRD requirements.

        Args:
            test_metrics: Test set evaluation metrics with keys:
                - recall: Heavy query recall
                - fnr: False negative rate
                - f1: F1 score
                - roc_auc: ROC-AUC score
            strict: If True, fail on warnings; if False, warnings don't fail overall status

        Returns:
            PRDComplianceReport with detailed results
        """
        from datetime import datetime

        self.logger.info("Checking PRD compliance...")

        checks = []

        # CRITICAL: Heavy query recall
        recall = test_metrics.get('recall', 0.0)
        recall_check = self._check_requirement(
            name="Heavy Query Recall",
            target=self.target_recall,
            actual=recall,
            comparison=">=",
            critical=True
        )
        checks.append(recall_check)

        # CRITICAL: False negative rate
        fnr = test_metrics.get('fnr', 1.0)
        fnr_check = self._check_requirement(
            name="False Negative Rate",
            target=self.target_fnr,
            actual=fnr,
            comparison="<=",
            critical=True
        )
        checks.append(fnr_check)

        # TARGET: F1 score
        f1 = test_metrics.get('f1', 0.0)
        f1_check = self._check_requirement(
            name="F1 Score",
            target=self.target_f1,
            actual=f1,
            comparison=">=",
            critical=False
        )
        checks.append(f1_check)

        # TARGET: ROC-AUC
        roc_auc = test_metrics.get('roc_auc', 0.0)
        roc_auc_check = self._check_requirement(
            name="ROC-AUC",
            target=self.target_roc_auc,
            actual=roc_auc,
            comparison=">=",
            critical=False
        )
        checks.append(roc_auc_check)

        # Determine overall status
        critical_failures = [c for c in checks if not c.passed and c.status == ComplianceStatus.FAILED]
        warnings = [c for c in checks if not c.passed and c.status == ComplianceStatus.WARNING]

        if critical_failures:
            overall_status = ComplianceStatus.FAILED
        elif warnings and strict:
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.PASSED

        # Generate summary
        summary = {
            'total_checks': len(checks),
            'passed': len([c for c in checks if c.passed]),
            'failed': len(critical_failures),
            'warnings': len(warnings),
            'critical_requirements_met': len(critical_failures) == 0,
            'all_requirements_met': all(c.passed for c in checks)
        }

        report = PRDComplianceReport(
            overall_status=overall_status,
            checks=checks,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )

        # Log results
        self._log_report(report)

        return report

    def _check_requirement(
        self,
        name: str,
        target: float,
        actual: float,
        comparison: str,
        critical: bool
    ) -> RequirementCheck:
        """
        Check single requirement.

        Args:
            name: Requirement name
            target: Target value
            actual: Actual value
            comparison: Comparison operator (">=" or "<=")
            critical: If True, failure is FAILED; if False, failure is WARNING

        Returns:
            RequirementCheck result
        """
        if comparison == ">=":
            passed = actual >= target
            symbol = "≥"
        elif comparison == "<=":
            passed = actual <= target
            symbol = "≤"
        else:
            raise ValueError(f"Invalid comparison: {comparison}")

        status = ComplianceStatus.PASSED if passed else (
            ComplianceStatus.FAILED if critical else ComplianceStatus.WARNING
        )

        if passed:
            message = f"✅ {name}: {actual:.4f} {symbol} {target:.4f}"
        else:
            severity = "CRITICAL" if critical else "WARNING"
            message = f"{'❌' if critical else '⚠️'} {name}: {actual:.4f} {symbol} {target:.4f} [{severity}]"

        return RequirementCheck(
            name=name,
            target_value=target,
            actual_value=actual,
            passed=passed,
            status=status,
            message=message
        )

    def _log_report(self, report: PRDComplianceReport):
        """Log compliance report."""
        self.logger.info("\n" + "="*70)
        self.logger.info("PRD COMPLIANCE REPORT")
        self.logger.info("="*70)

        # Overall status
        status_symbol = {
            ComplianceStatus.PASSED: "✅",
            ComplianceStatus.FAILED: "❌",
            ComplianceStatus.WARNING: "⚠️"
        }

        self.logger.info(f"\nOverall Status: {status_symbol[report.overall_status]} {report.overall_status.value}")

        # Individual checks
        self.logger.info("\nRequirement Checks:")
        for check in report.checks:
            self.logger.info(f"  {check.message}")

        # Summary
        summary = report.summary
        self.logger.info(f"\nSummary:")
        self.logger.info(f"  Total checks: {summary['total_checks']}")
        self.logger.info(f"  Passed: {summary['passed']}")
        self.logger.info(f"  Failed: {summary['failed']}")
        self.logger.info(f"  Warnings: {summary['warnings']}")
        self.logger.info(f"  Critical requirements met: {summary['critical_requirements_met']}")

        self.logger.info("\n" + "="*70)

    def generate_report_text(self, report: PRDComplianceReport) -> str:
        """
        Generate human-readable text report.

        Args:
            report: PRD compliance report

        Returns:
            Formatted report string
        """
        status_symbol = {
            ComplianceStatus.PASSED: "✅",
            ComplianceStatus.FAILED: "❌",
            ComplianceStatus.WARNING: "⚠️"
        }

        lines = [
            "="*70,
            "PRD COMPLIANCE REPORT",
            "="*70,
            "",
            f"Overall Status: {status_symbol[report.overall_status]} {report.overall_status.value}",
            f"Timestamp: {report.timestamp}",
            "",
            "Requirement Checks:",
            ""
        ]

        for check in report.checks:
            lines.append(f"  {check.message}")

        summary = report.summary
        lines.extend([
            "",
            "Summary:",
            f"  Total checks: {summary['total_checks']}",
            f"  Passed: {summary['passed']}",
            f"  Failed: {summary['failed']}",
            f"  Warnings: {summary['warnings']}",
            f"  Critical requirements met: {summary['critical_requirements_met']}",
            f"  All requirements met: {summary['all_requirements_met']}",
            "",
            "="*70
        ])

        return "\n".join(lines)

    def get_requirements_dict(self) -> Dict[str, Any]:
        """
        Get configured PRD requirements.

        Returns:
            Dictionary with all PRD requirements
        """
        return {
            'critical': {
                'heavy_recall': {
                    'target': self.target_recall,
                    'comparison': '>=',
                    'description': 'Minimum recall for heavy queries'
                },
                'fnr': {
                    'target': self.target_fnr,
                    'comparison': '<=',
                    'description': 'Maximum false negative rate'
                }
            },
            'target': {
                'f1': {
                    'target': self.target_f1,
                    'comparison': '>=',
                    'description': 'Minimum F1 score'
                },
                'roc_auc': {
                    'target': self.target_roc_auc,
                    'comparison': '>=',
                    'description': 'Minimum ROC-AUC'
                }
            }
        }

    def validate_metrics_format(self, metrics: Dict[str, Any]) -> bool:
        """
        Validate that metrics dictionary has required keys.

        Args:
            metrics: Metrics dictionary to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        required_keys = ['recall', 'fnr', 'f1', 'roc_auc']

        missing_keys = [key for key in required_keys if key not in metrics]

        if missing_keys:
            raise ValueError(f"Metrics missing required keys: {missing_keys}")

        return True

    def export_report_json(self, report: PRDComplianceReport) -> Dict[str, Any]:
        """
        Export report as JSON-serializable dictionary.

        Args:
            report: PRD compliance report

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            'overall_status': report.overall_status.value,
            'timestamp': report.timestamp,
            'checks': [
                {
                    'name': check.name,
                    'target_value': check.target_value,
                    'actual_value': check.actual_value,
                    'passed': check.passed,
                    'status': check.status.value,
                    'message': check.message
                }
                for check in report.checks
            ],
            'summary': report.summary
        }
