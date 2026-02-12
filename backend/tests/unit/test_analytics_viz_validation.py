"""Unit tests for AnalyticsAgent visualization semantic validation."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from app.services.agents.analytics.agent import AnalyticsAgent


@pytest.fixture
def analytics_agent():
    """Create AnalyticsAgent instance."""
    return AnalyticsAgent()


@pytest.fixture
def sample_dataframes():
    """Create sample dataframes for testing."""
    return {
        "employment_data": pd.DataFrame(
            {
                "year": [2020, 2021, 2022, 2023],
                "employment_rate": [65.5, 66.2, 67.1, 68.0],
                "employment_count": [2500, 2600, 2700, 2800],
            }
        )
    }


@pytest.fixture
def sample_data_summary():
    """Create sample data summary."""
    return {
        "employment_data": {
            "row_count": 4,
            "columns": ["year", "employment_rate", "employment_count"],
            "numeric_columns": ["year", "employment_rate", "employment_count"],
            "metadata": {},
        }
    }


class TestVisualizationSemanticValidation:
    """Test suite for visualization semantic validation."""

    def test_valid_percentage_chart_0_to_100(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test validation passes for correctly formatted percentage chart (0-100 range)."""
        fig, ax = plt.subplots()
        ax.plot([2020, 2021, 2022, 2023], [65.5, 66.2, 67.1, 68.0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate (%)")
        ax.set_ylim(60, 70)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True
        assert result["feedback"] is None

    def test_invalid_percentage_chart_exceeds_100(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test validation fails when percentage values exceed 100."""
        fig, ax = plt.subplots()
        # Plot raw counts (2500-2800) but label as percentage
        ax.plot([2020, 2021, 2022, 2023], [2500, 2600, 2700, 2800])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate (%)")
        ax.set_ylim(2000, 3000)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is False
        assert "percentage" in result["feedback"].lower() or "rate" in result["feedback"].lower()
        assert (
            "2500" in result["feedback"] or "2800" in result["feedback"]
        )  # Should mention the actual values

    def test_invalid_percentage_chart_scaled_incorrectly(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test validation fails when percentage is scaled 0-300 instead of 0-100."""
        fig, ax = plt.subplots()
        # Values 0-200 labeled as percentage
        ax.bar([2020, 2021, 2022, 2023], [150, 160, 170, 180])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Percentage (%)")
        ax.set_ylim(0, 200)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is False
        assert "exceed 100" in result["feedback"] or "scaling" in result["feedback"].lower()

    def test_valid_ratio_0_to_1(self, analytics_agent, sample_dataframes, sample_data_summary):
        """Test validation passes for ratio data (0-1 range)."""
        fig, ax = plt.subplots()
        ax.plot([2020, 2021, 2022, 2023], [0.655, 0.662, 0.671, 0.680])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate")
        ax.set_ylim(0.6, 0.7)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True
        # May have warning about missing % symbol or suggesting ratio label

    def test_warning_ratio_mislabeled_as_percentage(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test warning when 0-1 ratio is labeled as percentage."""
        fig, ax = plt.subplots()
        ax.plot([2020, 2021, 2022, 2023], [0.655, 0.662, 0.671, 0.680])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Percentage (%)")
        ax.set_ylim(0.6, 0.7)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        # Should be valid but have warnings
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert any("0-1" in w or "ratio" in w.lower() for w in result["warnings"])

    def test_warning_missing_units(self, analytics_agent, sample_dataframes, sample_data_summary):
        """Test warning when Y-axis label missing units for large values."""
        fig, ax = plt.subplots()
        ax.bar([2020, 2021, 2022, 2023], [2500, 2600, 2700, 2800])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment")  # Missing units (should be "Thousands" or "Count")
        ax.set_ylim(0, 3000)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert any("units" in w.lower() or "thousands" in w.lower() for w in result["warnings"])

    def test_invalid_years_on_y_axis(self, analytics_agent, sample_dataframes, sample_data_summary):
        """Test validation warns when years appear on Y-axis in vertical chart (line/scatter)."""
        fig, ax = plt.subplots()
        # Line chart with swapped axes - years on Y, values on X (incorrect)
        ax.plot([65, 66, 67, 68], [2020, 2021, 2022, 2023])
        ax.set_xlabel("Employment Rate")
        ax.set_ylabel("Category")  # Not labeled as year but contains years
        ax.set_ylim(2019, 2024)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        # Should detect years on Y-axis and fail validation
        assert result["valid"] is False
        assert "year" in result["feedback"].lower()
        assert "x-axis" in result["feedback"].lower() or "swap" in result["feedback"].lower()

    def test_warning_negative_values_for_rate(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test warning when rate/percentage has negative values."""
        fig, ax = plt.subplots()
        ax.plot([2020, 2021, 2022, 2023], [65.5, -10.2, 67.1, 68.0])  # Negative value
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate (%)")
        ax.set_ylim(-20, 80)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        # Should be valid but have warnings
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert any("negative" in w.lower() for w in result["warnings"])

    def test_warning_very_large_values(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test warning for very large values that should be scaled."""
        fig, ax = plt.subplots()
        ax.bar([2020, 2021, 2022, 2023], [250000, 260000, 270000, 280000])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment")  # Missing "Thousands" or "Millions"
        ax.set_ylim(0, 300000)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert any("thousands" in w.lower() or "millions" in w.lower() for w in result["warnings"])

    def test_valid_chart_with_proper_units(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test validation passes for well-formatted chart with proper units."""
        fig, ax = plt.subplots()
        ax.bar([2020, 2021, 2022, 2023], [2500, 2600, 2700, 2800])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment (Thousands)")  # Proper units
        ax.set_ylim(0, 3000)

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True
        assert result["feedback"] is None
        assert len(result["warnings"]) == 0

    def test_empty_figure(self, analytics_agent, sample_dataframes, sample_data_summary):
        """Test validation handles empty figure gracefully."""
        fig = plt.figure()
        # No axes added

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "no axes" in result["warnings"][0].lower()

    def test_non_figure_input(self, analytics_agent, sample_dataframes, sample_data_summary):
        """Test validation handles non-Figure input gracefully."""
        result = analytics_agent._validate_visualization_semantics(
            fig="not a figure", dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        assert result["valid"] is True
        assert result["feedback"] is None

    def test_bar_chart_validation(self, analytics_agent, sample_dataframes, sample_data_summary):
        """Test validation works for bar charts."""
        fig, ax = plt.subplots()
        bars = ax.bar([2020, 2021, 2022, 2023], [65.5, 66.2, 67.1, 68.0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate (%)")

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True

    def test_line_chart_validation(self, analytics_agent, sample_dataframes, sample_data_summary):
        """Test validation works for line charts."""
        fig, ax = plt.subplots()
        ax.plot([2020, 2021, 2022, 2023], [65.5, 66.2, 67.1, 68.0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate (%)")

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        assert result["valid"] is True

    def test_outlier_detection_data_cleaning_issue(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test validation detects outlier bar (data cleaning issue like '63.7' becoming '637')."""
        fig, ax = plt.subplots()
        # Most values are 60-68, but one value is 637 (10x higher - likely "63.7" with trailing char)
        years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
        values = [65.5, 66.2, 67.1, 637.0, 68.0, 66.5, 67.0, 65.8]  # 2023 is outlier
        ax.bar(years, values)
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate (%)")
        ax.set_ylim(0, 700)  # Y-axis goes way beyond 100

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        # Should detect BOTH the axis limit issue AND the outlier
        assert result["valid"] is False
        assert "outlier" in result["feedback"].lower() or "637" in result["feedback"]

    def test_axis_limit_exceeds_100_for_percentage(
        self, analytics_agent, sample_dataframes, sample_data_summary
    ):
        """Test validation catches y-axis limits 0-300 even if data values are correct."""
        fig, ax = plt.subplots()
        # Data values are correct (60-70), but axis is set to 0-300
        ax.bar([2020, 2021, 2022, 2023], [65.5, 66.2, 67.1, 68.0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Employment Rate (%)")
        ax.set_ylim(0, 300)  # Wrong axis limits!

        result = analytics_agent._validate_visualization_semantics(
            fig=fig, dataframes=sample_dataframes, data_summary=sample_data_summary
        )

        plt.close(fig)

        # Should detect that axis limits exceed 100 for percentage
        assert result["valid"] is False
        assert "300" in result["feedback"] or "axis" in result["feedback"].lower()
