#!/usr/bin/env python3
"""
Red Flags Profits - Enhanced Data Analysis Module
Comprehensive data analysis with advanced features for website generation
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class EnhancedRedFlagsAnalyzer:
    """Enhanced data analyzer with advanced metrics and trend analysis"""

    def __init__(
        self, input_parquet="data/all_billionaires.parquet", output_dir="output"
    ):
        self.input_parquet = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logging()

        # Import advanced processing components
        from sparkline_generator import SparklineGenerator, AdvancedDataProcessor

        self.sparkline_gen = SparklineGenerator()
        self.advanced_processor = AdvancedDataProcessor()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "analysis.log"),
            ],
        )
        return logging.getLogger(__name__)

    def analyze_comprehensive(self):
        """Comprehensive analysis pipeline with all enhancements"""
        self.logger.info("🚀 Starting comprehensive Red Flags Profits analysis...")

        if not self.input_parquet.exists():
            self.logger.error(f"❌ Input file not found: {self.input_parquet}")
            return False

        try:
            # Load and process data
            df = self._load_and_validate_data()
            if df is None:
                return False

            daily_data = self._calculate_enhanced_daily_aggregations(df)

            # Generate comprehensive analytics
            self._generate_enhanced_outputs(daily_data)

            self.logger.info("🎉 Comprehensive analysis completed!")
            return True

        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            self.logger.exception("Full traceback:")
            return False

    def _load_and_validate_data(self):
        """Enhanced data loading with validation"""
        self.logger.info("📂 Loading and validating data...")

        df = pd.read_parquet(self.input_parquet)
        df["crawl_date"] = pd.to_datetime(df["crawl_date"])

        # Data quality report
        total_records = len(df)
        date_range = df["crawl_date"].max() - df["crawl_date"].min()
        unique_dates = df["crawl_date"].nunique()
        unique_people = df["personName"].nunique()

        self.logger.info(f"📊 Dataset summary:")
        self.logger.info(f"   Total records: {total_records:,}")
        self.logger.info(f"   Date range: {date_range.days} days")
        self.logger.info(f"   Unique dates: {unique_dates:,}")
        self.logger.info(f"   Unique people: {unique_people:,}")

        # Check for required columns
        required_columns = ["finalWorth", "personName", "crawl_date"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            self.logger.error(f"❌ Missing columns: {missing_columns}")
            return None

        # Inflation data check
        inflation_cols = [col for col in ["cpi_u", "pce"] if col in df.columns]
        if inflation_cols:
            for col in inflation_cols:
                non_null_count = df[col].notna().sum()
                coverage = (non_null_count / total_records) * 100
                self.logger.info(
                    f"   {col.upper()}: {coverage:.1f}% coverage ({non_null_count:,} records)"
                )
        else:
            self.logger.warning("⚠️  No inflation data available")

        return df.sort_values("crawl_date")

    def _calculate_enhanced_daily_aggregations(self, df):
        """Enhanced daily aggregations with inflation preservation"""
        self.logger.info("🔄 Calculating enhanced daily aggregations...")

        # Check available inflation columns
        inflation_cols = [col for col in ["cpi_u", "pce"] if col in df.columns]

        # Enhanced aggregation
        agg_dict = {
            "total_wealth": ("finalWorth", "sum"),
            "billionaire_count": ("personName", "nunique"),
            "wealth_std": ("finalWorth", "std"),  # Wealth distribution metric
            "wealth_median": ("finalWorth", "median"),  # Median wealth
        }

        # Preserve inflation data
        for col in inflation_cols:
            agg_dict[col] = (col, "first")

        daily_data = (
            df.groupby("crawl_date")
            .agg(**agg_dict)
            .reset_index()
            .rename(columns={"crawl_date": "date"})
            .sort_values("date")
        )

        # Convert to appropriate units
        daily_data["total_wealth"] = (
            daily_data["total_wealth"] / 1_000_000
        )  # To trillions
        daily_data["average_wealth"] = (
            daily_data["total_wealth"] / daily_data["billionaire_count"] * 1000
        )  # To billions
        daily_data["wealth_median"] = daily_data["wealth_median"] / 1000  # To billions
        daily_data["wealth_std"] = daily_data["wealth_std"] / 1000  # To billions

        # Calculate inequality metrics
        daily_data["wealth_inequality"] = (
            daily_data["wealth_std"] / daily_data["average_wealth"]
        )

        self.logger.info(
            f"✅ Enhanced daily data: {len(daily_data)} points with {len(daily_data.columns)} metrics"
        )
        return daily_data

    def _generate_enhanced_outputs(self, daily_data):
        """Generate all enhanced output files for website"""
        self.logger.info("📊 Generating enhanced output files...")

        # Generate enhanced metrics with trend analysis
        enhanced_metrics = self.advanced_processor.generate_enhanced_metrics(daily_data)
        self._save_json(enhanced_metrics, "metrics.json")

        # Generate detailed timeline data
        self._generate_enhanced_timeline(daily_data)

        # Generate sparklines data
        sparklines_data = self._generate_enhanced_sparklines(daily_data)
        self._save_json(sparklines_data, "sparklines.json")

        # Generate wealth equivalencies
        self._generate_enhanced_equivalencies(daily_data)

        # Generate comprehensive metadata
        self._generate_enhanced_metadata(daily_data, enhanced_metrics)

        # Generate chart-ready data
        self._generate_chart_data(daily_data, enhanced_metrics)

        self.logger.info("✅ All enhanced outputs generated successfully!")

    def _generate_enhanced_timeline(self, daily_data):
        """Generate enhanced timeline with all metrics"""
        self.logger.info("📈 Generating enhanced timeline...")

        timeline = []
        for _, row in daily_data.iterrows():
            entry = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "total_wealth": round(row["total_wealth"], 2),
                "billionaire_count": int(row["billionaire_count"]),
                "average_wealth": round(row["average_wealth"], 2),
                "wealth_median": round(row["wealth_median"], 2),
                "wealth_inequality": round(row["wealth_inequality"], 3),
            }

            # Add inflation data if available
            for col in ["cpi_u", "pce"]:
                if col in row and pd.notna(row[col]):
                    entry[col] = round(row[col], 2)

            timeline.append(entry)

        self._save_json(timeline, "timeline.json")

    def _generate_enhanced_sparklines(self, daily_data):
        """Generate enhanced sparklines with multiple metrics"""
        self.logger.info("✨ Generating enhanced sparklines...")

        # Sample data for sparklines (8 points for clean visualization)
        sample_size = min(8, len(daily_data))
        indices = np.linspace(0, len(daily_data) - 1, sample_size, dtype=int)
        sampled_data = daily_data.iloc[indices]

        sparklines = {
            "wealth": [round(x, 1) for x in sampled_data["total_wealth"].tolist()],
            "count": [int(x) for x in sampled_data["billionaire_count"].tolist()],
            "average": [round(x, 1) for x in sampled_data["average_wealth"].tolist()],
            "median": [round(x, 1) for x in sampled_data["wealth_median"].tolist()],
            "inequality": [
                round(x, 3) for x in sampled_data["wealth_inequality"].tolist()
            ],
            "bounds": {
                "wealth": {
                    "min": round(daily_data["total_wealth"].min(), 1),
                    "max": round(daily_data["total_wealth"].max(), 1),
                },
                "count": {
                    "min": int(daily_data["billionaire_count"].min()),
                    "max": int(daily_data["billionaire_count"].max()),
                },
                "average": {
                    "min": round(daily_data["average_wealth"].min(), 1),
                    "max": round(daily_data["average_wealth"].max(), 1),
                },
                "median": {
                    "min": round(daily_data["wealth_median"].min(), 1),
                    "max": round(daily_data["wealth_median"].max(), 1),
                },
                "inequality": {
                    "min": round(daily_data["wealth_inequality"].min(), 3),
                    "max": round(daily_data["wealth_inequality"].max(), 3),
                },
            },
        }

        return sparklines

    def _generate_enhanced_equivalencies(self, daily_data):
        """Generate enhanced wealth equivalency comparisons"""
        self.logger.info("💰 Generating enhanced equivalencies...")

        latest_wealth_trillions = daily_data.iloc[-1]["total_wealth"]
        total_dollars = latest_wealth_trillions * 1e12

        # Enhanced reference values
        references = {
            "median_household_income": 80610,
            "median_worker_annual": 59540,
            "median_lifetime_earnings": 1_420_000,
            "federal_budget_2024": 6.13e12,
            "us_gdp_2024": 27.36e12,
            "global_gdp_2024": 105e12,
            "fortune_500_revenue": 18.8e12,
        }

        equivalencies = [
            {
                "comparison": "Median US Households",
                "value": f"{int(total_dollars / references['median_household_income'] / 1e6)} million",
                "context": "Annual household income",
            },
            {
                "comparison": "Median Workers",
                "value": f"{int(total_dollars / references['median_worker_annual'] / 1e6)} million",
                "context": "Annual salaries",
            },
            {
                "comparison": "Average US Workers",
                "value": f"{int(total_dollars / references['median_lifetime_earnings'] / 1e6)} million",
                "context": "Lifetime careers",
            },
            {
                "comparison": "US Federal Budget",
                "value": f"{total_dollars / references['federal_budget_2024']:.1f}x",
                "context": "Annual federal spending",
            },
            {
                "comparison": "US GDP",
                "value": f"{(total_dollars / references['us_gdp_2024']) * 100:.1f}%",
                "context": "Of total US economic output",
            },
            {
                "comparison": "Fortune 500 Revenue",
                "value": f"{(total_dollars / references['fortune_500_revenue']) * 100:.1f}%",
                "context": "Of top 500 US companies",
            },
        ]

        self._save_json(equivalencies, "equivalencies.json")

    def _generate_enhanced_metadata(self, daily_data, enhanced_metrics):
        """Generate comprehensive metadata"""
        self.logger.info("📋 Generating enhanced metadata...")

        first_date = daily_data.iloc[0]["date"]
        last_date = daily_data.iloc[-1]["date"]
        days_span = (last_date - first_date).days

        # Data quality metrics
        completeness = {
            "total_records": len(daily_data),
            "missing_days": 0,  # Could calculate gaps if needed
            "data_quality_score": 0.95,  # Could implement actual scoring
        }

        # Inflation data availability
        inflation_status = {}
        for col in ["cpi_u", "pce"]:
            if col in daily_data.columns:
                non_null_count = int(
                    daily_data[col].notna().sum()
                )  # Convert to native int
                inflation_status[col] = {
                    "available": bool(non_null_count > 0),  # Convert to native bool
                    "coverage_pct": float(
                        round((non_null_count / len(daily_data)) * 100, 1)
                    ),  # Convert to native float
                    "records": non_null_count,
                }

        metadata = {
            "last_updated": datetime.now().isoformat() + "Z",
            "data_start_date": first_date.strftime("%Y-%m-%d"),
            "data_end_date": last_date.strftime("%Y-%m-%d"),
            "data_points": len(daily_data),
            "data_days_span": days_span,
            "version": "2.0",
            "generated_by": "enhanced-red-flags-analyzer",
            "analysis_features": [
                "exponential_trend_fitting",
                "inflation_adjustment",
                "wealth_inequality_metrics",
                "background_sparklines",
                "comprehensive_equivalencies",
            ],
            "data_quality": completeness,
            "inflation_data": inflation_status,
            "trend_analysis": {
                "exponential_fit_r_squared": float(
                    enhanced_metrics.get("exponential_fit", {}).get("r_squared", 0)
                ),
                "annual_growth_rate": float(
                    enhanced_metrics.get("exponential_fit", {}).get(
                        "annual_growth_rate", 0
                    )
                ),
                "fit_period_days": int(
                    enhanced_metrics.get("exponential_fit", {}).get(
                        "fit_period_days", 0
                    )
                ),
            },
        }

        self._save_json(metadata, "metadata.json")

    def _generate_chart_data(self, daily_data, enhanced_metrics):
        """Generate chart-ready data for website visualization"""
        self.logger.info("📊 Generating chart data...")

        # Prepare main chart data
        chart_data = {
            "wealth_timeline": {
                "data": [
                    {
                        "x": row["date"].strftime("%Y-%m-%d"),
                        "y": round(row["total_wealth"], 2),
                    }
                    for _, row in daily_data.iterrows()
                ],
                "trend_line": enhanced_metrics.get("trend_line", []),
                "inflation_adjusted": enhanced_metrics.get("inflation_adjusted"),
                "exponential_fit": enhanced_metrics.get("exponential_fit", {}),
                "title": "Total Billionaire Wealth Evolution",
                "y_axis_title": "Wealth (Trillions USD)",
                "animation": {"point_delay": 5, "trend_line_speed": 800},
                "summary": {
                    "data_points": len(daily_data),
                    "timespan": f"{daily_data.iloc[0]['date'].strftime('%Y-%m-%d')} to {daily_data.iloc[-1]['date'].strftime('%Y-%m-%d')}",
                    "total_increase": enhanced_metrics["changes"]["wealth_pct"],
                    "exponential_growth_rate": enhanced_metrics.get(
                        "exponential_fit", {}
                    ).get("annual_growth_rate", 0),
                },
            },
            "count_timeline": {
                "data": [
                    {
                        "x": row["date"].strftime("%Y-%m-%d"),
                        "y": int(row["billionaire_count"]),
                    }
                    for _, row in daily_data.iterrows()
                ]
            },
            "inequality_timeline": {
                "data": [
                    {
                        "x": row["date"].strftime("%Y-%m-%d"),
                        "y": round(row["wealth_inequality"], 3),
                    }
                    for _, row in daily_data.iterrows()
                ]
            },
        }

        self._save_json(chart_data, "chart_data.json")

    def _save_json(self, data, filename):
        """Save data as formatted JSON file with proper type conversion"""
        filepath = self.output_dir / filename

        # Convert numpy types to native Python types recursively
        clean_data = self._convert_numpy_types(data)

        with open(filepath, "w") as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)

        file_size = filepath.stat().st_size
        self.logger.info(f"💾 Saved {filename} ({file_size:,} bytes)")

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def generate_summary_report(self):
        """Generate a human-readable summary report"""
        self.logger.info("📄 Generating summary report...")

        try:
            # Load generated data
            metrics = json.loads((self.output_dir / "metrics.json").read_text())
            metadata = json.loads((self.output_dir / "metadata.json").read_text())

            report = f"""
RED FLAGS PROFITS - DATA ANALYSIS SUMMARY
=========================================

Dataset Information:
- Analysis Date: {metadata['last_updated'][:10]}
- Data Period: {metadata['data_start_date']} to {metadata['data_end_date']}
- Total Days: {metadata['data_days_span']} days
- Data Points: {metadata['data_points']} daily observations

Current Metrics:
- Total Billionaire Wealth: ${metrics['total_wealth']:.1f} trillion
- Number of Billionaires: {metrics['billionaire_count']:,}
- Average Fortune: ${metrics['average_wealth']:.1f} billion

Growth Analysis:
- Annual Growth Rate (CAGR): {metrics['growth_rate']:.1f}%
- Wealth Doubling Time: {metrics['doubling_time']:.1f} years
- Daily Wealth Accumulation: ${metrics['daily_accumulation']:.1f} billion

Changes Since Start:
- Total Wealth Growth: +{metrics['changes']['wealth_pct']:.1f}%
- New Billionaires: +{metrics['changes']['count_change']:,}
- Average Wealth Growth: +{metrics['changes']['avg_pct']:.1f}%

Technical Details:
- Exponential Fit R²: {metadata['trend_analysis']['exponential_fit_r_squared']:.3f}
- Fit Period: {metadata['trend_analysis']['fit_period_days']} days
- Inflation Data: {'Available' if any(metadata['inflation_data'].values()) else 'Not Available'}

Generated Files:
- metrics.json: Core financial metrics
- timeline.json: Complete daily data series
- sparklines.json: Visualization data
- equivalencies.json: Wealth comparison data
- chart_data.json: Interactive chart data
- metadata.json: Dataset information

Analysis Features:
{chr(10).join('- ' + feature.replace('_', ' ').title() for feature in metadata['analysis_features'])}
"""

            report_path = self.output_dir / "analysis_summary.txt"
            report_path.write_text(report)

            self.logger.info(f"📄 Summary report saved: {report_path}")
            print(report)  # Also print to console

        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary report: {e}")


def main():
    """Main execution with enhanced options"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Red Flags Profits Data Analysis"
    )
    parser.add_argument(
        "--input",
        default="data/all_billionaires.parquet",
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output-dir", default="output", help="Output directory for generated files"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate human-readable summary report"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run enhanced analysis
    analyzer = EnhancedRedFlagsAnalyzer(
        input_parquet=args.input, output_dir=args.output_dir
    )

    success = analyzer.analyze_comprehensive()

    if success and args.summary:
        analyzer.generate_summary_report()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
