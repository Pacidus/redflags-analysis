"""
Enhanced Sparkline Generator for Red Flags Profits
Generates SVG data URIs for background sparklines in metric cards
"""

import numpy as np
import pandas as pd
from urllib.parse import quote


class SparklineGenerator:
    """Generates SVG sparklines for metric card backgrounds"""
    
    COLOR_SCHEMES = {
        "wealth": ("#404040", "#1a1a1a"),
        "count": ("#3a3a3a", "#222222"), 
        "average": ("#383838", "#1f1f1f"),
    }
    
    def __init__(self, card_width=280, card_height=120):
        self.card_width = card_width
        self.card_height = card_height
        
    def generate_sparkline_svg_uri(self, values, sparkline_type="wealth"):
        """Generate SVG data URI for sparkline background"""
        if not values or len(values) < 2:
            return None
            
        # Get color scheme
        bg_color, fill_color = self.COLOR_SCHEMES.get(sparkline_type, self.COLOR_SCHEMES["wealth"])
        
        # Normalize values to coordinate space
        coords = self._values_to_coordinates(values)
        
        # Create SVG polygon points (filled area under curve)
        polygon_points = self._create_polygon_points(coords)
        
        # Generate SVG
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.card_width} {self.card_height}">
  <rect width="100%" height="100%" fill="{bg_color}"/>
  <polygon points="{polygon_points}" fill="{fill_color}" stroke="none"/>
</svg>'''
        
        # Convert to data URI
        return f"data:image/svg+xml,{quote(svg)}"
    
    def _values_to_coordinates(self, values):
        """Convert data values to SVG coordinates"""
        values = np.array(values)
        
        # Normalize values to 0-1 range
        min_val, max_val = values.min(), values.max()
        range_val = max_val - min_val if max_val > min_val else 1
        normalized = (values - min_val) / range_val
        
        # Map to coordinate space with padding
        padding = 10
        x_coords = np.linspace(0, self.card_width, len(values))
        y_coords = self.card_height - padding - normalized * (self.card_height - 2 * padding)
        
        return list(zip(x_coords, y_coords))
    
    def _create_polygon_points(self, coords):
        """Create polygon points for filled area under curve"""
        # Start from bottom-left
        points = [f"0,{self.card_height}"]
        
        # Add curve points
        for x, y in coords:
            points.append(f"{x:.1f},{y:.1f}")
            
        # End at bottom-right
        points.append(f"{self.card_width},{self.card_height}")
        
        return " ".join(points)
    
    def generate_all_sparklines(self, daily_data):
        """Generate sparklines for all metrics"""
        sparklines = {}
        
        # Generate wealth sparkline
        wealth_values = daily_data['total_wealth'].tolist()
        sparklines['total_wealth'] = self.generate_sparkline_svg_uri(wealth_values, "wealth")
        
        # Generate count sparkline  
        count_values = daily_data['billionaire_count'].tolist()
        sparklines['billionaire_count'] = self.generate_sparkline_svg_uri(count_values, "count")
        
        # Generate average wealth sparkline
        avg_values = daily_data['average_wealth'].tolist()
        sparklines['average_wealth'] = self.generate_sparkline_svg_uri(avg_values, "average")
        
        return sparklines


class AdvancedDataProcessor:
    """Advanced data processing for enhanced analytics"""
    
    def __init__(self):
        self.sparkline_gen = SparklineGenerator()
        
    def calculate_exponential_fit(self, daily_data):
        """Calculate exponential trend fit for wealth growth"""
        df = daily_data.copy()
        df['days_from_start'] = (df['date'] - df['date'].iloc[0]).dt.days
        
        # Filter recent data for more accurate trend (last 2 years or all data if less)
        cutoff_date = df['date'].iloc[-1] - pd.Timedelta(days=730)
        recent_data = df[df['date'] >= cutoff_date]
        
        if len(recent_data) < 2:
            recent_data = df
            
        # Calculate exponential fit using log-linear regression
        x = recent_data['days_from_start'].values
        y = np.log(recent_data['total_wealth'].values)
        
        # Fit: log(y) = bx + log(a) => y = a * exp(bx)
        b, log_a = np.polyfit(x, y, 1)
        a = np.exp(log_a)
        
        # Calculate R-squared
        y_pred = b * x + log_a
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Convert to annual growth rate
        annual_growth_rate = (np.exp(b * 365.25) - 1) * 100
        
        return {
            'a': float(a),
            'b': float(b), 
            'r_squared': float(r_squared),
            'annual_growth_rate': float(annual_growth_rate),
            'fit_period_days': int(len(recent_data))
        }
    
    def generate_trend_line_data(self, daily_data, fit_params):
        """Generate trend line data points"""
        start_date = daily_data['date'].iloc[0]
        end_date = daily_data['date'].iloc[-1]
        days_range = (end_date - start_date).days
        
        # Generate 50 points for smooth curve
        trend_days = np.linspace(0, days_range, 50)
        trend_dates = [start_date + pd.Timedelta(days=int(d)) for d in trend_days]
        trend_values = [fit_params['a'] * np.exp(fit_params['b'] * d) for d in trend_days]
        
        return [
            {
                'x': date.strftime('%Y-%m-%d'),
                'y': round(value, 2)
            }
            for date, value in zip(trend_dates, trend_values)
        ]
    
    def calculate_inflation_adjusted_data(self, daily_data):
        """Calculate inflation-adjusted data if inflation columns are available"""
        inflation_cols = [col for col in ['cpi_u', 'pce'] if col in daily_data.columns]
        
        if not inflation_cols:
            return None
            
        # Use the first available inflation column
        inflation_col = inflation_cols[0]
        
        # Filter data with valid inflation values
        valid_data = daily_data[daily_data[inflation_col].notna()].copy()
        
        if len(valid_data) < 2:
            return None
            
        # Use the latest inflation value as the base (current dollars)
        base_inflation = valid_data[inflation_col].iloc[-1]
        
        # Adjust all values to current dollar purchasing power
        valid_data['inflation_adjusted_wealth'] = (
            valid_data['total_wealth'] * (base_inflation / valid_data[inflation_col])
        )
        
        inflation_data = []
        for _, row in valid_data.iterrows():
            inflation_data.append({
                'x': row['date'].strftime('%Y-%m-%d'),
                'y': round(row['inflation_adjusted_wealth'], 2)
            })
            
        return {
            'data': inflation_data,
            'inflation_type': inflation_col.upper().replace('_', '-'),
            'base_value': base_inflation,
            'base_date': valid_data['date'].iloc[-1].strftime('%Y-%m-%d')
        }
    
    def generate_enhanced_metrics(self, daily_data):
        """Generate enhanced metrics with trend analysis"""
        base_metrics = self._calculate_basic_metrics(daily_data)
        
        # Add exponential fit analysis
        fit_params = self.calculate_exponential_fit(daily_data)
        trend_line = self.generate_trend_line_data(daily_data, fit_params)
        
        # Add inflation analysis
        inflation_data = self.calculate_inflation_adjusted_data(daily_data)
        
        # Generate sparklines
        sparklines = self.sparkline_gen.generate_all_sparklines(daily_data)
        
        enhanced_metrics = {
            **base_metrics,
            'exponential_fit': fit_params,
            'trend_line': trend_line,
            'inflation_adjusted': inflation_data,
            'background_sparklines': sparklines
        }
        
        return enhanced_metrics
    
    def _calculate_basic_metrics(self, daily_data):
        """Calculate basic metrics (same as original analyzer)"""
        latest = daily_data.iloc[-1]
        first = daily_data.iloc[0]
        
        days_diff = (latest['date'] - first['date']).days
        years_diff = days_diff / 365.25
        
        if years_diff > 0 and first['total_wealth'] > 0:
            growth_rate = ((latest['total_wealth'] / first['total_wealth']) ** (1 / years_diff) - 1) * 100
        else:
            growth_rate = 0.0
            
        doubling_time = np.log(2) / np.log(1 + growth_rate / 100) if growth_rate > 0 else float('inf')
        daily_accumulation = (latest['total_wealth'] - first['total_wealth']) / days_diff if days_diff > 0 else 0.0
        
        return {
            "billionaire_count": int(latest['billionaire_count']),
            "total_wealth": float(round(latest['total_wealth'], 1)),
            "average_wealth": float(round(latest['average_wealth'], 1)),
            "growth_rate": float(round(growth_rate, 1)),
            "doubling_time": float(round(doubling_time, 1)) if doubling_time != float('inf') else 999.9,
            "daily_accumulation": float(round(daily_accumulation, 1)),
            "changes": {
                "wealth_pct": float(round(((latest['total_wealth'] - first['total_wealth']) / first['total_wealth']) * 100, 1)),
                "count_change": int(latest['billionaire_count'] - first['billionaire_count']),
                "avg_pct": float(round(((latest['average_wealth'] - first['average_wealth']) / first['average_wealth']) * 100, 1))
            }
        }
