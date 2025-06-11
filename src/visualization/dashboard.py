"""
Performance Metrics Dashboard

Interactive dashboard for visualizing algorithmic trading strategy performance,
comparisons, and detailed analysis results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os

from ..config import Config


class PerformanceDashboard:
    """
    Interactive performance dashboard for algorithmic trading strategies.
    
    Provides comprehensive visualization of:
    - Strategy performance comparison
    - Equity curves and drawdowns
    - Risk-return analysis
    - Trading activity analysis
    - Performance metrics tables
    """
    
    def __init__(self, title: str = "Algorithmic Trading Performance Dashboard"):
        """
        Initialize the performance dashboard.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.results_data = {}
        self.comparison_data = None
        
        # Color scheme for consistent visualization
        self.colors = {
            'EMA_Crossover': '#1f77b4',    # Blue
            'Momentum': '#ff7f0e',         # Orange  
            'Q_Learning': '#2ca02c',       # Green
            'Benchmark': '#d62728',        # Red
            'Background': '#f8f9fa',
            'Grid': '#e9ecef'
        }
        
        # Dashboard configuration
        self.config = {
            'height': 600,
            'width': 1200,
            'margin': dict(l=50, r=50, t=50, b=50),
            'font_size': 12,
            'title_font_size': 16
        }
    
    def add_strategy_results(self, strategy_name: str, results: Dict[str, Any]) -> None:
        """
        Add strategy results to the dashboard.
        
        Args:
            strategy_name: Name of the strategy
            results: Strategy backtest results
        """
        self.results_data[strategy_name] = results
        print(f"✅ Added {strategy_name} results to dashboard")
    
    def load_comparison_data(self, comparison_file: str = None) -> bool:
        """
        Load strategy comparison data from file.
        
        Args:
            comparison_file: Path to comparison results JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if comparison_file is None:
            comparison_file = "strategy_comparison_report.json"
        
        try:
            if os.path.exists(comparison_file):
                with open(comparison_file, 'r') as f:
                    self.comparison_data = json.load(f)
                print(f"✅ Loaded comparison data from {comparison_file}")
                return True
            else:
                print(f"⚠️ Comparison file {comparison_file} not found")
                return False
        except Exception as e:
            print(f"❌ Error loading comparison data: {e}")
            return False
    
    def create_equity_curves_chart(self) -> go.Figure:
        """Create interactive equity curves comparison chart."""
        fig = go.Figure()
        
        # Add equity curves for each strategy
        for strategy_name, results in self.results_data.items():
            if 'equity_curve' in results:
                equity_data = results['equity_curve']
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(equity_data))),
                    y=equity_data,
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=self.colors.get(strategy_name, '#000000'), width=2),
                    hovertemplate=f'{strategy_name}<br>Period: %{{x}}<br>Portfolio Value: $%{{y:,.2f}}<extra></extra>'
                ))
        
        # Add benchmark line (initial capital)
        if self.results_data:
            max_periods = max(len(results.get('equity_curve', [])) for results in self.results_data.values())
            initial_capital = Config.INITIAL_CAPITAL
            
            fig.add_trace(go.Scatter(
                x=list(range(max_periods)),
                y=[initial_capital] * max_periods,
                mode='lines',
                name='Benchmark (Hold)',
                line=dict(color=self.colors['Benchmark'], width=1, dash='dash'),
                hovertemplate='Benchmark<br>Period: %{x}<br>Portfolio Value: $%{y:,.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Strategy Equity Curves Comparison",
            xaxis_title="Trading Periods",
            yaxis_title="Portfolio Value ($)",
            height=self.config['height'],
            width=self.config['width'],
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98),
            plot_bgcolor='white',
            paper_bgcolor=self.colors['Background']
        )
        
        return fig
    
    def create_performance_metrics_table(self) -> go.Figure:
        """Create performance metrics comparison table."""
        if not self.results_data:
            return go.Figure()
        
        # Prepare table data
        strategies = list(self.results_data.keys())
        metrics = [
            'Total Return (%)', 'Annualized Return (%)', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 
            'Total Trades', 'Profit Factor', 'Calmar Ratio'
        ]
        
        table_data = []
        for strategy in strategies:
            summary = self.results_data[strategy].get('summary', {})
            row = [
                f"{summary.get('total_return_pct', 0):.2f}",
                f"{summary.get('annualized_return_pct', 0):.2f}",
                f"{summary.get('sharpe_ratio', 0):.3f}",
                f"{summary.get('sortino_ratio', 0):.3f}",
                f"{summary.get('max_drawdown_pct', 0):.2f}",
                f"{summary.get('win_rate_pct', 0):.1f}",
                f"{summary.get('total_trades', 0)}",
                f"{summary.get('profit_factor', 0):.2f}",
                f"{summary.get('calmar_ratio', 0):.3f}"
            ]
            table_data.append(row)
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Strategy'] + metrics,
                fill_color='#4472C4',
                font=dict(color='white', size=self.config['font_size']),
                align='center',
                height=40
            ),
            cells=dict(
                values=[strategies] + [[row[i] for row in table_data] for i in range(len(metrics))],
                fill_color=[['#f8f9fa', '#e9ecef'] * len(strategies)],
                font=dict(size=self.config['font_size']),
                align='center',
                height=35
            )
        )])
        
        fig.update_layout(
            title="Performance Metrics Comparison",
            height=400,
            width=self.config['width'],
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=self.colors['Background']
        )
        
        return fig
    
    def create_risk_return_scatter(self) -> go.Figure:
        """Create risk-return scatter plot."""
        fig = go.Figure()
        
        strategies = []
        returns = []
        risks = []
        sharpe_ratios = []
        
        for strategy_name, results in self.results_data.items():
            summary = results.get('summary', {})
            strategies.append(strategy_name)
            returns.append(summary.get('annualized_return_pct', 0))
            risks.append(summary.get('volatility_pct', summary.get('max_drawdown_pct', 0)))
            sharpe_ratios.append(summary.get('sharpe_ratio', 0))
        
        # Create scatter plot
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=strategies,
            textposition='top center',
            marker=dict(
                size=[abs(sr) * 20 + 10 for sr in sharpe_ratios],  # Size based on Sharpe ratio
                color=sharpe_ratios,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=2, color='DarkSlateGrey')
            ),
            hovertemplate='%{text}<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Risk (Volatility/Max Drawdown %)",
            yaxis_title="Annualized Return (%)",
            height=self.config['height'],
            width=self.config['width'],
            plot_bgcolor='white',
            paper_bgcolor=self.colors['Background']
        )
        
        return fig
    
    def create_drawdown_chart(self) -> go.Figure:
        """Create drawdown analysis chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Drawdown Periods', 'Drawdown Distribution'),
            vertical_spacing=0.12
        )
        
        # Drawdown periods for each strategy
        for strategy_name, results in self.results_data.items():
            equity_curve = results.get('equity_curve', [])
            if equity_curve:
                # Calculate drawdown
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (np.array(equity_curve) - peak) / peak * 100
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(drawdown))),
                    y=drawdown,
                    mode='lines',
                    name=f'{strategy_name} Drawdown',
                    line=dict(color=self.colors.get(strategy_name, '#000000')),
                    fill='tonexty' if strategy_name == list(self.results_data.keys())[0] else None
                ), row=1, col=1)
                
                # Drawdown distribution
                fig.add_trace(go.Histogram(
                    x=drawdown,
                    name=f'{strategy_name} Distribution',
                    marker_color=self.colors.get(strategy_name, '#000000'),
                    opacity=0.7,
                    nbinsx=30
                ), row=2, col=1)
        
        fig.update_xaxes(title_text="Trading Periods", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
        fig.update_xaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            title="Drawdown Analysis",
            height=700,
            width=self.config['width'],
            showlegend=True,
            paper_bgcolor=self.colors['Background']
        )
        
        return fig
    
    def create_trading_activity_chart(self) -> go.Figure:
        """Create trading activity analysis chart."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Orders by Strategy', 'Trade Distribution', 'Win Rate Comparison', 'Trade Frequency'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        strategies = list(self.results_data.keys())
        
        # Orders by strategy
        order_counts = [len(results.get('orders', [])) for results in self.results_data.values()]
        fig.add_trace(go.Bar(
            x=strategies,
            y=order_counts,
            name='Total Orders',
            marker_color=[self.colors.get(s, '#000000') for s in strategies]
        ), row=1, col=1)
        
        # Trade distribution (wins vs losses)
        wins = [results.get('summary', {}).get('winning_trades', 0) for results in self.results_data.values()]
        losses = [results.get('summary', {}).get('losing_trades', 0) for results in self.results_data.values()]
        
        fig.add_trace(go.Bar(x=strategies, y=wins, name='Winning Trades', marker_color='green'), row=1, col=2)
        fig.add_trace(go.Bar(x=strategies, y=losses, name='Losing Trades', marker_color='red'), row=1, col=2)
        
        # Win rate comparison
        win_rates = [results.get('summary', {}).get('win_rate_pct', 0) for results in self.results_data.values()]
        fig.add_trace(go.Bar(
            x=strategies,
            y=win_rates,
            name='Win Rate (%)',
            marker_color=[self.colors.get(s, '#000000') for s in strategies]
        ), row=2, col=1)
        
        # Trade frequency over time (example with first strategy)
        if self.results_data:
            first_strategy = list(self.results_data.keys())[0]
            trades = self.results_data[first_strategy].get('trades', [])
            if trades:
                trade_periods = [i for i, trade in enumerate(trades)]
                cumulative_trades = list(range(1, len(trades) + 1))
                
                fig.add_trace(go.Scatter(
                    x=trade_periods,
                    y=cumulative_trades,
                    mode='lines+markers',
                    name='Cumulative Trades',
                    line=dict(color=self.colors.get(first_strategy, '#000000'))
                ), row=2, col=2)
        
        fig.update_layout(
            title="Trading Activity Analysis",
            height=700,
            width=self.config['width'],
            showlegend=True,
            paper_bgcolor=self.colors['Background']
        )
        
        return fig
    
    def create_strategy_ranking_chart(self) -> go.Figure:
        """Create strategy ranking visualization."""
        if not self.comparison_data or 'strategy_rankings' not in self.comparison_data:
            return go.Figure()
        
        rankings = self.comparison_data['strategy_rankings']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Return Ranking', 'Sharpe Ratio Ranking', 'Win Rate Ranking'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Return ranking
        if 'Return (%)' in rankings:
            returns_data = rankings['Return (%)']
            strategies = list(returns_data.keys())
            values = list(returns_data.values())
            
            fig.add_trace(go.Bar(
                x=strategies,
                y=values,
                name='Average Return',
                marker_color=[self.colors.get(s, '#000000') for s in strategies]
            ), row=1, col=1)
        
        # Sharpe ratio ranking
        if 'Sharpe' in rankings:
            sharpe_data = rankings['Sharpe']
            strategies = list(sharpe_data.keys())
            values = list(sharpe_data.values())
            
            fig.add_trace(go.Bar(
                x=strategies,
                y=values,
                name='Average Sharpe',
                marker_color=[self.colors.get(s, '#000000') for s in strategies]
            ), row=1, col=2)
        
        # Win rate ranking
        if 'Win Rate (%)' in rankings:
            winrate_data = rankings['Win Rate (%)']
            strategies = list(winrate_data.keys())
            values = list(winrate_data.values())
            
            fig.add_trace(go.Bar(
                x=strategies,
                y=values,
                name='Average Win Rate',
                marker_color=[self.colors.get(s, '#000000') for s in strategies]
            ), row=1, col=3)
        
        fig.update_layout(
            title="Strategy Performance Rankings",
            height=500,
            width=self.config['width'],
            showlegend=False,
            paper_bgcolor=self.colors['Background']
        )
        
        return fig
    
    def generate_dashboard_html(self, output_file: str = "performance_dashboard.html") -> str:
        """
        Generate complete HTML dashboard with all visualizations.
        
        Args:
            output_file: Output HTML file path
            
        Returns:
            Path to generated HTML file
        """
        print("📊 Generating Performance Metrics Dashboard...")
        
        # Create all charts
        charts = {
            'equity_curves': self.create_equity_curves_chart(),
            'performance_table': self.create_performance_metrics_table(),
            'risk_return': self.create_risk_return_scatter(),
            'drawdown': self.create_drawdown_chart(),
            'trading_activity': self.create_trading_activity_chart(),
            'rankings': self.create_strategy_ranking_chart()
        }
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {self.colors['Background']};
            color: #333;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4472C4;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Comprehensive Analysis of Algorithmic Trading Strategies</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary-stats">
        <div class="stat-card">
            <div class="stat-value">{len(self.results_data)}</div>
            <div class="stat-label">Strategies Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{Config.DATA_START_DATE} - {Config.DATA_END_DATE}</div>
            <div class="stat-label">Analysis Period</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${Config.INITIAL_CAPITAL:,.0f}</div>
            <div class="stat-label">Initial Capital</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{Config.TIMEFRAME}</div>
            <div class="stat-label">Trading Timeframe</div>
        </div>
    </div>
"""
        
        # Add each chart to HTML
        chart_order = [
            ('performance_table', 'Performance Metrics Overview'),
            ('equity_curves', 'Equity Curves Comparison'),
            ('risk_return', 'Risk-Return Analysis'),
            ('rankings', 'Strategy Performance Rankings'),
            ('drawdown', 'Drawdown Analysis'),
            ('trading_activity', 'Trading Activity Analysis')
        ]
        
        for chart_key, chart_title in chart_order:
            if chart_key in charts and charts[chart_key].data:
                chart_html = charts[chart_key].to_html(include_plotlyjs=False, div_id=f"chart_{chart_key}")
                # Extract just the div content
                chart_div = chart_html.split('<div id=')[1].split('</div>')[0]
                chart_div = '<div id=' + chart_div + '</div>'
                
                html_content += f"""
    <div class="chart-container">
        <h2>{chart_title}</h2>
        {chart_div}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Dashboard generated successfully: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error generating dashboard: {e}")
            return None
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all strategies."""
        if not self.results_data:
            return {}
        
        # Calculate overall statistics
        total_strategies = len(self.results_data)
        
        # Find best performing strategy
        best_return_strategy = max(
            self.results_data.items(),
            key=lambda x: x[1].get('summary', {}).get('total_return_pct', -float('inf'))
        )
        
        best_sharpe_strategy = max(
            self.results_data.items(),
            key=lambda x: x[1].get('summary', {}).get('sharpe_ratio', -float('inf'))
        )
        
        # Calculate average metrics
        avg_return = np.mean([
            results.get('summary', {}).get('total_return_pct', 0) 
            for results in self.results_data.values()
        ])
        
        avg_sharpe = np.mean([
            results.get('summary', {}).get('sharpe_ratio', 0) 
            for results in self.results_data.values()
        ])
        
        summary = {
            'total_strategies': total_strategies,
            'analysis_period': f"{Config.DATA_START_DATE} to {Config.DATA_END_DATE}",
            'initial_capital': Config.INITIAL_CAPITAL,
            'best_return': {
                'strategy': best_return_strategy[0],
                'value': best_return_strategy[1].get('summary', {}).get('total_return_pct', 0)
            },
            'best_sharpe': {
                'strategy': best_sharpe_strategy[0],
                'value': best_sharpe_strategy[1].get('summary', {}).get('sharpe_ratio', 0)
            },
            'average_return': avg_return,
            'average_sharpe': avg_sharpe,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def __str__(self) -> str:
        """String representation of dashboard."""
        return f"PerformanceDashboard(strategies={len(self.results_data)}, title='{self.title}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"PerformanceDashboard(strategies={list(self.results_data.keys())}, "
                f"comparison_loaded={self.comparison_data is not None})") 