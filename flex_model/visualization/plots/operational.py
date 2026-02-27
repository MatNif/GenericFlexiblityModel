"""
Operational time-series visualizations.

This module provides interactive Plotly visualizations for operational analysis:
- Power dispatch profiles (stacked area charts)
- State of charge (SOC) evolution
- Price signal overlays
- Multi-asset coordination views
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, TYPE_CHECKING

from flex_model.settings import DT_HOURS

if TYPE_CHECKING:
    from flex_model.visualization.core.result_processor import OptimizationResult

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from flex_model.visualization.core.color_schemes import get_color_scheme, get_rgba_with_alpha


class OperationalPlots:
    """
    Factory class for operational time-series visualizations.

    All methods are static and return plotly.graph_objects.Figure instances
    that can be displayed (.show()) or saved (.write_html()).
    """

    @staticmethod
    def create_dispatch_profile(
        result: 'OptimizationResult',
        view_mode: str = 'system',
        asset_filter: Optional[List[str]] = None,
        start_date: str = None,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create power dispatch profile visualization.

        Shows how power is balanced across assets and market over time,
        with the imbalance profile as context.

        Args:
            result:
                OptimizationResult instance with optimization data.

            view_mode:
                'system' - Show all assets aggregated (default)
                'by_asset' - Show individual asset contributions stacked

            asset_filter:
                Optional list of asset names to include. If None, includes all.

            start_date:
                Optional start date/time as string (e.g., '2024-01-01 00:00').
                If None, uses hours from start.

        Returns:
            plotly.graph_objects.Figure with stacked area chart.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations. Install with: pip install plotly")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Get timesteps for x-axis
        timesteps = list(range(result.n_timesteps))

        # Create time axis (datetime if start_date provided, otherwise hours)
        if start_date:
            try:
                import pandas as pd
                start = pd.to_datetime(start_date)
                time_axis = [start + pd.Timedelta(hours=t * DT_HOURS) for t in timesteps]
            except:
                time_axis = [t * DT_HOURS for t in timesteps]
        else:
            time_axis = [t * DT_HOURS for t in timesteps]

        # Get imbalance profile (the problem we're solving)
        imbalance_data = result.get_imbalance_profile()
        imbalance = imbalance_data['imbalance']

        # Imbalance values are already in power [kW]
        imbalance_kw = imbalance

        # Create figure
        fig = go.Figure()

        # Determine which assets to plot
        assets_to_plot = asset_filter if asset_filter else list(result.assets.keys())

        # Extract power data for each asset
        if view_mode == 'system':
            # Aggregate all assets
            total_charge = [0.0] * result.n_timesteps
            total_discharge = [0.0] * result.n_timesteps
            total_import = [0.0] * result.n_timesteps
            total_export = [0.0] * result.n_timesteps

            for asset_name in assets_to_plot:
                power_data = result.get_power_profile(asset_name)

                for t in range(result.n_timesteps):
                    # Power profile values are already in power [kW]
                    total_charge[t] += power_data['P_charge'][t]
                    total_discharge[t] += power_data['P_discharge'][t]
                    total_import[t] += power_data['P_import'][t]
                    total_export[t] += power_data['P_export'][t]

            # Add system-level traces as stacked bars
            # Positive contributions (generation/discharge/import)
            # Battery discharge (positive contribution)
            if any(p > 1e-6 for p in total_discharge):
                fig.add_trace(go.Bar(
                    x=time_axis,
                    y=total_discharge,
                    name='Battery Discharge',
                    marker=dict(color=colors.battery_discharge_color),
                ))

            # Market import (positive contribution)
            if any(p > 1e-6 for p in total_import):
                fig.add_trace(go.Bar(
                    x=time_axis,
                    y=total_import,
                    name='Market Import',
                    marker=dict(color=colors.market_import_color),
                ))

            # Negative contributions (load/charge/export)
            # Battery charge (negative contribution - drawn from grid)
            if any(p > 1e-6 for p in total_charge):
                fig.add_trace(go.Bar(
                    x=time_axis,
                    y=[-p for p in total_charge],
                    name='Battery Charge',
                    marker=dict(color=colors.battery_charge_color),
                ))

            # Market export (negative contribution - sent to grid)
            if any(p > 1e-6 for p in total_export):
                fig.add_trace(go.Bar(
                    x=time_axis,
                    y=[-p for p in total_export],
                    name='Market Export',
                    marker=dict(color=colors.market_export_color),
                ))

            # Add imbalance line AFTER bars so it's on top
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=imbalance_kw,
                mode='lines',
                name='Imbalance',
                line=dict(color=colors.imbalance_color, width=2.5, dash='dash'),
            ))

        else:  # view_mode == 'by_asset'
            # Show individual assets
            for asset_name in assets_to_plot:
                power_data = result.get_power_profile(asset_name)

                # Net power for this asset
                net_power = power_data['P_net']

                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=net_power,
                    mode='lines',
                    name=asset_name,
                    stackgroup='power',
                ))

        # Layout
        xaxis_title = 'Date / Time' if start_date else 'Time [hours]'
        fig.update_layout(
            title='Power Dispatch Profile',
            xaxis_title=xaxis_title,
            yaxis_title='Power [kW]',
            hovermode='x unified',
            barmode='relative',  # Enable stacked bars
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template=template,
        )

        # Format x-axis for datetime if applicable
        if start_date:
            fig.update_xaxes(
                tickformat='%a %d/%m\n%H:%M',  # Weekday DD/MM, HH:MM
                # Auto-scale tick frequency based on zoom level
            )

        return fig

    @staticmethod
    def create_soc_evolution(
        result: 'OptimizationResult',
        battery_name: str,
        start_date: str = None,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create SOC evolution visualization for a battery.

        Shows state of charge over time with SOC limits as reference bounds.

        Args:
            result:
                OptimizationResult instance.

            battery_name:
                Name of battery asset to visualize.

            start_date:
                Optional start date/time as string (e.g., '2024-01-01 00:00').

        Returns:
            plotly.graph_objects.Figure with SOC line chart and limit bounds.

        Raises:
            ImportError: If plotly not installed.
            ValueError: If battery_name not found or not a battery.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations. Install with: pip install plotly")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Get SOC data
        soc_data = result.get_soc_profile(battery_name)

        # Get battery parameters
        battery = result.assets[battery_name]
        soc_min = battery.unit.soc_min
        soc_max = battery.unit.soc_max

        # Create timesteps (n_timesteps + 1 for SOC which includes initial state)
        timesteps = list(range(result.n_timesteps + 1))

        # Create time axis
        if start_date:
            try:
                import pandas as pd
                start = pd.to_datetime(start_date)
                time_axis = [start + pd.Timedelta(hours=t * DT_HOURS) for t in timesteps]
            except:
                time_axis = [t * DT_HOURS for t in timesteps]
        else:
            time_axis = [t * DT_HOURS for t in timesteps]

        # Create figure
        fig = go.Figure()

        # Add SOC limit bounds as shaded regions
        fig.add_hrect(
            y0=0,
            y1=soc_min * 100,
            fillcolor=colors.negative_indicator_color,
            opacity=0.1,
            line_width=0,
            annotation_text="Below minimum",
            annotation_position="left"
        )

        fig.add_hrect(
            y0=soc_max * 100,
            y1=100,
            fillcolor=colors.negative_indicator_color,
            opacity=0.1,
            line_width=0,
            annotation_text="Above maximum",
            annotation_position="left"
        )

        fig.add_hrect(
            y0=soc_min * 100,
            y1=soc_max * 100,
            fillcolor=colors.positive_indicator_color,
            opacity=0.05,
            line_width=0,
            annotation_text="Operating range",
            annotation_position="left"
        )

        # Add SOC line
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=soc_data['SOC_percent'],
            mode='lines+markers',
            name='SOC',
            line=dict(color=colors.battery_soc_color, width=3),
            marker=dict(size=6),
        ))

        # Add horizontal lines for limits
        fig.add_hline(
            y=soc_min * 100,
            line_dash="dash",
            line_color=colors.negative_indicator_color,
            annotation_text=f"Min SOC ({soc_min*100:.0f}%)",
            annotation_position="right"
        )

        fig.add_hline(
            y=soc_max * 100,
            line_dash="dash",
            line_color=colors.negative_indicator_color,
            annotation_text=f"Max SOC ({soc_max*100:.0f}%)",
            annotation_position="right"
        )

        # Layout
        xaxis_title = 'Date / Time' if start_date else 'Time [hours]'
        fig.update_layout(
            title=f'State of Charge Evolution: {battery_name}',
            xaxis_title=xaxis_title,
            yaxis_title='SOC [%]',
            yaxis=dict(range=[0, 105]),  # Slightly above 100% for annotations
            hovermode='x',
            template=template,
        )

        # Format x-axis for datetime if applicable
        if start_date:
            fig.update_xaxes(
                tickformat='%a %d/%m\n%H:%M',
                # Auto-scale tick frequency based on zoom level
            )

        return fig

    @staticmethod
    def create_price_overlay(
        result: 'OptimizationResult',
        market_name: str,
        start_date: str = None,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create price signal visualization overlaid with market operations.

        Shows buy/sell prices over time and correlates with market import/export.

        Args:
            result:
                OptimizationResult instance.

            market_name:
                Name of market asset (e.g., 'balancing_market').

            start_date:
                Optional start date/time as string (e.g., '2024-01-01 00:00').

        Returns:
            plotly.graph_objects.Figure with dual y-axes (prices and power).

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Get market asset
        if market_name not in result.assets:
            raise ValueError(f"Market '{market_name}' not found")

        market = result.assets[market_name]

        # Get timesteps
        timesteps = list(range(result.n_timesteps))

        # Create time axis
        if start_date:
            try:
                import pandas as pd
                start = pd.to_datetime(start_date)
                time_axis = [start + pd.Timedelta(hours=t * DT_HOURS) for t in timesteps]
            except:
                time_axis = [t * DT_HOURS for t in timesteps]
        else:
            time_axis = [t * DT_HOURS for t in timesteps]

        # Extract prices (assuming BalancingMarketCost model)
        if hasattr(market, 'cost_model'):
            p_buy = [market.cost_model.p_E_buy(t) for t in timesteps]
            p_sell = [market.cost_model.p_E_sell(t) for t in timesteps]
        else:
            raise ValueError(f"Market '{market_name}' has no cost_model")

        # Get power profile
        power_data = result.get_power_profile(market_name)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add price traces (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=p_buy,
                mode='lines',
                name='Buy Price',
                line=dict(color=colors.market_buy_price_color, width=2),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=p_sell,
                mode='lines',
                name='Sell Price',
                line=dict(color=colors.market_sell_price_color, width=2),
            ),
            secondary_y=False,
        )

        # Add power traces (secondary y-axis) - use lighter versions
        # Import bar (lighter red/cost color)
        import_color = get_rgba_with_alpha(colors.market_buy_price_color, 0.3)
        fig.add_trace(
            go.Bar(
                x=time_axis,
                y=power_data['P_import'],
                name='Market Import',
                marker_color=import_color,
            ),
            secondary_y=True,
        )

        # Export bar (lighter green/revenue color)
        export_color = get_rgba_with_alpha(colors.market_sell_price_color, 0.3)
        fig.add_trace(
            go.Bar(
                x=time_axis,
                y=[-p for p in power_data['P_export']],
                name='Market Export',
                marker_color=export_color,
            ),
            secondary_y=True,
        )

        # Update layout
        xaxis_title = 'Date / Time' if start_date else 'Time [hours]'
        fig.update_xaxes(title_text=xaxis_title)
        fig.update_yaxes(title_text="Price [EUR/kWh]", secondary_y=False)
        fig.update_yaxes(title_text="Power [kW]", secondary_y=True)

        # Align dual y-axes at zero for better readability
        # Calculate ranges for both axes
        price_min = min(min(p_buy), min(p_sell))
        price_max = max(max(p_buy), max(p_sell))
        power_max = max(max(power_data['P_import']), max(power_data['P_export']))
        power_min = -power_max  # Symmetric around zero for import/export

        # Add 10% padding
        price_padding = (price_max - price_min) * 0.1
        power_padding = (power_max - power_min) * 0.1

        # Set ranges to align at zero
        # Calculate the ratio to align zero points
        if price_min < 0:
            # If prices can be negative, align naturally
            fig.update_yaxes(range=[price_min - price_padding, price_max + price_padding], secondary_y=False)
            fig.update_yaxes(range=[power_min - power_padding, power_max + power_padding], secondary_y=True)
        else:
            # If all prices are positive, align zero at the bottom
            fig.update_yaxes(range=[0, price_max + price_padding], secondary_y=False)
            fig.update_yaxes(range=[power_min - power_padding, power_max + power_padding], secondary_y=True)

        fig.update_layout(
            title=f'Price Signals and Market Operations: {market_name}',
            hovermode='x unified',
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Format x-axis for datetime if applicable
        if start_date:
            fig.update_xaxes(
                tickformat='%a %d/%m\n%H:%M',
                # Auto-scale tick frequency based on zoom level
            )

        return fig
