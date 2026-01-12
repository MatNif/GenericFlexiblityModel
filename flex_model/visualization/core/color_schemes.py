"""
Color schemes for light and dark mode visualizations.

This module implements a three-layer color design system:
    1. Base Palette: Core color families with light/dark variants
    2. Visual Purpose: Design tokens (danger, success, warning, etc.)
    3. Semantic Mapping: Domain-specific assignments (cost, revenue, battery, etc.)

Design Philosophy:
    - Aiming for color harmony with 5 core families (Red, Green, Blue, Yellow, Neutral)
    - Complementary pairs: Red ↔ Green (economic), Blue ↔ Yellow (operational)
    - Theme-aware: Dark mode uses lighter shades for better contrast
    - Semantic naming: Code references meaning (cost_color), not hue (red)

Color Families:
    - Coral/Red: Costs, dangers, negative indicators
    - Mint/Green: Revenues, success, positive indicators
    - Sky/Blue: Primary, neutral operations
    - Sunny/Yellow: Warnings, highlights, energy storage
    - Neutral/Gray: Backgrounds, text, dividers
"""

from typing import Dict


# =============================================================================
# LAYER 1: BASE COLOR PALETTE
# =============================================================================

class BasePalette:
    """
    Base color palette defining core color families.

    Each family has 5 shades following design system conventions:
    - 100: Very light (backgrounds, subtle highlights)
    - 300: Light (hover states, secondary elements)
    - 500: Standard (primary usage, main color)
    - 700: Dark (emphasis, active states)
    - 900: Very dark (strong emphasis, text on light backgrounds)

    Additionally, the 500 and 700 shades have alpha variants (0.7-0.8 opacity)
    for overlays, stacked charts, and layered visualizations.

    Color values are curated for visual harmony and accessibility, not
    mathematically generated.
    """

    def __init__(
        self,
        # Red family (Coral/Danger tones)
        red_100: str,
        red_300: str,
        red_500: str,
        red_700: str,
        red_900: str,
        red_500_alpha: str,
        red_700_alpha: str,

        # Green family (Mint/Success tones)
        green_100: str,
        green_300: str,
        green_500: str,
        green_700: str,
        green_900: str,
        green_500_alpha: str,
        green_700_alpha: str,

        # Blue family (Sky/Primary tones)
        blue_100: str,
        blue_300: str,
        blue_500: str,
        blue_700: str,
        blue_900: str,
        blue_500_alpha: str,
        blue_700_alpha: str,

        # Yellow family (Sunny/Warning tones)
        yellow_100: str,
        yellow_300: str,
        yellow_500: str,
        yellow_700: str,
        yellow_900: str,
        yellow_500_alpha: str,
        yellow_700_alpha: str,

        # Neutral family (Gray tones)
        neutral_100: str,
        neutral_300: str,
        neutral_500: str,
        neutral_700: str,
        neutral_900: str,
    ):
        # Red family
        self.red_100 = red_100
        self.red_300 = red_300
        self.red_500 = red_500
        self.red_700 = red_700
        self.red_900 = red_900
        self.red_500_alpha = red_500_alpha
        self.red_700_alpha = red_700_alpha

        # Green family
        self.green_100 = green_100
        self.green_300 = green_300
        self.green_500 = green_500
        self.green_700 = green_700
        self.green_900 = green_900
        self.green_500_alpha = green_500_alpha
        self.green_700_alpha = green_700_alpha

        # Blue family
        self.blue_100 = blue_100
        self.blue_300 = blue_300
        self.blue_500 = blue_500
        self.blue_700 = blue_700
        self.blue_900 = blue_900
        self.blue_500_alpha = blue_500_alpha
        self.blue_700_alpha = blue_700_alpha

        # Yellow family
        self.yellow_100 = yellow_100
        self.yellow_300 = yellow_300
        self.yellow_500 = yellow_500
        self.yellow_700 = yellow_700
        self.yellow_900 = yellow_900
        self.yellow_500_alpha = yellow_500_alpha
        self.yellow_700_alpha = yellow_700_alpha

        # Neutral family
        self.neutral_100 = neutral_100
        self.neutral_300 = neutral_300
        self.neutral_500 = neutral_500
        self.neutral_700 = neutral_700
        self.neutral_900 = neutral_900


# Light mode base palette (standard saturation)
LIGHT_PALETTE = BasePalette(
    # Red family - Coral/Danger tones (based on Bootstrap danger)
    red_100='rgb(255, 235, 238)',          # Very pale pink - subtle backgrounds
    red_300='rgb(239, 154, 154)',          # Light coral - hover states
    red_500='rgb(220, 53, 69)',            # Bootstrap danger red - primary
    red_700='rgb(183, 28, 28)',            # Dark red - emphasis
    red_900='rgb(130, 15, 15)',            # Very dark red - strong emphasis
    red_500_alpha='rgba(220, 53, 69, 0.7)',
    red_700_alpha='rgba(183, 28, 28, 0.7)',

    # Green family - Bootstrap success green
    green_100='rgb(232, 245, 233)',        # Very pale mint
    green_300='rgb(129, 199, 132)',        # Light green
    green_500='rgb(40, 167, 69)',          # Standard (current)
    green_700='rgb(27, 128, 45)',          # Dark forest green
    green_900='rgb(15, 90, 25)',           # Very dark green
    green_500_alpha='rgba(40, 167, 69, 0.7)',
    green_700_alpha='rgba(27, 128, 45, 0.7)',

    # Blue family - Sky/primary tones
    blue_100='rgb(227, 242, 253)',         # Very pale sky
    blue_300='rgb(100, 181, 246)',         # Light blue
    blue_500='rgb(0, 123, 255)',           # Standard Bootstrap primary
    blue_700='rgb(13, 71, 161)',           # Dark navy blue
    blue_900='rgb(6, 46, 110)',            # Very dark navy
    blue_500_alpha='rgba(0, 123, 255, 0.7)',
    blue_700_alpha='rgba(13, 71, 161, 0.7)',

    # Yellow family - Sunny/warning tones
    yellow_100='rgb(255, 249, 230)',       # Very pale cream
    yellow_300='rgb(255, 224, 130)',       # Light amber
    yellow_500='rgb(255, 193, 7)',         # Standard warning yellow
    yellow_700='rgb(245, 166, 35)',        # Dark amber/orange
    yellow_900='rgb(230, 145, 0)',         # Very dark amber
    yellow_500_alpha='rgba(255, 193, 7, 0.7)',
    yellow_700_alpha='rgba(245, 166, 35, 0.7)',

    # Neutral family - Warm grays
    neutral_100='rgb(248, 249, 250)',      # Almost white (backgrounds)
    neutral_300='rgb(206, 212, 218)',      # Light gray (dividers)
    neutral_500='rgb(108, 117, 125)',      # Medium gray (text, borders)
    neutral_700='rgb(63, 63, 63)',         # Dark gray (emphasis)
    neutral_900='rgb(33, 37, 41)',         # Very dark charcoal
)


# Dark mode base palette (lighter, more saturated for visibility)
DARK_PALETTE = BasePalette(
    # Red family - Light coral (inverted scale: lighter = higher visibility)
    red_100='rgb(76, 29, 33)',             # Dark burgundy background
    red_300='rgb(229, 83, 83)',            # Medium coral
    red_500='rgb(255, 107, 107)',          # Standard light coral
    red_700='rgb(255, 138, 138)',          # Lighter coral
    red_900='rgb(255, 179, 179)',          # Very light pink/coral
    red_500_alpha='rgba(255, 107, 107, 0.8)',
    red_700_alpha='rgba(255, 138, 138, 0.8)',

    # Green family - Light mint
    green_100='rgb(26, 51, 32)',           # Dark forest background
    green_300='rgb(56, 198, 100)',         # Medium mint
    green_500='rgb(72, 219, 127)',         # Standard light mint
    green_700='rgb(102, 236, 152)',        # Lighter mint
    green_900='rgb(147, 250, 192)',        # Very light mint
    green_500_alpha='rgba(72, 219, 127, 0.8)',
    green_700_alpha='rgba(102, 236, 152, 0.8)',

    # Blue family - Light sky blue
    blue_100='rgb(25, 39, 57)',            # Dark navy background
    blue_300='rgb(75, 161, 255)',          # Medium sky blue
    blue_500='rgb(99, 179, 255)',          # Standard light blue
    blue_700='rgb(130, 200, 255)',         # Lighter blue
    blue_900='rgb(179, 225, 255)',         # Very light sky
    blue_500_alpha='rgba(99, 179, 255, 0.8)',
    blue_700_alpha='rgba(130, 200, 255, 0.8)',

    # Yellow family - Light sunny yellow
    yellow_100='rgb(61, 48, 20)',          # Dark brown background
    yellow_300='rgb(255, 204, 77)',        # Medium sunny yellow
    yellow_500='rgb(255, 214, 102)',       # Standard light yellow
    yellow_700='rgb(255, 230, 138)',       # Lighter yellow
    yellow_900='rgb(255, 244, 194)',       # Very light cream
    yellow_500_alpha='rgba(255, 214, 102, 0.8)',
    yellow_700_alpha='rgba(255, 230, 138, 0.8)',

    # Neutral family - Cool metallic grays (inverted: darker = backgrounds)
    neutral_100='rgb(33, 37, 41)',         # Very dark background
    neutral_300='rgb(73, 80, 87)',         # Dark gray
    neutral_500='rgb(173, 181, 189)',      # Medium light gray (text, borders)
    neutral_700='rgb(206, 212, 218)',      # Light gray (emphasis)
    neutral_900='rgb(233, 236, 239)',      # Very light gray/off-white
)


# =============================================================================
# LAYER 2 & 3: VISUAL PURPOSE + SEMANTIC MAPPING
# =============================================================================

class ColorScheme:
    """
    Complete color scheme combining visual purpose tokens with semantic mappings.

    This class bridges design tokens (danger, success, warning) with domain-specific
    usage (cost, revenue, battery operations). Built from a BasePalette.

    Attributes are organized into semantic groups:
        - Economic: Financial analysis colors
        - Battery: Energy storage operations
        - Market: Grid/market interactions
        - UI: Interface elements and indicators
        - Gauges: Performance threshold colors
    """

    def __init__(self, palette: BasePalette):
        """
        Build semantic color scheme from base palette.

        Args:
            palette: BasePalette instance (light or dark mode)
        """
        # =====================================================================
        # ECONOMIC COLORS
        # =====================================================================
        # Costs represent spending (danger/negative) → Red
        self.cost_color = palette.red_500

        # Revenues represent earning (success/positive) → Green
        self.revenue_color = palette.green_500
        self.savings_color = palette.green_300  # Same as revenue

        # Totals are neutral aggregates → Blue
        self.total_color = palette.blue_500

        # =====================================================================
        # BATTERY OPERATION COLORS
        # =====================================================================
        # Discharge: Battery providing power (generating value) → Green
        self.battery_discharge_color = palette.green_700_alpha

        # Charge: Battery consuming power (storing energy) → Yellow
        self.battery_charge_color = palette.blue_700_alpha

        # SOC: Neutral state indicator → Blue
        self.battery_soc_color = palette.blue_500

        # =====================================================================
        # MARKET OPERATION COLORS
        # =====================================================================
        # Import: Buying from market (cost) → Blue (neutral operation)
        # Note: We use blue instead of red to distinguish operation from cost
        self.market_import_color = palette.blue_500_alpha

        # Export: Selling to market (revenue) → Green
        self.market_export_color = palette.green_500_alpha

        # Buy Price: Price signal for importing (cost indicator) → Red
        self.market_buy_price_color = palette.red_500

        # Sell Price: Price signal for exporting (revenue indicator) → Green
        self.market_sell_price_color = palette.green_500

        # =====================================================================
        # UI ELEMENT COLORS
        # =====================================================================
        # Imbalance: Problem/mismatch signal → Red (danger)
        self.imbalance_color = palette.red_500

        # Annotations: Highlights for user attention → Yellow (warning)
        self.annotation_color = palette.yellow_500

        # Positive indicators: Success states → Green
        self.positive_indicator_color = palette.green_500

        # Negative indicators: Error/problem states → Red
        self.negative_indicator_color = palette.red_500

        # Neutral indicators: Default/inactive states → Gray
        self.neutral_indicator_color = palette.neutral_500

        # =====================================================================
        # GAUGE THRESHOLD COLORS
        # =====================================================================
        # Good: Target exceeded → Green (success)
        self.gauge_good_color = palette.green_500

        # Medium: Approaching target → Yellow (warning)
        self.gauge_medium_color = palette.yellow_500

        # Bad: Below target → Red (danger)
        self.gauge_bad_color = palette.red_500

    def to_dict(self) -> Dict[str, str]:
        """
        Convert color scheme to dictionary for programmatic access.

        Returns:
            Dictionary mapping semantic names to color values.
        """
        return {
            'cost': self.cost_color,
            'revenue': self.revenue_color,
            'savings': self.savings_color,
            'total': self.total_color,
            'battery_discharge': self.battery_discharge_color,
            'battery_charge': self.battery_charge_color,
            'battery_soc': self.battery_soc_color,
            'market_import': self.market_import_color,
            'market_export': self.market_export_color,
            'market_buy_price': self.market_buy_price_color,
            'market_sell_price': self.market_sell_price_color,
            'imbalance': self.imbalance_color,
            'annotation': self.annotation_color,
            'positive_indicator': self.positive_indicator_color,
            'negative_indicator': self.negative_indicator_color,
            'neutral_indicator': self.neutral_indicator_color,
            'gauge_good': self.gauge_good_color,
            'gauge_medium': self.gauge_medium_color,
            'gauge_bad': self.gauge_bad_color,
        }


# Pre-built color schemes for light and dark modes
LIGHT_MODE = ColorScheme(LIGHT_PALETTE)
DARK_MODE = ColorScheme(DARK_PALETTE)


# =============================================================================
# PUBLIC API
# =============================================================================

def get_color_scheme(template: str = 'plotly_white') -> ColorScheme:
    """
    Get the appropriate color scheme based on the Plotly template.

    This is the main entry point for visualization code. It automatically
    selects the right color scheme (light or dark) based on the template name.

    Args:
        template: Plotly template name. Templates containing 'dark' return
                 DARK_MODE, all others return LIGHT_MODE.
                 Examples: 'plotly_white', 'plotly_dark', 'plotly', 'seaborn'

    Returns:
        ColorScheme instance configured for the specified theme.

    Example:
        >>> colors = get_color_scheme('plotly_dark')
        >>> colors.cost_color
        'rgb(255, 107, 107)'
    """
    if 'dark' in template.lower():
        return DARK_MODE
    else:
        return LIGHT_MODE


def get_rgba_with_alpha(color: str, alpha: float) -> str:
    """
    Convert rgb() or rgba() color string to rgba() with specified alpha.

    Useful for creating transparent versions of existing colors for overlays,
    backgrounds, or layered visualizations.

    Args:
        color: Color string in format 'rgb(r, g, b)' or 'rgba(r, g, b, a)'
        alpha: Alpha/opacity value between 0.0 (transparent) and 1.0 (opaque)

    Returns:
        Color string in format 'rgba(r, g, b, alpha)'. If input is not rgb/rgba,
        returns the original color unchanged.

    Examples:
        >>> get_rgba_with_alpha('rgb(255, 0, 0)', 0.5)
        'rgba(255, 0, 0, 0.5)'
        >>> get_rgba_with_alpha('rgba(255, 0, 0, 0.8)', 0.3)
        'rgba(255, 0, 0, 0.3)'
        >>> get_rgba_with_alpha('#FF0000', 0.5)
        '#FF0000'
    """
    # Extract RGB values
    if color.startswith('rgba'):
        # Extract r, g, b from rgba(r, g, b, a)
        parts = color.replace('rgba(', '').replace(')', '').split(',')
        r, g, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
    elif color.startswith('rgb'):
        # Extract r, g, b from rgb(r, g, b)
        parts = color.replace('rgb(', '').replace(')', '').split(',')
        r, g, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
    else:
        # Return original if not rgb/rgba (e.g., hex colors, named colors)
        return color

    return f'rgba({r}, {g}, {b}, {alpha})'