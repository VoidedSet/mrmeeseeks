"""Design tokens — colors, radii and sizing used across the UI.

Tuned to feel like a single designed system: dark frosted-glass panels with a
soft drop shadow, a bright cyan/blue Meeseeks accent for active affordances.
"""

from __future__ import annotations


class Colors:
    # Frosted glass panel surface. Vertical gradient top→bottom.
    BACKGROUND_PANEL = "rgba(15, 23, 30, 232)"
    BACKGROUND_PANEL_TOP = "rgba(22, 33, 44, 232)"
    BACKGROUND_PANEL_BOTTOM = "rgba(10, 16, 22, 232)"
    BACKGROUND_INPUT = "rgba(25, 38, 50, 220)"

    BORDER_SUBTLE = "rgba(0, 210, 255, 35)"
    BORDER_HAIRLINE = "rgba(0, 210, 255, 20)"
    INNER_HIGHLIGHT = "rgba(0, 210, 255, 30)"  # 1px top inner stroke

    TEXT_PRIMARY = "#f0f8ff"  # Alice blue
    TEXT_SECONDARY = "#8da2b5"
    TEXT_MUTED = "#607485"
    TEXT_PLACEHOLDER = "#4b5d6c"

    ACCENT_BLUE = "#00d2ff"          # Meeseeks cyan-blue
    ACCENT_BLUE_BRIGHT = "#66e5ff"
    ACCENT_BLUE_DEEP = "#009bbd"

    CURSOR_BLUE = "#00d2ff"
    CURSOR_GLOW = "#00d2ff"

    METAL_HIGHLIGHT = "#e6f9ff"
    METAL_LIGHT = "#a6e3e9"
    METAL_MID = "#71c9ce"
    METAL_DEEP = "#112e51"
    METAL_RIM = "#0f233c"
    METAL_GLOW = "#00d2ff"

    # Soft dark metallic surfaces for input bar and response bubble.
    SURFACE_METAL_TOP = "#1a2a3a"
    SURFACE_METAL_BOTTOM = "#0f1a26"
    SURFACE_METAL_BORDER = "rgba(0, 210, 255, 50)"
    SURFACE_METAL_INNER_HIGHLIGHT = "rgba(0, 210, 255, 80)"

    TEXT_ON_METAL = "#f0f8ff"
    TEXT_ON_METAL_SECONDARY = "#8da2b5"
    TEXT_ON_METAL_PLACEHOLDER = "#4b5d6c"

    BUBBLE_BACKGROUND = "rgba(15, 23, 30, 232)"
    BUBBLE_BORDER = "rgba(0, 210, 255, 35)"


class Radius:
    PANEL = 14
    INPUT = 12
    BUBBLE = 16
    BUTTON = 8
    PILL = 999  # fully rounded pill shape


class Sizes:
    PANEL_WIDTH = 340
    INPUT_HEIGHT = 44
    BUBBLE_MAX_WIDTH = 460
    BUBBLE_MAX_HEIGHT = 360
    CURSOR_SIZE = 48
    ACCENT_STRIPE_WIDTH = 3  # vertical accent on the leading edge of cards
