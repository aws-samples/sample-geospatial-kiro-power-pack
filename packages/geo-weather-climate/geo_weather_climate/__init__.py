"""geo-weather-climate: Pillar A (expansion) weather/climate MCP server.

Exposes ``observations``, which returns weather, climate, and environmental
observations for a point location and a ``(start, end)`` time range from open
sources (Open-Meteo weather + Open-Meteo air-quality by default; NOAA/NWS via
api.weather.gov for US points), and rejects malformed locations or a time range
whose start is
later than its end with an ``Error_Taxonomy`` validation error (Requirements
7.6, 7.12).
"""

from __future__ import annotations

from geo_weather_climate.models import Coordinate, Observation
from geo_weather_climate.observations import (
    DEFAULT_SOURCE,
    NoaaCdoSource,
    NwsSource,
    OpenMeteoAirQualitySource,
    OpenMeteoSource,
    WeatherSource,
    default_sources,
    observations,
)
from geo_weather_climate.server import GeoWeatherClimateServer
from geo_weather_climate.timerange import validate_location, validate_time_range

__all__ = [
    # Data models
    "Coordinate",
    "Observation",
    # Validation
    "validate_location",
    "validate_time_range",
    # Sources + connector
    "WeatherSource",
    "OpenMeteoSource",
    "OpenMeteoAirQualitySource",
    "NwsSource",
    "NoaaCdoSource",
    "default_sources",
    "DEFAULT_SOURCE",
    "observations",
    # Server
    "GeoWeatherClimateServer",
]
