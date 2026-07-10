"""Gated live checks for the open geo-weather-climate sources.

These hit the real public APIs and are skipped unless ``RUN_LIVE_WEATHER=1`` is
set, so the default (and CI) test run stays hermetic. They verify that the
shipped default endpoints actually return data with the expected shape — the
review that caught the retired OpenAQ v2 endpoint (now replaced by the open,
credential-free Open-Meteo air-quality API).

Run with::

    RUN_LIVE_WEATHER=1 pytest packages/geo-weather-climate/tests/test_observations_live.py
"""

from __future__ import annotations

import os

import pytest

from geo_weather_climate.server import GeoWeatherClimateServer

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_WEATHER") != "1",
    reason="set RUN_LIVE_WEATHER=1 to run live weather/air-quality checks",
)

# Berlin, a two-day historical window (archive/air-quality have a short lag).
BERLIN = {"lat": 52.5, "lon": 13.4}
RANGE = ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")


async def test_live_open_meteo_weather_returns_hourly() -> None:
    server = GeoWeatherClimateServer()
    try:
        records = await server.observations(
            location=BERLIN, time_range=RANGE, source="open-meteo"
        )
    finally:
        await server.aclose()
    assert records, "Open-Meteo weather returned no observations"
    assert all(r["source"] == "open-meteo" for r in records)
    assert any("temperature_2m" in r for r in records)


async def test_live_air_quality_returns_pollutants() -> None:
    server = GeoWeatherClimateServer()
    try:
        records = await server.observations(
            location=BERLIN, time_range=RANGE, source="air-quality"
        )
    finally:
        await server.aclose()
    assert records, "Open-Meteo air-quality returned no observations"
    assert all(r["source"] == "air-quality" for r in records)
    # At least one pollutant field present across the returned hours.
    assert any(("pm2_5" in r) or ("pm10" in r) or ("ozone" in r) for r in records)
