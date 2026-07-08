"""``read_window`` and ``band_math`` for ``geo-raster`` (Reqs 7.2, 7.12, 12.1).

This module ties the COG byte-range engine (:mod:`geo_raster.cog`) to the
windowed-imagery public capabilities:

* :func:`read_window` — return only the pixels inside a requested window of a
  Cloud-Optimized GeoTIFF, reading only the overlapping tiles' byte ranges
  directly from S3/HTTP, never copying the full asset (Requirements 7.2, 12.1).
* :func:`band_math` — evaluate an NDVI/NDWI/NBR-style expression over a windowed
  read, fetching only the bands the expression references.

The production byte-range reader (:class:`~geo_raster.reader.HttpRangeReader`)
and the ``s3://`` resolution are shared with :mod:`geo_raster.reader`, so reads
inherit the Power Pack's retry/backoff and 30-second timeout.

Every user-supplied parameter is validated **before** any byte-range read, and
malformed parameters raise the shared ``Error_Taxonomy``
:class:`~geo_common.errors.ValidationError` (Requirement 7.12).
"""

from __future__ import annotations

import ast
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

from pydantic import ValidationError as PydanticValidationError

from geo_common.errors import ValidationError
from geo_common.http import HttpClient

from geo_raster.cog import ByteRangeReader, CogMetadata, CogReader
from geo_raster.reader import DEFAULT_S3_REGION, HttpRangeReader
from geo_raster.window_models import GeoWindow, PixelWindow, RasterArray

__all__ = [
    "read_window",
    "band_math",
    "resolve_pixel_window",
    "DEFAULT_S3_REGION",
]

#: A window argument may be given as a typed model or a plain mapping.
WindowArg = Union[PixelWindow, GeoWindow, Dict[str, object]]

_BAND_TOKEN = re.compile(r"^B(\d+)$")


async def read_window(
    *,
    asset_href: str,
    window: WindowArg,
    bands: Optional[Sequence[int]] = None,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    reader: Optional[ByteRangeReader] = None,
) -> RasterArray:
    """Return only the pixels inside ``window`` of a COG asset (Req 7.2, 12.1).

    Parameters
    ----------
    asset_href:
        ``s3://<bucket>/<key>`` or ``http(s)://...`` location of the COG asset.
    window:
        A :class:`~geo_raster.window_models.PixelWindow`,
        :class:`~geo_raster.window_models.GeoWindow`, or an equivalent mapping. A
        GeoWindow is resolved to pixels via the asset's georeferencing.
    bands:
        Optional list of 1-based band indices to return; defaults to every band.
    http:
        Optional shared :class:`HttpClient`; created and closed for the call
        when omitted (ignored when an explicit ``reader`` is supplied).
    region:
        AWS region used to resolve ``s3://`` URIs (default ``us-east-1``).
    reader:
        Optional pre-built :class:`~geo_raster.cog.ByteRangeReader` (used by
        tests with an in-memory asset); bypasses HTTP/S3 entirely.

    Raises
    ------
    ValidationError
        For a malformed asset href, window, or band list (Requirement 7.12).
    """
    href = _validate_href(asset_href)
    win_model = _coerce_window(window)

    owns_client = reader is None and http is None
    if reader is not None:
        source: ByteRangeReader = reader
        close_client = None
    else:
        client = http if http is not None else HttpClient()
        source = HttpRangeReader(href, client, region=region, source_id=href)
        close_client = client if owns_client else None

    try:
        cog = CogReader(source, source_id=href)
        meta = await cog.open()
        band_indices = _validate_bands(bands, meta, source_id=href)
        pixel = resolve_pixel_window(win_model, meta, source_id=href)
        data = await cog.read_window_pixels(
            col_off=pixel.col_off,
            row_off=pixel.row_off,
            width=pixel.width,
            height=pixel.height,
            bands=band_indices,
        )
    finally:
        if close_client is not None:
            await close_client.aclose()

    return RasterArray(
        width=pixel.width,
        height=pixel.height,
        band_indices=band_indices,
        dtype=meta.dtype,
        nodata=meta.nodata,
        data=data,
    )


async def band_math(
    *,
    asset_href: str,
    expression: str,
    window: WindowArg,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    reader: Optional[ByteRangeReader] = None,
) -> RasterArray:
    """Evaluate a band-math ``expression`` over a windowed read (NDVI/NDWI/NBR).

    The expression references bands as ``B<n>`` (1-based), e.g. NDVI is
    ``"(B5 - B4) / (B5 + B4)"``. Only the referenced bands are read, and only
    the requested window's tiles are fetched (Requirements 7.2, 12.1). The
    result is a single-band :class:`RasterArray` (``band_indices == [0]``) of the
    per-pixel expression value; a pixel whose denominator is zero or whose inputs
    are nodata yields the asset nodata value (or NaN when none is declared).

    Raises
    ------
    ValidationError
        For a malformed href/window, an empty/unparseable expression, an
        expression that uses an unsupported operation, or a referenced band that
        the asset does not have (Requirement 7.12).
    """
    href = _validate_href(asset_href)
    win_model = _coerce_window(window)
    node, referenced = _parse_expression(expression, source_id=href)

    block = await read_window(
        asset_href=href,
        window=win_model,
        bands=sorted(referenced),
        http=http,
        region=region,
        reader=reader,
    )

    nodata = block.nodata
    fill = nodata if nodata is not None else math.nan
    width, height = block.width, block.height
    out: List[float] = [fill] * (width * height)

    band_grids = {b: block.band(b) for b in referenced}
    # Map each literal identifier in the expression (e.g. "B08", "B8") to its
    # band number, so the evaluator can look up env by the exact node.id and
    # both zero-padded and unpadded spellings resolve (not just "B<int>").
    id_to_band = {
        name.id: int(_BAND_TOKEN.match(name.id).group(1))
        for name in ast.walk(node)
        if isinstance(name, ast.Name) and _BAND_TOKEN.match(name.id)
    }
    for i in range(width * height):
        env: Dict[str, float] = {ident: band_grids[b][i] for ident, b in id_to_band.items()}
        # nodata-in -> nodata-out so masked pixels never contaminate the result.
        if nodata is not None and any(v == nodata for v in env.values()):
            out[i] = fill
            continue
        try:
            out[i] = float(_eval_node(node, env))
        except (ZeroDivisionError, ValueError):
            out[i] = fill

    return RasterArray(
        width=width,
        height=height,
        band_indices=[0],
        dtype="float64",
        nodata=None if nodata is None else float(nodata),
        data=[out],
    )


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------


def resolve_pixel_window(
    window: Union[PixelWindow, GeoWindow],
    meta: CogMetadata,
    *,
    source_id: str,
) -> PixelWindow:
    """Resolve any window to a clamped, in-bounds :class:`PixelWindow`.

    A :class:`PixelWindow` is validated against the asset extent; a
    :class:`GeoWindow` is mapped to pixel space via the asset's geotransform.
    Raises :class:`ValidationError` when the window is out of bounds, when the
    asset lacks georeferencing for a GeoWindow, or when the resolved window is
    empty (Requirement 7.12).
    """
    if isinstance(window, PixelWindow):
        if window.col_off >= meta.width or window.row_off >= meta.height:
            raise ValidationError(
                "window origin lies outside the asset extent "
                f"({meta.width}x{meta.height})",
                source=source_id,
                detail={"parameter": "window"},
            )
        width = min(window.width, meta.width - window.col_off)
        height = min(window.height, meta.height - window.row_off)
        return PixelWindow(
            col_off=window.col_off, row_off=window.row_off, width=width, height=height
        )

    # GeoWindow -> pixel window via the geotransform.
    if meta.geotransform is None:
        raise ValidationError(
            "asset has no georeferencing; supply a pixel window instead",
            source=source_id,
            detail={"parameter": "window"},
        )
    min_x, min_y, max_x, max_y = window.bbox
    gt = meta.geotransform
    corners = [
        _world_to_pixel(min_x, min_y, gt),
        _world_to_pixel(min_x, max_y, gt),
        _world_to_pixel(max_x, min_y, gt),
        _world_to_pixel(max_x, max_y, gt),
    ]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    col_start = max(0, int(math.floor(min(cols))))
    col_end = min(meta.width, int(math.ceil(max(cols))))
    row_start = max(0, int(math.floor(min(rows))))
    row_end = min(meta.height, int(math.ceil(max(rows))))
    if col_end <= col_start or row_end <= row_start:
        raise ValidationError(
            "geo window does not overlap the asset extent",
            source=source_id,
            detail={"parameter": "window"},
        )
    return PixelWindow(
        col_off=col_start,
        row_off=row_start,
        width=col_end - col_start,
        height=row_end - row_start,
    )


def _world_to_pixel(
    x: float, y: float, gt: Tuple[float, float, float, float, float, float]
) -> Tuple[float, float]:
    """Invert a GDAL-style geotransform to fractional (col, row)."""
    origin_x, px_w, row_rot, origin_y, col_rot, px_h = gt
    det = px_w * px_h - row_rot * col_rot
    if det == 0:
        raise ValidationError("asset geotransform is not invertible")
    dx = x - origin_x
    dy = y - origin_y
    col = (px_h * dx - row_rot * dy) / det
    row = (-col_rot * dx + px_w * dy) / det
    return col, row


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def _validate_href(asset_href: object) -> str:
    if not isinstance(asset_href, str) or not asset_href.strip():
        raise ValidationError(
            "asset_href must be a non-empty string",
            detail={"parameter": "asset_href"},
        )
    href = asset_href.strip()
    parts = urlsplit(href)
    scheme = parts.scheme.lower()
    if scheme not in ("s3", "http", "https"):
        raise ValidationError(
            "asset_href must be an s3:// or http(s):// URL",
            detail={"parameter": "asset_href"},
        )
    if scheme == "s3" and (not parts.netloc or not parts.path.lstrip("/")):
        raise ValidationError(
            "s3 asset_href must be of the form s3://<bucket>/<key>",
            detail={"parameter": "asset_href"},
        )
    return href


def _coerce_window(window: WindowArg) -> Union[PixelWindow, GeoWindow]:
    if isinstance(window, (PixelWindow, GeoWindow)):
        return window
    if isinstance(window, dict):
        try:
            if "bbox" in window:
                return GeoWindow(**window)
            return PixelWindow(**window)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"invalid window: {exc.errors()[0].get('msg', 'malformed window')}",
                detail={"parameter": "window"},
            ) from exc
    raise ValidationError(
        "window must be a PixelWindow, GeoWindow, or equivalent mapping",
        detail={"parameter": "window"},
    )


def _validate_bands(
    bands: Optional[Sequence[int]],
    meta: CogMetadata,
    *,
    source_id: str,
) -> List[int]:
    if bands is None:
        return list(range(1, meta.samples_per_pixel + 1))
    band_list = list(bands)
    if not band_list:
        raise ValidationError(
            "bands must be a non-empty list of 1-based band indices",
            source=source_id,
            detail={"parameter": "bands"},
        )
    for b in band_list:
        if isinstance(b, bool) or not isinstance(b, int):
            raise ValidationError(
                "band indices must be integers",
                source=source_id,
                detail={"parameter": "bands"},
            )
        if b < 1 or b > meta.samples_per_pixel:
            raise ValidationError(
                f"band {b} is out of range; asset has {meta.samples_per_pixel} "
                "band(s) (1-based)",
                source=source_id,
                detail={"parameter": "bands"},
            )
    return band_list


# ---------------------------------------------------------------------------
# Band-math expression parsing + evaluation (safe AST subset)
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def _parse_expression(expression: object, *, source_id: str) -> Tuple[ast.AST, List[int]]:
    """Parse a band-math expression into an AST and its referenced band set.

    Only a safe arithmetic subset is allowed: numeric literals, ``B<n>`` band
    references, ``+ - * /``, parentheses, unary +/-, and the whitelisted
    functions ``abs``/``min``/``max``/``sqrt``. Anything else raises a
    :class:`ValidationError` (Requirement 7.12).
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValidationError(
            "expression must be a non-empty string",
            source=source_id,
            detail={"parameter": "expression"},
        )
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValidationError(
            f"expression is not valid: {exc.msg}",
            source=source_id,
            detail={"parameter": "expression"},
        ) from exc

    referenced: set[int] = set()
    _check_node(tree.body, referenced, source_id=source_id)
    if not referenced:
        raise ValidationError(
            "expression must reference at least one band (e.g. B4, B5)",
            source=source_id,
            detail={"parameter": "expression"},
        )
    return tree.body, sorted(referenced)


def _check_node(node: ast.AST, referenced: set, *, source_id: str) -> None:
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            _reject(type(node.op).__name__, source_id)
        _check_node(node.left, referenced, source_id=source_id)
        _check_node(node.right, referenced, source_id=source_id)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            _reject(type(node.op).__name__, source_id)
        _check_node(node.operand, referenced, source_id=source_id)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            _reject("function call", source_id)
        if node.keywords:
            _reject("keyword arguments", source_id)
        for arg in node.args:
            _check_node(arg, referenced, source_id=source_id)
    elif isinstance(node, ast.Name):
        match = _BAND_TOKEN.match(node.id)
        if not match:
            raise ValidationError(
                f"unknown identifier {node.id!r}; reference bands as B<n>",
                source=source_id,
                detail={"parameter": "expression"},
            )
        band = int(match.group(1))
        if band < 1:
            raise ValidationError(
                "band references must be 1-based (e.g. B1)",
                source=source_id,
                detail={"parameter": "expression"},
            )
        referenced.add(band)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            _reject("non-numeric constant", source_id)
    else:
        _reject(type(node).__name__, source_id)


def _reject(what: str, source_id: str) -> None:
    raise ValidationError(
        f"unsupported expression element: {what}",
        source=source_id,
        detail={"parameter": "expression"},
    )


def _eval_node(node: ast.AST, env: Dict[str, float]) -> float:
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right  # ZeroDivisionError handled by caller
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env)
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call):
        func = _ALLOWED_FUNCS[node.func.id]  # type: ignore[attr-defined]
        return float(func(*[_eval_node(a, env) for a in node.args]))
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Constant):
        return float(node.value)
    raise ValueError("unevaluable expression node")  # pragma: no cover - guarded by parse
