"""Data models for the ``geo-pointcloud`` server (Pillar B, expansion).

This module defines the data types the ``geo-pointcloud`` tools accept and
return when reading and writing Cloud-Optimized Point Cloud (COPC) data
(Requirement 8.11):

* :class:`PointRecord` - a single point: its ``x``/``y``/``z`` position plus the
  common optional LAS/LAZ point dimensions (intensity, classification, return
  numbering, RGB color, GPS time). Coordinates are kept as ``float`` and the
  attributes as ``int`` so a point read back from a written file compares equal
  to the point that was written (design Property 20).
* :class:`PointCloudChunk` - a set of points plus the cloud's ``crs`` and
  optional schema (the names of the non-positional dimensions carried by every
  point). It is the value ``read_pointcloud`` returns and ``write_pointcloud``
  consumes.
* :class:`GeoWindow` - a spatial query window used to read only the points that
  fall inside a sub-region of a COPC file, exploiting COPC's spatially-indexed
  octree for partial reads.
* :class:`FormatResult` - the result of ``write_pointcloud`` (and, in general, a
  format-producing operation): the output ``href``, the ``format`` produced,
  whether the output is ``valid``, and the number of points written.

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations on 3.9+.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = [
    "PointRecord",
    "PointCloudChunk",
    "GeoWindow",
    "FormatResult",
    "POSITIONAL_DIMENSIONS",
    "OPTIONAL_DIMENSIONS",
]

#: The three positional dimensions every point carries (LAS X/Y/Z).
POSITIONAL_DIMENSIONS: Tuple[str, ...] = ("x", "y", "z")

#: The optional, non-positional point dimensions this server round-trips. These
#: mirror the common LAS/LAZ point-record fields so a COPC written from a chunk
#: and read back preserves each point's full attribute set (design Property 20).
OPTIONAL_DIMENSIONS: Tuple[str, ...] = (
    "intensity",
    "classification",
    "return_number",
    "number_of_returns",
    "red",
    "green",
    "blue",
    "gps_time",
)


class PointRecord(BaseModel):
    """A single point in a point cloud.

    Carries the required ``x``/``y``/``z`` position (in the cloud's CRS) and the
    common optional LAS/LAZ dimensions. An optional dimension left ``None`` is
    simply absent for that point. Equality is field-wise (pydantic), so two
    records compare equal exactly when every dimension matches - which is what
    "the point set is preserved" means for the round-trip property (Property
    20).
    """

    x: float
    y: float
    z: float = 0.0
    intensity: Optional[int] = None
    classification: Optional[int] = None
    return_number: Optional[int] = None
    number_of_returns: Optional[int] = None
    red: Optional[int] = None
    green: Optional[int] = None
    blue: Optional[int] = None
    gps_time: Optional[float] = None

    def position(self) -> Tuple[float, float, float]:
        """The point's ``(x, y, z)`` position as a tuple."""
        return (self.x, self.y, self.z)


class PointCloudChunk(BaseModel):
    """A set of points plus the cloud's CRS (the read/write unit, Req 8.11).

    ``points`` holds the point records; ``crs`` identifies the coordinate
    reference system the positions are expressed in (defaulting to WGS84). The
    chunk is order-tolerant: ``read_pointcloud`` may return the points in a
    different order than they were written (COPC organizes points into a
    spatial octree), so the preserved invariant is the *set* of points, not
    their sequence (design Property 20).
    """

    points: List[PointRecord] = Field(default_factory=list)
    crs: str = "EPSG:4326"

    @property
    def point_count(self) -> int:
        """The number of points in the chunk."""
        return len(self.points)

    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """The 3D extent ``(min_x, min_y, min_z, max_x, max_y, max_z)``.

        Returns ``None`` for an empty chunk (an empty point set has no extent).
        """
        if not self.points:
            return None
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        zs = [p.z for p in self.points]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


class GeoWindow(BaseModel):
    """A spatial window for partial point-cloud reads.

    Bounds are an axis-aligned box ``(min_x, min_y, max_x, max_y)`` in the
    cloud's CRS; ``min_z``/``max_z`` optionally constrain elevation as well. A
    point is *inside* the window when its ``x``/``y`` (and ``z`` when the z
    bounds are given) fall within the inclusive box. Reading with a window
    returns only the points inside it, exploiting COPC's octree index.
    """

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_z: Optional[float] = None
    max_z: Optional[float] = None

    def contains(self, point: PointRecord) -> bool:
        """Whether ``point`` falls inside this window (inclusive bounds)."""
        if not (self.min_x <= point.x <= self.max_x):
            return False
        if not (self.min_y <= point.y <= self.max_y):
            return False
        if self.min_z is not None and point.z < self.min_z:
            return False
        if self.max_z is not None and point.z > self.max_z:
            return False
        return True


class FormatResult(BaseModel):
    """The result of producing a cloud-optimized output (Req 8.11).

    Returned by ``write_pointcloud``: ``href`` is where the output was written,
    ``format`` is the produced format (``"COPC"`` for this server), ``valid``
    reports whether the written output is well-formed, and ``point_count`` is
    the number of points written. ``message`` carries a short, human-readable
    note (e.g. which backend produced the file).
    """

    href: str = Field(min_length=1)
    format: str = Field(default="COPC", min_length=1)
    valid: bool = True
    point_count: int = Field(default=0, ge=0)
    message: Optional[str] = None
