"""Embedding change detection for the Foundation_Model_Server (Req 9.5/9.9).

``detect_change`` compares two embeddings of the **same area at different
times** and returns a single scalar change measure normalized to ``[0.0, 1.0]``
where higher means more change (Requirement 9.5). It maps cosine similarity
onto the change scale so that:

* two **identical** embeddings have cosine similarity ``1`` and therefore the
  **minimum** change ``0.0`` (Property 12);
* fully anti-correlated embeddings (cosine ``-1``) have the maximum change
  ``1.0``.

If the two embeddings differ in dimensionality the request is rejected with an
``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError` identifying the
mismatch, and **no** change measure is produced (Requirement 9.9). All checks
run before any measure is computed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from geo_common.errors import ValidationError

__all__ = ["detect_change"]

#: Source identifier used on errors raised by this server.
_SOURCE = "geo-foundation-models"


def detect_change(
    embedding_a: Sequence[float],
    embedding_b: Sequence[float],
) -> float:
    """Return a normalized change measure in ``[0.0, 1.0]`` (Req 9.5/9.9).

    ``embedding_a`` and ``embedding_b`` must have equal dimensionality; a
    mismatch raises a ``ValidationError`` (taxonomy ``validation``) naming both
    lengths and yields no measure (Requirement 9.9). On success the change is
    derived from cosine similarity ``s`` as ``(1 - s) / 2`` and clamped to
    ``[0.0, 1.0]``, so identical embeddings yield exactly ``0.0`` (Property 12)
    and higher values indicate greater change (Requirement 9.5).
    """
    if len(embedding_a) != len(embedding_b):
        raise ValidationError(
            "embedding dimensionality mismatch: len(embedding_a)=%d != "
            "len(embedding_b)=%d" % (len(embedding_a), len(embedding_b)),
            source=_SOURCE,
            detail={
                "dimension_a": len(embedding_a),
                "dimension_b": len(embedding_b),
            },
        )

    a = np.asarray(embedding_a, dtype=np.float64)
    b = np.asarray(embedding_b, dtype=np.float64)

    # Identical embeddings are, by definition, the minimum change. Short-circuit
    # so the result is exactly 0.0 rather than a floating-point epsilon away from
    # it (Property 12 / Requirement 9.5).
    if a.size == b.size and np.array_equal(a, b):
        return 0.0

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 and norm_b == 0.0:
        # Two (equal-dimension) zero embeddings are identical: no change.
        similarity = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        # One zero, one non-zero: cosine is undefined; treat as orthogonal so
        # the measure sits at the neutral midpoint rather than a false extreme.
        similarity = 0.0
    else:
        similarity = float(np.dot(a, b) / (norm_a * norm_b))

    # Guard against floating-point drift outside the valid cosine range.
    similarity = max(-1.0, min(1.0, similarity))

    change = (1.0 - similarity) / 2.0
    # Identical embeddings -> similarity 1.0 -> change 0.0 (Property 12).
    return max(0.0, min(1.0, change))
