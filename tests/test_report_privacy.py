"""The report must never carry the Wikipedia evidence sentence."""

from __future__ import annotations

import json

from src.resolver import _build_report

TRACK = {
    "title": "Track Title",
    "artist": "Artist",
    "mbid": "11111111-1111-1111-1111-111111111111",
}


async def test_report_has_no_evidence_key_anywhere() -> None:
    node = _build_report()
    result = await node(
        {
            "chosen": TRACK,
            "resolution": "auto",
            "genres": [],
            "samples": [
                {
                    "title": "Parent",
                    "artist": "Artist",
                    "source": "Wikipedia",
                    "confidence": "medium",
                    "samples": [
                        {
                            "title": "Child",
                            "source": "Wikipedia",
                            "confidence": "medium",
                            "samples": [],
                        }
                    ],
                }
            ],
            "missing_sources": [],
        }
    )

    dumped = json.dumps(result["report"])

    assert "evidence" not in dumped
