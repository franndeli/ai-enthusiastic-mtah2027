"""Minimal reader for the Smart Tagging XML (SportsCode). Times are seconds from the first frame of
the tactical video (DATA.md), so frame = round(seconds * fps)."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def load(path: str, fps: float = 25.0) -> list:
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        start, end = inst.findtext("start"), inst.findtext("end")
        if start is None or end is None:
            continue
        tags = {}
        for label in inst.findall("label"):
            group, text = label.findtext("group"), label.findtext("text")
            if group is not None:
                tags[group.strip()] = (text or "").strip()
        out.append({"code": (inst.findtext("code") or "").strip(),
                    "start": int(round(float(start) * fps)), "end": int(round(float(end) * fps)),
                    "tags": tags})
    return out


def kickoff_windows(events: list) -> list:
    """[(start, end)] of the 'Kick Off (Start)' set pieces, in time order."""
    wins = [(e["start"], e["end"]) for e in events
            if e["code"].lower() == "set pieces"
            and e["tags"].get("Type", "").lower() == "kick off (start)"]
    return sorted(wins)
