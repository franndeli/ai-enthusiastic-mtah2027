#!/bin/sh
# Report which match assets are present in the private data dir: names and sizes only, never content.
# Safe for agents to run. Expected layout: data/README.md and DATA.md.
DATA_DIR="${MTAH_DATA_DIR:-$HOME/.mtah2027/data}"
echo "data dir: $DATA_DIR"
for f in \
  tactical/BAR-ATH_tactical.mp4 tactical/BAR-ATH_tactical_atd.xml tactical/BAR-ATH_tactical_atd.txt \
  tactical/BAR-ATH_tactical_homography.csv tactical/BAR-ATH_tactical_smart_tagging.xml \
  broadcast/BAR-ATH_broadcast.mp4 broadcast/BAR-ATH_broadcast_atd.xml broadcast/BAR-ATH_broadcast_atd.txt \
  broadcast/BAR-ATH_broadcast_homography.csv
do
  if [ -f "$DATA_DIR/$f" ]; then
    printf '  ok       %-45s %s\n' "$f" "$(du -h "$DATA_DIR/$f" | cut -f1)"
  else
    printf '  MISSING  %s\n' "$f"
  fi
done
