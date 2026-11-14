#!/usr/bin/env bash
# verify_deploy.sh FOLDER [FOLDER...]  -  does each folder hold exactly the files of the last build (dist)?
#   scripts/verify_deploy.sh /mnt/sda1/nexus /media/nexus/Ventoy/nexus
# Prints OK / DIFFERENT / MISSING for every engine file, with the file's date, so a stale copy is obvious.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DIST="${DIST:-$ROOT/dist}"
sha() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -c1-12; }
[ -d "$DIST" ] || { echo "no dist folder at $DIST"; exit 2; }
bad=0
for d in "$@"; do
  echo "== $d   (compared with $DIST)"
  for f in "$DIST"/nexus_server* "$DIST"/libnexus_* "$DIST"/nexus*.dll; do
    [ -f "$f" ] || continue; n="$(basename "$f")"
    if [ ! -f "$d/$n" ]; then printf '  MISSING    %s\n' "$n"; bad=1
    elif cmp -s "$f" "$d/$n"; then printf '  OK         %-26s %s\n' "$n" "$(date -r "$d/$n" '+%Y-%m-%d %H:%M')"
    else printf '  DIFFERENT  %-26s dist: %s [%s]   here: %s [%s]   <- STALE\n' "$n" "$(date -r "$f" '+%m-%d %H:%M')" "$(sha "$f")" "$(date -r "$d/$n" '+%m-%d %H:%M')" "$(sha "$d/$n")"; bad=1; fi
  done
done
exit $bad
