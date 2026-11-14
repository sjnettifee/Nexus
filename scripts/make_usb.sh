#!/usr/bin/env bash
# make_usb.sh -- assemble the portable release folder (copy it to a USB stick, or zip it).
#
#   scripts/make_usb.sh [OUT_DIR]                 default OUT_DIR = ./nexus-usb
#   DIST=/path/to/engine/files  LAUNCHERS="/path/nexus_launcher /path/nexus_launcher.exe"  scripts/make_usb.sh /media/usb/Nexus
#
# Result:
#   nexus_launcher / nexus_launcher.exe   top level: what the user double-clicks (whichever OSes you provide)
#   models/   loras/   nexus-data/        the user's files (chats, projects and launcher settings live in nexus-data)
#   server/                               nexus_server[.exe] + nexus_server_v2/v3/v4 + libnexus_v2/v3/v4 (.so and/or .dll), together
#   README.txt
# Linux and Windows files can sit side by side: their names never clash.
set -euo pipefail
OUT="${1:-./nexus-usb}"
DIST="${DIST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dist}"
LAUNCHERS="${LAUNCHERS:-$DIST/nexus_launcher $DIST/nexus_launcher.exe}"
[ -d "$DIST" ] || { echo "ERROR: engine folder not found: $DIST  (run scripts/build_release.sh first, or set DIST=)"; exit 2; }
mkdir -p "$OUT/models" "$OUT/loras" "$OUT/nexus-data" "$OUT/server"

n_srv=0; n_l=0
for f in "$DIST"/nexus_server "$DIST"/nexus_server.exe "$DIST"/nexus_server_v[234] "$DIST"/nexus_server_v[234].exe \
         "$DIST"/libnexus_v[234].so "$DIST"/libnexus_v[234].dll "$DIST"/libnexus_v[234].dylib; do
  [ -f "$f" ] || continue
  cp -f "$f" "$OUT/server/"; chmod +x "$OUT/server/$(basename "$f")" 2>/dev/null || true; n_srv=$((n_srv+1))
done
for f in $LAUNCHERS; do
  [ -f "$f" ] || continue
  cp -f "$f" "$OUT/"; chmod +x "$OUT/$(basename "$f")" 2>/dev/null || true; n_l=$((n_l+1))
done
[ $n_srv -gt 0 ] || { echo "ERROR: no server files found in $DIST"; exit 1; }
[ $n_l -gt 0 ]   || echo "WARNING: no launcher found (set LAUNCHERS=...). The folder still works with nexus_server directly."
[ -f "$OUT/models/PUT_MODELS_HERE.txt" ] || echo "Put model files (.gguf or .nxm) in this folder. The launcher finds them on any computer." > "$OUT/models/PUT_MODELS_HERE.txt"
[ -f "$OUT/loras/PUT_LORAS_HERE.txt" ]   || echo "Put LoRA adapter .gguf files in this folder." > "$OUT/loras/PUT_LORAS_HERE.txt"
cat > "$OUT/README.txt" <<'R'
NEXUS - portable
  Windows: double-click nexus_launcher.exe
  Linux:   run ./nexus_launcher   (if it will not start from a USB stick: the stick is mounted "noexec" or is FAT/exFAT without
           permissions. Copy this whole folder to your home folder, or run:  sh nexus_launcher_run.sh  to copy-and-run)

  models/      your model files (.gguf / .nxm)
  loras/       optional LoRA adapters
  nexus-data/  your chats, projects and the launcher's settings (back this folder up; copy it to keep your stuff)
  server/      the engine (nexus_server + versions for older/newer CPUs). Do not split these files up.

The first time, pick a model in the launcher and press Start. Nothing is installed and nothing is written outside this folder.
R
# Linux helper for sticks that cannot run programs in place: copy the program part to a temp folder and run it there, using this
# folder's models and data.
cat > "$OUT/nexus_launcher_run.sh" <<'R'
#!/bin/sh
# Copies the program to a temporary folder (so it can run even from a noexec stick) and starts it there.
# Models and settings stay on the stick: they are linked in.
here="$(cd "$(dirname "$0")" && pwd)"
tmp="${TMPDIR:-/tmp}/nexus-run-$(id -u)"
rm -rf "$tmp"; mkdir -p "$tmp/server"
cp "$here/nexus_launcher" "$tmp/" && cp "$here"/server/nexus_server* "$here"/server/libnexus_* "$tmp/server/" 2>/dev/null
chmod +x "$tmp/nexus_launcher" "$tmp"/server/nexus_server* 
ln -s "$here/models" "$tmp/models"; ln -s "$here/loras" "$tmp/loras"; ln -s "$here/nexus-data" "$tmp/nexus-data"
cd "$tmp" && exec ./nexus_launcher "$@"
R
chmod +x "$OUT/nexus_launcher_run.sh" 2>/dev/null || true
echo "Release folder ready: $OUT"; (cd "$OUT" && find . -maxdepth 2 | sort | sed 's/^/  /')
