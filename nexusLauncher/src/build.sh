#!/usr/bin/env bash
# Builds ONLY the launcher (independent of the engine).  Linux needs X11 headers at build time (libx11-dev);
# at run time X11 is loaded dynamically, so the binary starts on any desktop without extra packages.
# Each OS/drive gets its own build folder, and a folder that was configured somewhere else (e.g. on Windows, as D:/...) is discarded.
set -e
cd "$(dirname "$0")"
SRC="$(pwd)"
B="build-$(uname -s | tr 'A-Z' 'a-z')"
if [ -f "$B/CMakeCache.txt" ] && ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC" "$B/CMakeCache.txt"; then rm -rf "$B"; fi
mkdir -p "$B" && cd "$B"
cmake .. -DCMAKE_BUILD_TYPE=Release ${GEN:+-G "$GEN"}
cmake --build . -j
echo "Built: $(pwd)/nexus_launcher*"
echo "Put it next to nexus_server, with models/ and loras/ folders beside it."
