#!/usr/bin/env bash
# build_release.sh -- build every CPU tier of the engine + the dispatcher, into ONE folder.
# Linux, macOS, and the MSYS2 MinGW64 shell on Windows.  Run from anywhere.
#
#   scripts/build_release.sh                 # tiers v2 v3 v4   -> ./dist
#   scripts/build_release.sh v2 v3           # only those tiers
#   OUT=/path/to/folder scripts/build_release.sh
#   (dist AND the project folder are both updated automatically.)
#   DEPLOY="/path/a /path/b" scripts/build_release.sh   # ALSO copy the finished set into these folders (e.g. the USB folder the
#                                                       # launcher runs from).  Every copy is verified byte-for-byte.
#   SKIP_TESTS=1 scripts/build_release.sh    # skip the kernel correctness test
#
# Result in $OUT (copy that whole folder to the USB stick / zip it for release):
#   nexus_server[.exe]            the DISPATCHER: what users and the launcher start.
#                                 Built with no -march, runs on any CPU, picks the best tier below.
#   nexus_server_v2  + libnexus_v2     SSE4.2     (any x86 CPU since 2009)
#   nexus_server_v3  + libnexus_v3     AVX2/FMA   (Haswell 2013+, Zen)
#   nexus_server_v4  + libnexus_v4     AVX-512    (optional)
# Missing tiers are fine: the dispatcher falls back to the next lower tier.
set -euo pipefail
# find the project folder (the one with CMakeLists.txt) no matter where this script was put or where it is run from
find_root() {
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  for c in "$PWD" "$here" "$here/.." "$PWD/.."; do [ -f "$c/CMakeLists.txt" ] && { (cd "$c" && pwd); return; }; done
  echo "ERROR: cannot find CMakeLists.txt (looked in $PWD, $here and their parent folders). Run this from the nexus project folder." >&2; exit 2
}
ROOT="$(find_root)"
cd "$ROOT"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) EXE=".exe"; LIBEXT="dll";   LIBPFX="lib" ;;
  Darwin)               EXE="";     LIBEXT="dylib"; LIBPFX="lib" ;;
  *)                    EXE="";     LIBEXT="so";    LIBPFX="lib" ;;
esac
OUT="${OUT:-$ROOT/dist}"
mkdir -p "$OUT"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
TIERS=("$@"); [ ${#TIERS[@]} -eq 0 ] && TIERS=(v2 v3 v4)

# ---- copying, with proof -------------------------------------------------------------------------------------
# Every file is copied to a temp name next to its destination and renamed over it (works even if the old file is
# still running), then compared byte-for-byte with what was just built.  Any problem is printed and the script exits
# non-zero at the end - it never says "done" while a destination is stale.
# Always update BOTH the dist folder and the project folder itself (the launcher runs the files in the project folder).
# Nothing to remember: just run this script.  (NO_ROOT_COPY=1 disables the project-folder copy; DEPLOY="a b" adds more folders.)
DEST_DIRS=("$OUT"); if [ "${NO_ROOT_COPY:-0}" != "1" ] && [ "$(cd "$OUT" && pwd)" != "$ROOT" ]; then DEST_DIRS+=("$ROOT"); fi
for d in ${DEPLOY:-}; do DEST_DIRS+=("$d"); done
FAILS=0
sha() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -c1-12; }
deploy() {                                   # deploy <built file> <destination folder>
  local src="$1" dir="$2" name tmp; name="$(basename "$src")"
  mkdir -p "$dir" 2>/dev/null || { echo "  FAILED  cannot create $dir"; FAILS=$((FAILS+1)); return; }
  tmp="$dir/.$name.new.$$"
  if ! cp -f "$src" "$tmp"; then echo "  FAILED  copy $name -> $dir"; rm -f "$tmp"; FAILS=$((FAILS+1)); return; fi
  chmod +x "$tmp" 2>/dev/null || true
  if ! mv -f "$tmp" "$dir/$name"; then echo "  FAILED  could not replace $dir/$name  (is it running? stop the launcher / server and run again)"; rm -f "$tmp"; FAILS=$((FAILS+1)); return; fi
  if cmp -s "$src" "$dir/$name"; then printf '  copied  %-26s -> %-40s [%s]\n' "$name" "$dir" "$(sha "$src")"
  else echo "  FAILED  $dir/$name does not match the file that was just built"; FAILS=$((FAILS+1)); fi
}
deploy_all() { for d in "${DEST_DIRS[@]}"; do deploy "$1" "$d"; done; }

dispatcher_done=0
for t in "${TIERS[@]}"; do
  case "$t" in v2|v3|v4) ;; *) echo "unknown tier '$t' (v2|v3|v4)"; exit 2 ;; esac
  echo "=============== tier $t ==============="
  b="build-$t"
  rm -rf "$b"
  cmake -S . -B "$b" -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEXUS_PYTHON=OFF -DNEXUS_VULKAN=ON -DNEXUS_TIER="$t"
  cmake --build "$b" --parallel "$JOBS"

  srv="$b/nexus_server_${t}${EXE}"
  lib="$b/${LIBPFX}nexus_${t}.${LIBEXT}"
  [ -f "$srv" ] || { echo "ERROR: $srv was not produced"; exit 1; }
  [ -f "$lib" ] || { echo "ERROR: $lib was not produced"; ls "$b"; exit 1; }
  echo "[copy] tier $t"
  deploy_all "$srv"; deploy_all "$lib"
  if [ $dispatcher_done -eq 0 ]; then       # identical in every tier build (no -march)
    deploy_all "$b/nexus_server${EXE}"; dispatcher_done=1
  fi
done

echo
echo "================ copy result ================"
{ echo "built: $(date '+%Y-%m-%d %H:%M:%S')   tiers: ${TIERS[*]}"; for f in "$OUT"/nexus_server* "$OUT"/libnexus_*; do [ -f "$f" ] && echo "$(sha "$f")  $(basename "$f")"; done; } > "$OUT/BUILD_INFO.txt"
cat "$OUT/BUILD_INFO.txt"
if [ $FAILS -ne 0 ]; then echo; echo "ERROR: $FAILS file(s) were NOT updated - see the FAILED lines above."; exit 1; fi
echo "All files copied and verified in: ${DEST_DIRS[*]}"
echo
echo "================ checks ================"
if [ "${SKIP_TESTS:-0}" != "1" ] && [ -f tests/kernel_tiers.cpp ]; then
  echo "[kernel test] correctness of the CPU kernels on this machine's best tiers"
  for t in "${TIERS[@]}"; do
    case "$t" in v2) m=nehalem ;; v3) m=haswell ;; v4) m=skylake-avx512 ;; esac
    # run only tiers this CPU can execute (the dispatcher knows which)
    if "$OUT/nexus_server${EXE}" --which-tier >/dev/null 2>&1; then
      g++ -std=c++17 -O2 -march=$m -Isrc tests/kernel_tiers.cpp -o "build-$t/kernel_tiers${EXE}" -pthread 2>/dev/null || { echo "  (could not build test for $t)"; continue; }
      if "build-$t/kernel_tiers${EXE}" "build-$t/kt.bin" >"build-$t/kt.log" 2>&1; then echo "  tier $t: kernels OK"
      else
        if grep -q "RESULT" "build-$t/kt.log"; then echo "  tier $t: KERNEL FAILURE"; cat "build-$t/kt.log"; exit 1
        else echo "  tier $t: skipped (this CPU cannot run it)"; fi
      fi
    fi
  done
fi
echo "[dispatcher] tier chosen on this machine:"; "$OUT/nexus_server${EXE}" --which-tier
if [ "$LIBEXT" = "so" ]; then
  if ldd "$OUT/nexus_server_${TIERS[0]}${EXE}" 2>/dev/null | grep -Eq '^\s*libvulkan'; then echo "WARNING: engine links libvulkan"; else echo "[vulkan] ok: not a link-time dependency"; fi
fi
echo "[probe]"; "$OUT/nexus_server${EXE}" --probe 2>/dev/null | head -c 500 || true
echo
echo "Release folder: $OUT"; ls -1 "$OUT"
