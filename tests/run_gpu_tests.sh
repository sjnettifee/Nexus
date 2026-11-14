#!/usr/bin/env bash
# Runs the REAL Vulkan backend against the CPU kernels on a software Vulkan device (Mesa
# lavapipe) -- a second, independent Vulkan implementation, so NVIDIA-only assumptions in the
# shaders show up here.  Needs:  sudo apt install mesa-vulkan-drivers   and a finished build
# (the generated shader header lives in build-<tier>/gen).
#   tests/run_gpu_tests.sh [tier]       (default v3)
set -euo pipefail
cd "$(dirname "$0")/.."
T="${1:-v3}"; GEN="build-$T/gen"
[ -f "$GEN/nexus_shader_spirv.h" ] || { echo "build first (scripts/build_release.sh $T)"; exit 1; }
ICD=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json 2>/dev/null | head -1 || true)
[ -n "$ICD" ] || { echo "lavapipe not installed (apt install mesa-vulkan-drivers)"; exit 1; }
case "$T" in v2) M=nehalem ;; v3) M=haswell ;; v4) M=skylake-avx512 ;; esac
g++ -std=c++17 -O1 -march=$M -DNEXUS_VULKAN -DNEXUS_SHADERS_EMBEDDED -DVK_NO_PROTOTYPES \
    -Isrc -I"$GEN" tests/gpu_test.cpp -o "build-$T/gpu_test" -ldl -pthread
export VK_ICD_FILENAMES="$ICD" NEXUS_VULKAN_ALLOW_SOFTWARE=1
"build-$T/gpu_test" 2>&1 | sed -n '/^type/,$p'
echo "--- lm_head-sized matrix (must fall back to the CPU on a 128 MiB device, not mis-compute):"
"build-$T/gpu_test" big 2>&1 | grep -E "weight .* MB|^type|^NQ|RESULT"
