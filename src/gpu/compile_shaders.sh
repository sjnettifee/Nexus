#!/bin/bash
cd "$(dirname "$0")"
OUT="shader_spirv.h"
printf '// Auto-generated — do not edit\n#pragma once\n#include <cstdint>\n' > "$OUT"

compile() {
    local sym=$1 file=$2 spv="/tmp/${sym}.spv"
    if command -v glslc &>/dev/null; then
        glslc -o "$spv" "$file" || exit 1
    else
        glslangValidator -V -o "$spv" "$file" || exit 1
    fi
    python3 -c "
import sys, struct
d = open(sys.argv[1],'rb').read()
sym = sys.argv[2]
words = [struct.unpack_from('<I',d,i)[0] for i in range(0,len(d)-len(d)%4,4)]
print('static const uint32_t %s[] = {%s};' % (sym, ','.join(hex(w) for w in words)))
print('static const size_t %sSize = sizeof(%s);' % (sym, sym))
" "$spv" "$sym" >> "$OUT"
    rm -f "$spv"
}

compile kSpvMatVecF32   shaders/mat_vec_f32.comp
compile kSpvMatVecF16   shaders/mat_vec_f16.comp
compile kSpvMatVecQ8_0  shaders/mat_vec_q8_0.comp
compile kSpvMatMatQ8_0 shaders/mat_mat_q8_0.comp
compile kSpvMatVecQ4K   shaders/mat_vec_q4k.comp
compile kSpvMatMatQ4K  shaders/mat_mat_q4k.comp
compile kSpvMatVecQ5K   shaders/mat_vec_q5k.comp
compile kSpvMatMatQ5K  shaders/mat_mat_q5k.comp
compile kSpvMatVecQ6K   shaders/mat_vec_q6k.comp
compile kSpvMatMatQ6K  shaders/mat_mat_q6k.comp
compile kSpvMatVecNQ4G64 shaders/mat_vec_nq4_g64.comp
compile kSpvMatMatNQ4G64 shaders/mat_mat_nq4_g64.comp
compile kSpvMatVecNQ5G64 shaders/mat_vec_nq5_g64.comp
compile kSpvMatMatNQ5G64 shaders/mat_mat_nq5_g64.comp
compile kSpvMatVecNQ6G64 shaders/mat_vec_nq6_g64.comp
compile kSpvMatMatNQ6G64 shaders/mat_mat_nq6_g64.comp
compile kSpvMatVecNQ8G64 shaders/mat_vec_nq8_g64.comp
compile kSpvMatMatNQ8G64 shaders/mat_mat_nq8_g64.comp
compile kSpvQwen35Conv1D shaders/qwen35_conv1d.comp
compile kSpvQwen35Scan   shaders/qwen35_scan.comp
compile kSpvRope        shaders/rope.comp
compile kSpvRmsNorm     shaders/rms_norm.comp
compile kSpvSiluMul     shaders/silu_mul.comp
compile kSpvAddResidual shaders/add_residual.comp
compile kSpvAttnScore   shaders/attn_score.comp
compile kSpvAttnScoreBatch shaders/attn_score_batch.comp
compile kSpvAttnScoreStream shaders/attn_score_stream.comp
compile kSpvVisionAttnFull shaders/vision_attn_full.comp
echo "Done."
