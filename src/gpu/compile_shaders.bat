@echo off
setlocal
set OUTFILE=%~dp0shader_spirv.h
echo // Auto-generated — do not edit > "%OUTFILE%"
echo #pragma once >> "%OUTFILE%"
echo #include ^<cstdint^> >> "%OUTFILE%"

call :compile mat_vec_f32.comp  kSpvMatVecF32
call :compile mat_vec_f16.comp  kSpvMatVecF16
call :compile mat_vec_q8_0.comp kSpvMatVecQ8_0
call :compile mat_vec_q4k.comp  kSpvMatVecQ4K
call :compile mat_vec_q5k.comp  kSpvMatVecQ5K
call :compile mat_vec_q6k.comp  kSpvMatVecQ6K
call :compile mat_mat_q8_0.comp kSpvMatMatQ8_0
call :compile mat_mat_q4k.comp  kSpvMatMatQ4K
call :compile mat_mat_q5k.comp  kSpvMatMatQ5K
call :compile mat_mat_q6k.comp  kSpvMatMatQ6K
call :compile mat_vec_nq4_g64.comp kSpvMatVecNQ4G64
call :compile mat_mat_nq4_g64.comp kSpvMatMatNQ4G64
call :compile mat_vec_nq5_g64.comp kSpvMatVecNQ5G64
call :compile mat_mat_nq5_g64.comp kSpvMatMatNQ5G64
call :compile mat_vec_nq6_g64.comp kSpvMatVecNQ6G64
call :compile mat_mat_nq6_g64.comp kSpvMatMatNQ6G64
call :compile mat_vec_nq8_g64.comp kSpvMatVecNQ8G64
call :compile mat_mat_nq8_g64.comp kSpvMatMatNQ8G64
call :compile qwen35_conv1d.comp kSpvQwen35Conv1D
call :compile qwen35_scan.comp   kSpvQwen35Scan
call :compile rope.comp         kSpvRope
call :compile rms_norm.comp     kSpvRmsNorm
call :compile silu_mul.comp     kSpvSiluMul
call :compile add_residual.comp kSpvAddResidual
call :compile attn_score.comp   kSpvAttnScore
call :compile attn_score_batch.comp kSpvAttnScoreBatch
call :compile attn_score_stream.comp kSpvAttnScoreStream
call :compile vision_attn_full.comp kSpvVisionAttnFull
echo Done.
goto :eof

:compile
set SHADER=%~dp0shaders\%1
set SYM=%2
set SPV=%TEMP%\nexus_%2.spv

rem cmake sets GLSLC=GLSLC-NOTFOUND when not found — check the value, not just existence
set USE_GLSLC=0
if defined GLSLC (
    echo %GLSLC% | findstr /i "NOTFOUND" >nul || set USE_GLSLC=1
)
if "%USE_GLSLC%"=="1" (
    "%GLSLC%" -o "%SPV%" "%SHADER%"
) else if defined GLSLANG (
    "%GLSLANG%" -V -S comp "%SHADER%" -o "%SPV%"
) else (
    glslangValidator -V -S comp "%SHADER%" -o "%SPV%"
)
if errorlevel 1 ( echo ERROR compiling %1 & exit /b 1 )
python -c "import sys;d=open(sys.argv[1],'rb').read();n=len(d);n=n-(n&3);vals=','.join(hex(int.from_bytes(d[i:i+4],'little')) for i in range(0,n,4));sym=sys.argv[2];print('static const uint32_t '+sym+'[]={'+vals+'};');print('static const size_t '+sym+'Size=sizeof('+sym+');')" "%SPV%" "%SYM%" >> "%OUTFILE%"
del "%SPV%"
goto :eof
