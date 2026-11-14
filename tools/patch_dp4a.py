#!/usr/bin/env python3
"""Post-process a glslang-compiled SPIR-V file: replace the body of the helper `int dp4(uint a, uint b, int c)` with OpSDotAccSat
(4x8-bit packed signed dot product with accumulate), add the DotProduct capabilities/extension, inline, and re-assemble.
usage: patch_dp4a.py in.spv out.spv     (needs spirv-dis, spirv-as, spirv-opt, spirv-val on PATH)"""
import subprocess, sys, re
src, dst = sys.argv[1], sys.argv[2]
txt = subprocess.check_output(['spirv-dis', '--raw-id', src] if False else ['spirv-dis', src]).decode()
lines = txt.split('\n')
# locate the dp4 function
name_id = None
for l in lines:
    m = re.match(r'\s*OpName (%\S+) "dp4\(', l)
    if m: name_id = m.group(1); break
if not name_id: sys.exit('dp4 helper not found')
start = next(i for i, l in enumerate(lines) if re.match(r'\s*%\S+ = OpFunction ', l) and name_id in l.split('=')[0].split()[0:1] + [l.split('=')[0].strip()])
end = next(i for i in range(start, len(lines)) if 'OpFunctionEnd' in lines[i])
hdr = lines[start]            # "%dp4 = OpFunction %int None %10"
params = [l for l in lines[start+1:end] if 'OpFunctionParameter' in l]
pa, pb, pc = [p.split('=')[0].strip() for p in params]
ptr_types = [p.split('OpFunctionParameter')[1].strip() for p in params]
int_t = hdr.split('OpFunction')[1].split()[0]
uint_t = ptr_types[0].replace('_ptr_Function_', '')   # not used for real type id; resolved below
# find OpTypeInt ids
uint_id = next(re.match(r'\s*(%\S+) = OpTypeInt 32 0', l).group(1) for l in lines if re.match(r'\s*(%\S+) = OpTypeInt 32 0', l))
int_id = int_t
body = [hdr] + params + [
    '%dpL = OpLabel',
    f'%dpa = OpLoad {uint_id} {pa}',
    f'%dpb = OpLoad {uint_id} {pb}',
    f'%dpc = OpLoad {int_id} {pc}',
    f'%dpr = OpSDotAccSat {int_id} %dpa %dpb %dpc PackedVectorFormat4x8Bit',
    'OpReturnValue %dpr',
    'OpFunctionEnd']
lines = lines[:start] + body + lines[end+1:]
out = '\n'.join(lines)
out = out.replace('OpCapability Shader', 'OpCapability Shader\n               OpCapability DotProduct\n               OpCapability DotProductInput4x8BitPacked\n               OpExtension "SPV_KHR_integer_dot_product"', 1)
open(dst + '.spvasm', 'w').write(out)
subprocess.check_call(['spirv-as', '--target-env', 'vulkan1.2', dst + '.spvasm', '-o', dst])
subprocess.check_call(['spirv-opt', '--inline-entry-points-exhaustive', '--eliminate-dead-functions', dst, '-o', dst])
subprocess.check_call(['spirv-val', '--target-env', 'vulkan1.2', dst])
