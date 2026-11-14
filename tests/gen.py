import numpy as np, struct
from gguf import quants, GGMLQuantizationType as T
rng = np.random.default_rng(7)
def f16bytes(v): return np.array([v], dtype=np.float16).tobytes()
N=64
for name, qt, bs, dpos in (('q2k', T.Q2_K, 84, [(80,0.001,0.01),(82,0.001,0.01)]), ('q3k', T.Q3_K, 110, [(108,0.0005,0.01)])):
    raw = bytearray(rng.integers(0,256,size=N*bs,dtype=np.uint8).tobytes())
    for b in range(N):
        for pos,lo,hi in dpos:
            raw[b*bs+pos:b*bs+pos+2] = f16bytes(rng.uniform(lo,hi))
    arr = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(N, bs)
    truth = quants.dequantize(arr, qt).astype(np.float32)
    open(name+'.blocks','wb').write(bytes(raw)); open(name+'.truth','wb').write(truth.tobytes())
    print(name, truth.shape, float(np.abs(truth).max()))
