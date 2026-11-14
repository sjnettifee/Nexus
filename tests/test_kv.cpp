#include "src/model/attention.h"
#include <random>
#include <cassert>
using namespace nexus; using namespace nexus::model;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
struct Ent { int i; float w; };
// mirror of Transformer::kv_rotate_positions for one head
static void rotate(KVCache& kv, int l, int h, int from, int to, int delta, const RoPEConfig& cfg) {
    const int hd = kv.head_dim_for_layer(l); const int ad = std::abs(delta);
    std::vector<std::vector<Ent>> col(hd), use(hd); std::vector<float> e(hd);
    for (int c = 0; c < hd; ++c) {
        std::fill(e.begin(), e.end(), 0.f); e[c] = 1.f; rope_apply_all(e.data(), 1, ad, hd, cfg);
        double n = 0; for (float v : e) n += (double)v * v; float inv = n > 0 ? 1.0 / std::sqrt(n) : 1;
        for (int r = 0; r < hd; ++r) if (e[r] != 0) col[c].push_back({r, e[r] * inv});
    }
    if (delta < 0) use = col; else for (int c = 0; c < hd; ++c) for (auto& en : col[c]) use[en.i].push_back({c, en.w});
    uint16_t* kp = kv.k_mut_ptr(l, h); std::vector<float> in(hd), out(hd);
    for (int p = from; p < to; ++p) { uint16_t* row = kp + (size_t)p * hd;
        for (int d = 0; d < hd; ++d) in[d] = quant::f16_to_f32(row[d]);
        for (int d = 0; d < hd; ++d) { float a = 0; for (auto& en : use[d]) a += en.w * in[en.i]; out[d] = a; }
        for (int d = 0; d < hd; ++d) row[d] = KVCache::f32_to_f16_rne(out[d]); }
}
int main() {
    // 1) RNE f16 vs compiler _Float16
    std::mt19937 rng(1); std::uniform_real_distribution<float> U(-1, 1);
    int bad = 0;
    for (int i = 0; i < 2000000; ++i) {
        uint32_t x = rng(); float f; memcpy(&f, &x, 4);
        if (std::isnan(f)) continue;
        _Float16 h = (_Float16)f; uint16_t ref; memcpy(&ref, &h, 2);
        if (KVCache::f32_to_f16_rne(f) != ref) { if (bad++ < 5) printf("rne mismatch %g: %04x vs %04x\n", f, KVCache::f32_to_f16_rne(f), ref); }
    }
    CHECK(bad == 0, "RNE mismatches: %d", bad);
    // 2) remove a middle range and re-rotate == keys written at the shifted positions
    for (int type : {0, 1, 2}) for (int layout : {0, 1}) {
        RoPEConfig cfg; cfg.freq_base = 1e6f; cfg.layout = layout; cfg.type = type;
        cfg.scale = type ? 4.f : 1.f; cfg.n_ctx_orig = 4096;
        const int hd = 128, N = 300, A = 40, B = 170;
        KVCache kv; kv.init(1, 1, hd);
        std::vector<std::vector<float>> raw(N, std::vector<float>(hd)); std::vector<float> vbuf(hd);
        for (int p = 0; p < N; ++p) { for (auto& v : raw[p]) v = U(rng); std::vector<float> k = raw[p];
            rope_apply_all(k.data(), 1, p, hd, cfg); for (int d = 0; d < hd; ++d) vbuf[d] = raw[p][d] * 0.5f; kv.store(0, p, k.data(), vbuf.data(), hd); }
        // export/archive middle range raw
        std::vector<uint16_t> ak((size_t)(B - A) * hd), av(ak.size());
        CHECK(kv.read_raw(0, 0, A, B - A, ak.data(), av.data()), "read_raw");
        kv.evict_range(A, B); CHECK(kv.n_tokens() == N - (B - A), "n after evict");
        rotate(kv, 0, 0, A, kv.n_tokens(), -(B - A), cfg);
        double maxerr = 0;
        for (int p = A; p < kv.n_tokens(); ++p) { std::vector<float> want = raw[p + (B - A)]; rope_apply_all(want.data(), 1, p, hd, cfg);
            std::vector<float> got(hd); kv.get_k_f32(0, p, 0, got.data());
            for (int d = 0; d < hd; ++d) maxerr = std::max(maxerr, (double)std::fabs(got[d] - want[d])); }
        // restore archived at the end: write raw, set n, rotate by pos0 - A
        const int pos0 = kv.n_tokens();
        CHECK(kv.write_raw(0, 0, pos0, B - A, ak.data(), av.data()), "write_raw");
        kv.set_n_tokens(pos0 + (B - A));
        rotate(kv, 0, 0, pos0, pos0 + (B - A), pos0 - A, cfg);
        double maxerr2 = 0, verr = 0;
        for (int i = 0; i < B - A; ++i) { std::vector<float> want = raw[A + i]; rope_apply_all(want.data(), 1, pos0 + i, hd, cfg);
            std::vector<float> got(hd), gv(hd); kv.get_k_f32(0, pos0 + i, 0, got.data()); kv.get_v_f32(0, pos0 + i, 0, gv.data());
            for (int d = 0; d < hd; ++d) { maxerr2 = std::max(maxerr2, (double)std::fabs(got[d] - want[d])); verr = std::max(verr, (double)std::fabs(gv[d] - raw[A+i][d]*0.5f)); } }
        printf("type=%d layout=%d  shift-back max err %.5f   restore max err %.5f  v err %.5f\n", type, layout, maxerr, maxerr2, verr);
        CHECK(maxerr < 0.02 && maxerr2 < 0.02 && verr < 0.002, "rotation error too large (type %d layout %d)", type, layout);
    }
    printf("bytes/token check: "); { KVCache kv; kv.init(32, 4, 256); printf("%zu (expect %d)\n", kv.bytes_per_token(), 32*4*256*2*2); CHECK(kv.bytes_per_token()==32*4*256*4, "bpt"); }
    printf(fails ? "%d FAILURE(S)\n" : "ALL PASSED\n", fails); return fails != 0;
}
