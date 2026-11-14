static void replay(const L& lw_, bool grouped, const float* mixed, const float* a, const float* b, int keep, std::vector<float>& conv_state, std::vector<float>& rec_state){
 struct R{const float*md,*ad,*bd; struct V{const float*p; const float*data()const{return p;}} mixed,a,b;} r{mixed,a,b,{mixed},{a},{b}};
 struct LW{int q35_conv_dim,q35_key_dim,q35_num_k_heads,q35_num_v_heads,q35_head_dim,q35_kernel; const float*q35_conv,*q35_A,*q35_dt;} lw{lw_.conv_dim,lw_.key_dim,lw_.kh,lw_.vh,lw_.hd,lw_.kernel,lw_.conv.data(),lw_.A.data(),lw_.dt.data()};
 const int conv_dim = lw.q35_conv_dim, key_dim = lw.q35_key_dim;
 const int kh = lw.q35_num_k_heads, vh = lw.q35_num_v_heads, hd = lw.q35_head_dim, kernel = lw.q35_kernel;
 const int repeat = kh > 0 ? vh / kh : 0;
 auto nx_l2norm_inplace=[](float*x,int n,float e){l2n(x,n,e);}; auto nx_sigmoid_f32=sig; auto nx_softplus_f32=sp;
    std::vector<float> convt((size_t)conv_dim), k_flat((size_t)key_dim), v_flat((size_t)vh * hd);
    for (int t = 0; t < keep; ++t) {
        const float* mixt = r.mixed.data() + (size_t)t * (size_t)conv_dim;
        for (int c = 0; c < conv_dim; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < kernel; ++k) {
                const float xv = (k < kernel - 1) ? conv_state[(size_t)c * (size_t)(kernel - 1) + (size_t)k] : mixt[c];
                sum += xv * lw.q35_conv[(size_t)c * (size_t)kernel + (size_t)k];
            }
            convt[(size_t)c] = sum / (1.0f + expf(-sum));
        }
        if (kernel > 1)
            for (int c = 0; c < conv_dim; ++c) {
                float* st = conv_state.data() + (size_t)c * (size_t)(kernel - 1);
                for (int k = 0; k < kernel - 2; ++k) st[k] = st[k + 1];
                st[kernel - 2] = mixt[c];
            }
        const float* v_base;
        if (grouped) {
            const int span = 2 * hd + repeat * hd;
            for (int hk = 0; hk < kh; ++hk) {
                const size_t src = (size_t)hk * (size_t)span;
                std::memcpy(k_flat.data() + (size_t)hk * hd, convt.data() + src + (size_t)hd, (size_t)hd * sizeof(float));
                for (int rr = 0; rr < repeat; ++rr)
                    std::memcpy(v_flat.data() + ((size_t)hk * repeat + rr) * hd,
                                convt.data() + src + (size_t)(2 * hd) + (size_t)rr * hd, (size_t)hd * sizeof(float));
            }
            v_base = v_flat.data();
        } else {
            std::memcpy(k_flat.data(), convt.data() + key_dim, (size_t)key_dim * sizeof(float));
            v_base = convt.data() + 2 * (size_t)key_dim;
        }
        for (int hk = 0; hk < kh; ++hk) nx_l2norm_inplace(k_flat.data() + (size_t)hk * hd, hd, 1e-6f);
        const float* at = r.a.data() + (size_t)t * (size_t)vh;
        const float* bt = r.b.data() + (size_t)t * (size_t)vh;
        for (int hv = 0; hv < vh; ++hv) {
            const float* k = k_flat.data() + (size_t)(hv / repeat) * hd;
            const float* v = v_base + (size_t)hv * hd;
            const float beta = nx_sigmoid_f32(bt[hv]);
            float g = lw.q35_A[hv] * nx_softplus_f32(at[hv] + lw.q35_dt[hv]);
            if (!std::isfinite(g)) g = -60.0f;
            if (g > 0.0f) g = 0.0f;
            if (g < -60.0f) g = -60.0f;
            const float decay = expf(g);
            float* S = rec_state.data() + (size_t)hv * hd * hd;
            for (int i = 0; i < hd * hd; ++i) S[i] *= decay;
            float kv_mem[256], delta[256];
            for (int j = 0; j < hd; ++j) { float s2 = 0.0f; for (int i = 0; i < hd; ++i) s2 += k[i] * S[(size_t)i * hd + j]; kv_mem[j] = s2; }
            for (int j = 0; j < hd; ++j) delta[j] = (v[j] - kv_mem[j]) * beta;
            for (int i = 0; i < hd; ++i) { const float ki = k[i]; float* Sr = S + (size_t)i * hd; for (int j = 0; j < hd; ++j) Sr[j] += ki * delta[j]; }
        }
    }
}
