/*
 * nexus/src/model/audio_encoder.h
 *
 * Audio encoder for speech transcription and audio understanding.
 * Implements Whisper-compatible encoder (used by whisper.cpp models).
 *
 * Architecture:
 *   1. Log-Mel spectrogram: raw PCM → 80-bin mel features
 *   2. Conv stem: two 1D convolutions with GELU
 *   3. Sinusoidal positional embedding
 *   4. Transformer encoder: N layers of self-attention
 *   5. Output: per-frame embeddings OR sequence of audio tokens
 *
 * Supported:
 *   Whisper tiny/base/small/medium/large (encoder only — decoder is text model)
 *   Audio features can be injected into LLM token stream like vision tokens
 *
 * GGUF format: standard whisper.cpp GGUF files load directly.
 *
 * Usage:
 *   AudioEncoder enc;
 *   enc.load("whisper-small.gguf");
 *   std::vector<float> pcm = load_wav("audio.wav");  // 16kHz mono float32
 *   std::vector<float> tokens = enc.encode(pcm);
 *   // tokens is [n_frames * d_model] — inject into LLM or run Whisper decoder
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <numeric>

#include "../gguf/parser.h"
#include "../gguf/quant.h"

namespace nexus {
namespace audio {

// ── WAV / PCM helpers ─────────────────────────────────────────────────────────

struct WavInfo {
    int    sample_rate  = 0;
    int    channels     = 0;
    int    bits         = 0;
    size_t n_samples    = 0;
    bool   valid        = false;
};

// Load WAV file to mono float32 at original sample rate.
// Returns empty vector on failure.
inline std::vector<float> load_wav(const std::string& path, WavInfo* info_out = nullptr) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};

    // RIFF header
    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); return {}; }

    uint32_t file_size; fread(&file_size, 4, 1, f);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); return {}; }

    uint16_t audio_fmt = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0, data_size = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            uint32_t byte_rate; uint16_t block_align;
            fread(&audio_fmt,   2, 1, f);
            fread(&channels,    2, 1, f);
            fread(&sample_rate, 4, 1, f);
            fread(&byte_rate,   4, 1, f);
            fread(&block_align, 2, 1, f);
            fread(&bits,        2, 1, f);
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            break;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (audio_fmt != 1 || channels == 0 || bits == 0 || data_size == 0) {
        fclose(f); return {};
    }

    size_t n_samples = data_size / (bits / 8) / channels;
    std::vector<float> out(n_samples);

    // Read samples, convert to float, mix to mono
    if (bits == 16) {
        std::vector<int16_t> raw(n_samples * channels);
        fread(raw.data(), 2, n_samples * channels, f);
        for (size_t i = 0; i < n_samples; ++i) {
            float v = 0.f;
            for (int c = 0; c < channels; ++c) v += raw[i * channels + c];
            out[i] = v / (channels * 32768.0f);
        }
    } else if (bits == 32) {
        std::vector<int32_t> raw(n_samples * channels);
        fread(raw.data(), 4, n_samples * channels, f);
        for (size_t i = 0; i < n_samples; ++i) {
            float v = 0.f;
            for (int c = 0; c < channels; ++c) v += raw[i * channels + c];
            out[i] = v / (channels * 2147483648.0f);
        }
    }

    fclose(f);
    if (info_out) {
        info_out->sample_rate = sample_rate;
        info_out->channels    = channels;
        info_out->bits        = bits;
        info_out->n_samples   = n_samples;
        info_out->valid       = true;
    }
    return out;
}

// Resample PCM to target_rate using linear interpolation.
inline std::vector<float> resample(const std::vector<float>& pcm,
                                    int src_rate, int dst_rate)
{
    if (src_rate == dst_rate) return pcm;
    size_t n_out = (size_t)((double)pcm.size() * dst_rate / src_rate);
    std::vector<float> out(n_out);
    double ratio = (double)pcm.size() / n_out;
    for (size_t i = 0; i < n_out; ++i) {
        double pos = i * ratio;
        size_t i0 = (size_t)pos;
        size_t i1 = std::min(i0 + 1, pcm.size() - 1);
        float t = (float)(pos - i0);
        out[i] = pcm[i0] * (1.f - t) + pcm[i1] * t;
    }
    return out;
}

// ── Log-Mel spectrogram ───────────────────────────────────────────────────────

struct MelConfig {
    int    n_fft        = 400;    // FFT window size (25ms at 16kHz)
    int    hop_length   = 160;    // hop size (10ms at 16kHz)
    int    n_mel        = 80;     // mel filterbank bins
    int    sample_rate  = 16000;
    float  f_min        = 0.0f;
    float  f_max        = 8000.0f;
    int    chunk_length = 30;     // seconds per chunk (Whisper: 30s)
};

// FFT using Cooley-Tukey (in-place, size must be power of 2)
inline void fft(std::vector<std::complex<float>>& x) {
    int n = (int)x.size();
    if (n <= 1) return;
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        std::complex<float> wlen(cosf(ang), sinf(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1, 0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<float> u = x[i + j];
                std::complex<float> v = x[i + j + len/2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Compute log-mel spectrogram from mono 16kHz PCM.
// Output: [n_mel, n_frames] row-major
inline std::vector<float> log_mel_spectrogram(
    const std::vector<float>& pcm,
    const MelConfig& cfg = {})
{
    const int n_fft      = cfg.n_fft;
    const int hop        = cfg.hop_length;
    const int n_mel      = cfg.n_mel;
    const int fft_size   = 1 << (int)ceil(log2(n_fft));

    // Build Hann window
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n_fft - 1)));

    // Pad PCM
    int max_samples = cfg.chunk_length * cfg.sample_rate;
    std::vector<float> padded(max_samples + n_fft, 0.f);
    size_t copy_n = std::min((size_t)max_samples, pcm.size());
    memcpy(padded.data() + n_fft / 2, pcm.data(), copy_n * sizeof(float));

    // Mel filterbank
    auto hz_to_mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float m)  { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); };

    float mel_lo = hz_to_mel(cfg.f_min);
    float mel_hi = hz_to_mel(cfg.f_max);
    std::vector<float> mel_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i)
        mel_pts[i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (n_mel + 1));

    std::vector<float> fbank(n_mel * (fft_size / 2 + 1), 0.f);
    for (int m = 0; m < n_mel; ++m) {
        float lo = mel_pts[m], ctr = mel_pts[m + 1], hi = mel_pts[m + 2];
        for (int k = 0; k <= fft_size / 2; ++k) {
            float hz = (float)k * cfg.sample_rate / fft_size;
            float v = 0.f;
            if      (hz >= lo && hz <= ctr) v = (hz - lo)  / (ctr - lo);
            else if (hz >  ctr && hz <= hi) v = (hi - hz)  / (hi - ctr);
            fbank[m * (fft_size / 2 + 1) + k] = v;
        }
    }

    // STFT → mel → log
    int n_frames = (max_samples) / hop;
    std::vector<float> mel_spec(n_mel * n_frames, 0.f);

    std::vector<std::complex<float>> frame(fft_size);
    for (int t = 0; t < n_frames; ++t) {
        // Apply window
        for (int i = 0; i < n_fft; ++i)
            frame[i] = (i < n_fft) ? std::complex<float>(padded[t * hop + i] * window[i], 0.f)
                                    : std::complex<float>(0.f, 0.f);
        for (int i = n_fft; i < fft_size; ++i) frame[i] = 0.f;

        fft(frame);

        // Power spectrum
        for (int m = 0; m < n_mel; ++m) {
            float energy = 0.f;
            for (int k = 0; k <= fft_size / 2; ++k) {
                float mag2 = frame[k].real() * frame[k].real() +
                             frame[k].imag() * frame[k].imag();
                energy += fbank[m * (fft_size / 2 + 1) + k] * mag2;
            }
            mel_spec[m * n_frames + t] = energy;
        }
    }

    // Log-compress + normalize
    float max_val = *std::max_element(mel_spec.begin(), mel_spec.end());
    max_val = std::max(max_val, 1e-10f);
    for (auto& v : mel_spec) {
        v = log10f(std::max(v, 1e-10f));
        v = std::max(v, log10f(max_val) - 8.0f);  // Whisper: clamp to max-8
        v = (v + 4.0f) / 4.0f;                     // normalize to ~[-1, 1]
    }
    return mel_spec;  // [n_mel, n_frames]
}

// ── Audio Encoder ─────────────────────────────────────────────────────────────

struct AudioConfig {
    int n_mel        = 80;
    int n_audio_ctx  = 1500;   // max audio frames (30s / 20ms)
    int n_audio_head = 8;
    int n_audio_layer= 6;      // Whisper small: 6
    int n_audio_state= 512;    // Whisper small: 512
    int sample_rate  = 16000;
    MelConfig mel;
};

struct AudioLayerWeights {
    std::vector<float> attn_ln_w, attn_ln_b;
    std::vector<float> wq, bq, wk, bk, wv, bv, wo, bo;
    std::vector<float> ffn_ln_w, ffn_ln_b;
    std::vector<float> fc1_w, fc1_b, fc2_w, fc2_b;
};

class AudioEncoder {
public:
    bool load(const std::string& gguf_path,
              std::function<void(const std::string&, float)> progress_cb = nullptr);
    bool loaded() const { return loaded_; }
    const AudioConfig& config() const { return cfg_; }

    // Encode mono 16kHz PCM → audio token embeddings [n_frames * d_model]
    std::vector<float> encode(const std::vector<float>& pcm_16k);

    // Encode pre-computed log-mel [n_mel, n_frames]
    std::vector<float> encode_mel(const std::vector<float>& mel_spec);

    // Load and encode a WAV file (auto-resamples to 16kHz)
    std::vector<float> encode_wav(const std::string& wav_path);

private:
    bool loaded_ = false;
    AudioConfig cfg_;

    // Conv stem weights (Whisper uses 2x conv1d)
    std::vector<float> conv1_w_, conv1_b_;  // [n_state, n_mel, 3]
    std::vector<float> conv2_w_, conv2_b_;  // [n_state, n_state, 3]
    std::vector<float> pos_emb_;            // [n_ctx, n_state]
    std::vector<float> ln_post_w_, ln_post_b_;
    std::vector<AudioLayerWeights> layers_;

    static void layer_norm(float* x, const float* w, const float* b, int n);
    static void gelu(float* x, int n);
    std::vector<float> conv1d(const std::vector<float>& x, int in_ch, int out_ch,
                               int seq_len, const float* w, const float* b,
                               int kernel=3, int stride=1);
};

inline void AudioEncoder::layer_norm(float* x, const float* w, const float* b, int n) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    var = sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; ++i)
        x[i] = (x[i] - mean) / var * (w ? w[i] : 1.f) + (b ? b[i] : 0.f);
}

inline void AudioEncoder::gelu(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

inline std::vector<float> AudioEncoder::conv1d(
    const std::vector<float>& x, int in_ch, int out_ch, int seq_len,
    const float* w, const float* b, int kernel, int stride)
{
    int out_len = (seq_len - kernel) / stride + 1;
    std::vector<float> out(out_ch * out_len, 0.f);
    for (int o = 0; o < out_ch; ++o) {
        for (int t = 0; t < out_len; ++t) {
            float v = b ? b[o] : 0.f;
            for (int k = 0; k < kernel; ++k)
                for (int c = 0; c < in_ch; ++c)
                    v += w[o * in_ch * kernel + c * kernel + k] *
                         x[c * seq_len + t * stride + k];
            out[o * out_len + t] = v;
        }
    }
    return out;
}

inline bool AudioEncoder::load(
    const std::string& path,
    std::function<void(const std::string&, float)> progress_cb)
{
    gguf::Parser parser;
    if (!parser.open(path)) {
        fprintf(stderr, "[audio] Failed to open %s\n", path.c_str());
        return false;
    }

    cfg_.n_mel         = (int)parser.get_u32("whisper.encoder.mels",              80);
    cfg_.n_audio_ctx   = (int)parser.get_u32("whisper.encoder.context_length",    1500);
    cfg_.n_audio_head  = (int)parser.get_u32("whisper.encoder.attention_heads",   8);
    cfg_.n_audio_layer = (int)parser.get_u32("whisper.encoder.layer_count",       6);
    cfg_.n_audio_state = (int)parser.get_u32("whisper.encoder.embedding_length",  512);
    cfg_.mel.n_mel     = cfg_.n_mel;

    // Read file into buffer for manual tensor extraction
    FILE* ff = fopen(path.c_str(), "rb");
    std::vector<uint8_t> filebuf;
    if (ff) {
        fseek(ff, 0, SEEK_END); long fsz = ftell(ff); rewind(ff);
        filebuf.resize(fsz);
        (void)fread(filebuf.data(), 1, fsz, ff);
        fclose(ff);
    }
    size_t total_tbytes = 0;
    for (auto& [n, t] : parser.tensors()) total_tbytes += t.data_size;
    size_t raw_data_start = filebuf.empty() ? 0 :
        ((filebuf.size() - total_tbytes) & ~(size_t)31);

    auto load_f32 = [&](const std::string& name, std::vector<float>& out) {
        gguf::TensorDesc* td = parser.find_tensor(name);
        if (!td || filebuf.empty()) return;
        size_t n = td->n_elements;
        if (!n) { n = 1; for (auto d : td->shape) n *= d; }
        out.resize(n);
        quant::QuantType qt;
        switch (td->type) {
            case gguf::GGMLType::F32: qt = quant::QuantType::F32; break;
            case gguf::GGMLType::F16: qt = quant::QuantType::F16; break;
            default: qt = quant::QuantType::F32; break;
        }
        const uint8_t* base = filebuf.data() + raw_data_start + td->offset;
        quant::dequantize(base, out.data(), n, qt);
    };

    load_f32("encoder.conv1.weight", conv1_w_);
    load_f32("encoder.conv1.bias",   conv1_b_);
    load_f32("encoder.conv2.weight", conv2_w_);
    load_f32("encoder.conv2.bias",   conv2_b_);
    load_f32("encoder.positional_embedding", pos_emb_);
    load_f32("encoder.ln_post.weight", ln_post_w_);
    load_f32("encoder.ln_post.bias",   ln_post_b_);

    layers_.resize(cfg_.n_audio_layer);
    for (int i = 0; i < cfg_.n_audio_layer; ++i) {
        auto& lw = layers_[i];
        std::string pfx = "encoder.blocks." + std::to_string(i) + ".";
        load_f32(pfx + "attn_ln.weight",   lw.attn_ln_w);
        load_f32(pfx + "attn_ln.bias",     lw.attn_ln_b);
        load_f32(pfx + "attn.query.weight", lw.wq);
        load_f32(pfx + "attn.query.bias",   lw.bq);
        load_f32(pfx + "attn.key.weight",   lw.wk);
        load_f32(pfx + "attn.value.weight", lw.wv);
        load_f32(pfx + "attn.value.bias",   lw.bv);
        load_f32(pfx + "attn.out.weight",   lw.wo);
        load_f32(pfx + "attn.out.bias",     lw.bo);
        load_f32(pfx + "mlp_ln.weight",    lw.ffn_ln_w);
        load_f32(pfx + "mlp_ln.bias",      lw.ffn_ln_b);
        load_f32(pfx + "mlp.0.weight",     lw.fc1_w);
        load_f32(pfx + "mlp.0.bias",       lw.fc1_b);
        load_f32(pfx + "mlp.2.weight",     lw.fc2_w);
        load_f32(pfx + "mlp.2.bias",       lw.fc2_b);
        if (progress_cb)
            progress_cb("Loading audio encoder", (float)(i+1)/cfg_.n_audio_layer);
    }

    loaded_ = true;
    fprintf(stderr, "[audio] Loaded %s: %d mel, %d ctx, %d layers, d=%d\n",
            path.c_str(), cfg_.n_mel, cfg_.n_audio_ctx,
            cfg_.n_audio_layer, cfg_.n_audio_state);
    return true;
}

inline std::vector<float> AudioEncoder::encode_wav(const std::string& path) {
    WavInfo info;
    auto pcm = load_wav(path, &info);
    if (pcm.empty()) {
        fprintf(stderr, "[audio] Failed to load WAV: %s\n", path.c_str());
        return {};
    }
    if (info.sample_rate != 16000)
        pcm = resample(pcm, info.sample_rate, 16000);
    return encode(pcm);
}

inline std::vector<float> AudioEncoder::encode(const std::vector<float>& pcm) {
    auto mel = log_mel_spectrogram(pcm, cfg_.mel);
    return encode_mel(mel);
}

inline std::vector<float> AudioEncoder::encode_mel(const std::vector<float>& mel_spec) {
    const int d     = cfg_.n_audio_state;
    const int n_mel = cfg_.n_mel;
    int n_frames    = (int)mel_spec.size() / n_mel;

    // ── Conv stem ─────────────────────────────────────────────────────────────
    // conv1: [n_mel, n_frames] → [d, n_frames], kernel=3, stride=1
    auto h1 = conv1d(mel_spec, n_mel, d, n_frames,
                     conv1_w_.data(), conv1_b_.data(), 3, 1);
    gelu(h1.data(), (int)h1.size());
    int t1 = (n_frames - 3) / 1 + 1;

    // conv2: [d, t1] → [d, t2], kernel=3, stride=2
    auto h2 = conv1d(h1, d, d, t1, conv2_w_.data(), conv2_b_.data(), 3, 2);
    gelu(h2.data(), (int)h2.size());
    int t2 = (t1 - 3) / 2 + 1;
    t2 = std::min(t2, cfg_.n_audio_ctx);

    // ── Positional embedding ──────────────────────────────────────────────────
    // h2 is [d, t2] — transpose to [t2, d] for transformer
    std::vector<float> tokens(t2 * d);
    for (int t = 0; t < t2; ++t)
        for (int c = 0; c < d; ++c)
            tokens[t * d + c] = h2[c * ((t1 - 3) / 2 + 1) + t];

    if (!pos_emb_.empty()) {
        for (int t = 0; t < t2; ++t)
            for (int c = 0; c < d; ++c)
                tokens[t * d + c] += pos_emb_[t * d + c];
    }

    // ── Transformer encoder layers ────────────────────────────────────────────
    const int nh = cfg_.n_audio_head;
    const int hd = d / nh;
    std::vector<float> buf(t2 * d);
    std::vector<float> Q(t2 * d), K(t2 * d), V(t2 * d);
    std::vector<float> attn(t2 * t2);

    for (int layer = 0; layer < cfg_.n_audio_layer; ++layer) {
        auto& lw = layers_[layer];
        if (lw.attn_ln_w.empty()) continue;

        // Pre-norm attention
        for (int t = 0; t < t2; ++t) {
            memcpy(buf.data() + t * d, tokens.data() + t * d, d * sizeof(float));
            layer_norm(buf.data() + t * d, lw.attn_ln_w.data(), lw.attn_ln_b.data(), d);
        }

        // Q, K, V
        for (int t = 0; t < t2; ++t) {
            const float* x = buf.data() + t * d;
            for (int j = 0; j < d; ++j) {
                float vq = lw.bq.empty() ? 0.f : lw.bq[j];
                float vv = lw.bv.empty() ? 0.f : lw.bv[j];
                for (int k = 0; k < d; ++k) {
                    vq += lw.wq[j*d+k] * x[k];
                    vv += lw.wv[j*d+k] * x[k];
                    // K has no bias in Whisper
                }
                Q[t*d+j] = vq; V[t*d+j] = vv;
                float vk = 0.f;
                for (int k = 0; k < d; ++k) vk += lw.wk[j*d+k] * x[k];
                K[t*d+j] = vk;
            }
        }

        // Attention
        float scale = 1.0f / sqrtf((float)hd);
        std::vector<float> attn_out(t2 * d, 0.f);
        for (int h = 0; h < nh; ++h) {
            int off = h * hd;
            for (int i = 0; i < t2; ++i)
                for (int j = 0; j < t2; ++j) {
                    float s = 0.f;
                    for (int k = 0; k < hd; ++k)
                        s += Q[i*d+off+k] * K[j*d+off+k];
                    attn[i*t2+j] = s * scale;
                }
            for (int i = 0; i < t2; ++i) {
                float mx = *std::max_element(attn.data()+i*t2, attn.data()+(i+1)*t2);
                float sm = 0.f;
                for (int j = 0; j < t2; ++j) sm += (attn[i*t2+j] = expf(attn[i*t2+j]-mx));
                for (int j = 0; j < t2; ++j) attn[i*t2+j] /= sm;
            }
            for (int i = 0; i < t2; ++i)
                for (int k = 0; k < hd; ++k) {
                    float s = 0.f;
                    for (int j = 0; j < t2; ++j) s += attn[i*t2+j] * V[j*d+off+k];
                    attn_out[i*d+off+k] = s;
                }
        }

        // Output projection + residual
        for (int t = 0; t < t2; ++t) {
            const float* x = attn_out.data() + t * d;
            for (int j = 0; j < d; ++j) {
                float v = lw.bo.empty() ? 0.f : lw.bo[j];
                for (int k = 0; k < d; ++k) v += lw.wo[j*d+k] * x[k];
                tokens[t*d+j] += v;
            }
        }

        // Pre-norm FFN
        for (int t = 0; t < t2; ++t) {
            memcpy(buf.data()+t*d, tokens.data()+t*d, d*sizeof(float));
            layer_norm(buf.data()+t*d, lw.ffn_ln_w.data(), lw.ffn_ln_b.data(), d);
        }

        // FFN
        int ffn_d = (int)lw.fc1_b.size();
        for (int t = 0; t < t2; ++t) {
            const float* x = buf.data() + t * d;
            std::vector<float> h(ffn_d);
            for (int j = 0; j < ffn_d; ++j) {
                float v = lw.fc1_b.empty() ? 0.f : lw.fc1_b[j];
                for (int k = 0; k < d; ++k) v += lw.fc1_w[j*d+k] * x[k];
                h[j] = v;
            }
            gelu(h.data(), ffn_d);
            for (int j = 0; j < d; ++j) {
                float v = lw.fc2_b.empty() ? 0.f : lw.fc2_b[j];
                for (int k = 0; k < ffn_d; ++k) v += lw.fc2_w[j*ffn_d+k] * h[k];
                tokens[t*d+j] += v;
            }
        }
    }

    // Final layer norm
    if (!ln_post_w_.empty())
        for (int t = 0; t < t2; ++t)
            layer_norm(tokens.data()+t*d, ln_post_w_.data(), ln_post_b_.data(), d);

    return tokens;  // [t2 * d_model]
}

} // namespace audio
} // namespace nexus
