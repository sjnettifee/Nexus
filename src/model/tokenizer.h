/*
 * nexus/src/model/tokenizer.h
 *
 * BPE tokenizer that reads directly from GGUF metadata.
 *
 * Supports:
 *   - SentencePiece BPE (LLaMA-1/2, Mistral, Qwen)
 *   - Tiktoken BPE (LLaMA-3, Qwen2.5)
 *   - Byte-fallback encoding for unknown characters
 *
 * GGUF keys we read:
 *   tokenizer.ggml.model       — "llama", "gpt2", "bert"
 *   tokenizer.ggml.tokens      — vocab list
 *   tokenizer.ggml.scores      — merge scores (SentencePiece log-probs)
 *   tokenizer.ggml.token_type  — 0=normal,1=unknown,2=control,3=user_defined,4=unused,5=byte
 *   tokenizer.ggml.merges      — merge rules (tiktoken / GPT-2 style)
 *   tokenizer.ggml.bos_token_id
 *   tokenizer.ggml.eos_token_id
 *   tokenizer.ggml.padding_token_id
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "../gguf/parser.h"

namespace nexus {
namespace model {

// Token type flags (from GGUF spec)
enum class TokenType : int {
    NORMAL      = 1,
    UNKNOWN     = 2,
    CONTROL     = 3,
    USER_DEFINED= 4,
    UNUSED      = 5,
    BYTE        = 6,
};

struct TokenizerVocab {
    std::vector<std::string>  tokens;      // token string for each id
    std::vector<float>        scores;      // merge scores (log-probs)
    std::vector<TokenType>    types;       // token type per id
    std::unordered_map<std::string, int32_t> token_to_id;

    int32_t bos_id     = 1;
    int32_t eos_id     = 2;
    int32_t unk_id     = 0;
    int32_t pad_id     = -1;
    int32_t nl_id      = -1;  // newline token
    std::string model_type;   // "llama", "gpt2", etc.
};

// ─── BPE merge pair ──────────────────────────────────────────────────────────

struct MergePair {
    int32_t left;
    int32_t right;
    int32_t result;
    float   score;   // lower index = higher priority
};

// ─── Main Tokenizer class ────────────────────────────────────────────────────

class Tokenizer {
public:
    Tokenizer() = default;

    // Load from GGUF parser (call after parser.open())
    bool load(const gguf::Parser& parser);

    // Encode text to token IDs
    // add_bos: prepend BOS token (recommended for generation)
    // add_eos: append EOS token (use for training, not generation)
    std::vector<int32_t> encode(const std::string& text,
                                 bool add_bos = true,
                                 bool add_eos = false) const;

    // Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& tokens,
                       bool skip_special = true) const;

    // Decode a single token
    std::string decode_token(int32_t id) const;

    // Check special tokens
    bool is_special(int32_t id) const;
    bool is_eos(int32_t id) const { return id == vocab_.eos_id; }
    bool is_bos(int32_t id) const { return id == vocab_.bos_id; }

    int32_t bos_id() const { return vocab_.bos_id; }
    int32_t eos_id() const { return vocab_.eos_id; }
    int32_t unk_id() const { return vocab_.unk_id; }

    size_t vocab_size() const { return vocab_.tokens.size(); }

    // Find token ID for a string (exact match)
    int32_t token_to_id(const std::string& s) const;

    const TokenizerVocab& vocab() const { return vocab_; }

    void print_info() const;

private:
    TokenizerVocab vocab_;
    std::vector<MergePair> merges_;

    // For fast merge lookup: (left_id, right_id) → merge result
    std::unordered_map<uint64_t, MergePair> merge_map_;

    bool is_loaded_ = false;
    bool is_sentencepiece_ = false;  // vs tiktoken/GPT2

    // Qwen-family byte-level BPE with the proper HF pre-tokenizer regex split
    // (see qwen_pretok.h). Without it, greedy longest-match across word boundaries
    // yields non-canonical token splits the model never saw in training.
    bool qwen_pretok_ = false;
    std::vector<std::string> byte_chars_;   // byte -> GPT-2 unicode vocab string

    // Special-token-aware pre-pass: scans for CONTROL/USER_DEFINED tokens
    // and obvious special marker strings (<|im_start|>, <|im_end|>, etc.)
    // and emits their IDs directly,
    // running encode_spm/encode_gpt2 only on the non-special text segments.
    std::vector<int32_t> encode_with_specials(const std::string& text) const;

    // SentencePiece BPE encode
    std::vector<int32_t> encode_spm(const std::string& text) const;

    // GPT2/tiktoken BPE encode
    std::vector<int32_t> encode_gpt2(const std::string& text) const;

    // Byte fallback: encode a single UTF-8 byte as a token
    int32_t byte_token(uint8_t byte) const;

    // Clean up SentencePiece ▁ → space, handle escapes
    std::string clean_token(const std::string& tok) const;

    // Build merge map from merge rules
    void build_merge_map();

    static uint64_t merge_key(int32_t left, int32_t right) {
        return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    }
};

} // namespace model
} // namespace nexus
