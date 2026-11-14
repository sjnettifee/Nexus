/*
 * nexus/src/model/tokenizer.cpp
 *
 * BPE tokenizer implementation.
 */

#include "tokenizer.h"
#include "qwen_pretok.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <sstream>

namespace nexus {
namespace model {

// ─── Load from GGUF parser ────────────────────────────────────────────────────

bool Tokenizer::load(const gguf::Parser& parser) {
    const auto& kv = parser.kv();

    // Model type
    vocab_.model_type = parser.get_str("tokenizer.ggml.model", "llama");

    // Determine if this is a SentencePiece tokenizer (uses ▁ space markers and
    // <0xXX> byte tokens) vs GPT-2/BPE style (uses Ġ/Ā/etc Unicode byte encoding).
    //
    // We can't rely solely on vocab_.model_type == "llama" because:
    //   - True Llama-2/Mistral: "llama" type, SentencePiece, ▁ markers ✓
    //   - Llama-3 derivatives: sometimes stored as "llama" type but use tiktoken BPE
    //   - Qwen/GPT variants: "gpt2" type, GPT-2 byte encoding
    //
    // Definitive test: does the vocabulary contain the ▁ marker (U+2581, UTF-8: E2 96 81)?
    // If any token starts with ▁, it's SentencePiece. GPT-2/tiktoken vocabs never contain it.
    is_sentencepiece_ = false;
    const std::string sp_mark = "\xe2\x96\x81";

    // Special token IDs
    vocab_.bos_id = (int32_t)parser.get_u32("tokenizer.ggml.bos_token_id", 1);
    vocab_.eos_id = (int32_t)parser.get_u32("tokenizer.ggml.eos_token_id", 2);
    vocab_.unk_id = (int32_t)parser.get_u32("tokenizer.ggml.unknown_token_id", 0);
    vocab_.pad_id = (int32_t)parser.get_u32("tokenizer.ggml.padding_token_id",
                                             (uint32_t)-1);

    // Token strings
    auto tok_it = kv.find("tokenizer.ggml.tokens");
    if (tok_it == kv.end()) {
        fprintf(stderr, "[tokenizer] No token list found in GGUF\n");
        return false;
    }

    const auto* tok_arr = std::get_if<std::shared_ptr<gguf::GGUFArray>>(&tok_it->second);
    if (!tok_arr || !*tok_arr) return false;

    const size_t n_vocab = (*tok_arr)->values.size();
    vocab_.tokens.reserve(n_vocab);

    for (size_t i = 0; i < n_vocab; ++i) {
        std::string tok;
        if (const auto* s = std::get_if<std::string>(&(*tok_arr)->values[i])) {
            tok = *s;
        }
        vocab_.tokens.push_back(tok);
        vocab_.token_to_id[tok] = (int32_t)i;
    }

    // Detect SentencePiece by scanning for the ▁ word-boundary marker (U+2581).
    // This is definitive: SentencePiece always uses it, GPT-2/tiktoken never does.
    // Only scan the first 2000 tokens — it appears in the first few hundred if present.
    {
        size_t scan = std::min(n_vocab, (size_t)2000);
        for (size_t i = 0; i < scan; ++i) {
            if (vocab_.tokens[i].find(sp_mark) != std::string::npos) {
                is_sentencepiece_ = true;
                break;
            }
        }
        fprintf(stderr, "[tokenizer] model_type=%s  sentencepiece=%s\n",
                vocab_.model_type.c_str(), is_sentencepiece_ ? "yes" : "no");
    }

    // Scores
    auto sc_it = kv.find("tokenizer.ggml.scores");
    if (sc_it != kv.end()) {
        const auto* sc_arr = std::get_if<std::shared_ptr<gguf::GGUFArray>>(&sc_it->second);
        if (sc_arr && *sc_arr) {
            vocab_.scores.reserve(n_vocab);
            for (auto& v : (*sc_arr)->values) {
                if (const auto* f = std::get_if<float>(&v))
                    vocab_.scores.push_back(*f);
                else
                    vocab_.scores.push_back(0.0f);
            }
        }
    }
    if (vocab_.scores.size() != n_vocab)
        vocab_.scores.resize(n_vocab, 0.0f);

    // Token types
    auto ty_it = kv.find("tokenizer.ggml.token_type");
    if (ty_it != kv.end()) {
        const auto* ty_arr = std::get_if<std::shared_ptr<gguf::GGUFArray>>(&ty_it->second);
        if (ty_arr && *ty_arr) {
            vocab_.types.reserve(n_vocab);
            for (auto& v : (*ty_arr)->values) {
                int type_val = 1;
                if (const auto* i = std::get_if<int32_t>(&v))  type_val = *i;
                if (const auto* i = std::get_if<uint32_t>(&v)) type_val = (int)*i;
                vocab_.types.push_back((TokenType)type_val);
            }
        }
    }
    if (vocab_.types.size() != n_vocab)
        vocab_.types.resize(n_vocab, TokenType::NORMAL);

    // Merge rules (GPT-2/tiktoken style)
    auto mg_it = kv.find("tokenizer.ggml.merges");
    if (mg_it != kv.end()) {
        const auto* mg_arr = std::get_if<std::shared_ptr<gguf::GGUFArray>>(&mg_it->second);
        if (mg_arr && *mg_arr) {
            int merge_idx = 0;
            for (auto& v : (*mg_arr)->values) {
                const std::string* rule = std::get_if<std::string>(&v);
                if (!rule) continue;

                // Merge rule format: "token_a token_b"
                size_t space = rule->find(' ');
                if (space == std::string::npos) continue;

                std::string left_str  = rule->substr(0, space);
                std::string right_str = rule->substr(space + 1);

                auto l_it = vocab_.token_to_id.find(left_str);
                auto r_it = vocab_.token_to_id.find(right_str);
                auto res_it = vocab_.token_to_id.find(left_str + right_str);

                if (l_it == vocab_.token_to_id.end() ||
                    r_it == vocab_.token_to_id.end() ||
                    res_it == vocab_.token_to_id.end()) {
                    ++merge_idx;
                    continue;
                }

                MergePair mp;
                mp.left   = l_it->second;
                mp.right  = r_it->second;
                mp.result = res_it->second;
                mp.score  = (float)merge_idx;  // lower = higher priority
                merges_.push_back(mp);
                ++merge_idx;
            }
            build_merge_map();
            // Do NOT override is_sentencepiece_ here.
            // Phi-3 and several other models use SentencePiece BPE but also
            // include merge rules in their GGUF. The â marker scan
            // above is the authoritative signal. If we already detected
            // SentencePiece tokens, having merges doesn't change that.
            // Only treat as GPT-2 style if no SentencePiece markers were found.
            if (!is_sentencepiece_) {
                // confirmed GPT-2/tiktoken: has merges and no â tokens
            }
        }
    }

    // Find newline token
    for (const auto& [tok, id] : vocab_.token_to_id) {
        if (tok == "\n" || tok == "<0x0A>") {
            vocab_.nl_id = id;
            break;
        }
    }

    // Qwen family (Qwen2/2.5/3/3.5): ChatML markers + byte-level BPE merges.
    // Use the exact HF pre-tokenizer split. NEXUS_TOK_LEGACY=1 restores the old path.
    {
        const char* legacy = std::getenv("NEXUS_TOK_LEGACY");
        const bool force_legacy = legacy && legacy[0] == '1';
        const bool is_qwen =
            !is_sentencepiece_ && !merge_map_.empty() &&
            vocab_.token_to_id.count("<|im_start|>") &&
            vocab_.token_to_id.count("<|endoftext|>");
        qwen_pretok_ = is_qwen && !force_legacy;
        if (qwen_pretok_) {
            // byte -> GPT-2 unicode char (same table as encode_gpt2)
            std::vector<int> bs, cs;
            for (int b = '!'; b <= '~'; ++b) { bs.push_back(b); cs.push_back(b); }
            for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); cs.push_back(b); }
            for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); cs.push_back(b); }
            int n = 0;
            for (int b = 0; b < 256; ++b) {
                bool found = false;
                for (int x : bs) if (x == b) { found = true; break; }
                if (!found) { bs.push_back(b); cs.push_back(256 + n++); }
            }
            byte_chars_.assign(256, std::string());
            for (int i = 0; i < 256; ++i) {
                const uint32_t cp = (uint32_t)cs[i];
                std::string s;
                if (cp < 0x80) s += (char)cp;
                else { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
                byte_chars_[(size_t)bs[i]] = s;
            }
        }
        fprintf(stderr, "[tokenizer] pre-tokenizer: %s\n",
                qwen_pretok_ ? "Qwen regex split (exact)" : "legacy greedy longest-match");
    }

    fprintf(stderr, "[tokenizer] Loaded: %zu tokens, model=%s, bos=%d, eos=%d\n",
            n_vocab, vocab_.model_type.c_str(), vocab_.bos_id, vocab_.eos_id);

    is_loaded_ = true;
    return true;
}

void Tokenizer::build_merge_map() {
    merge_map_.clear();
    merge_map_.reserve(merges_.size());
    for (const auto& mp : merges_) {
        uint64_t key = merge_key(mp.left, mp.right);
        // Keep highest priority (lowest score) merge
        auto it = merge_map_.find(key);
        if (it == merge_map_.end() || mp.score < it->second.score) {
            merge_map_[key] = mp;
        }
    }
}

// ─── Special-token pre-pass ───────────────────────────────────────────────────
//
// Production tokenizers (HF tokenizers, llama.cpp, tiktoken) do special-token
// scanning BEFORE byte-level/BPE encoding. Without this pass, greedy
// longest-match in encode_gpt2() can collide with adjacent characters and
// shred a special token across multiple regular tokens.
//
// Concrete failure mode for Qwen3 ChatML:
//   ".<|im_end|>"  →  longest-match finds ".<" (token 15757), then "|i", "m",
//                     "_end", "|", ">\n" — six tokens instead of one (151645).
// The model then sees a corrupted user-turn boundary, generates a couple of
// reasonable tokens, then degrades as KV cache references the malformed
// position relationships.
//
// Fix: scan the input string for occurrences of any token whose type is
// CONTROL or USER_DEFINED. Sort matches by length (longest first) so longer
// special tokens win over shorter ones at the same position. Emit special
// tokens as their direct vocab IDs, and run the regular encoder only on the
// in-between segments.
//
// Returns the token list with special tokens already resolved and surrounding
// text segments encoded with the regular path.
//
// Cost: one O(vocab) pass at first call to build the special-token list,
// cached after that. Per-encode cost is O(text_len * n_specials) which for
// Qwen3 (~30 special tokens) is negligible vs the BPE pass that follows.
static bool nx_token_looks_special(const std::string& s) {
    // Main-route special-token detection. This is NOT model-specific and does
    // not hardcode token IDs. It catches chat/template markers that are present
    // in the vocabulary even when token_type metadata is missing or wrong.
    //
    // Examples:
    //   <|im_start|>, <|im_end|>, <|endoftext|>
    //   <think>, </think>
    //   <s>, </s>
    if (s.size() < 3 || s[0] != '<') return false;
    if (s.find('|') != std::string::npos && s.back() == '>') return true;
    if (s.size() >= 7 && s.rfind("<think>", 0) == 0) return true;
    if (s.size() >= 8 && s.rfind("</think>", 0) == 0) return true;
    if (s == "<s>" || s == "</s>") return true;
    return false;
}

std::vector<int32_t> Tokenizer::encode_with_specials(const std::string& text) const {
    // Lazy-init: gather special tokens and sort by length desc.
    //
    // A token is considered special if either:
    //   1. tokenizer.ggml.token_type marks it CONTROL / USER_DEFINED / UNKNOWN, or
    //   2. the token string itself is an obvious chat/control marker.
    //
    // The second rule is important for native .nxm and some GGUFs because
    // token_type metadata can be incomplete even though tokenizer.ggml.tokens
    // contains the exact marker token. This keeps special-token support in the
    // tokenizer's normal encode path instead of pushing model-specific hacks
    // into Engine::tokenize().
    static thread_local std::vector<std::pair<std::string, int32_t>> specials_cache;
    static thread_local const TokenizerVocab* cached_for = nullptr;

    if (cached_for != &vocab_) {
        specials_cache.clear();

        for (size_t id = 0; id < vocab_.tokens.size(); ++id) {
            const std::string& s = vocab_.tokens[id];
            if (s.empty()) continue;

            bool special_by_type = false;
            if (id < vocab_.types.size()) {
                const TokenType t = vocab_.types[id];
                special_by_type =
                    t == TokenType::CONTROL ||
                    t == TokenType::USER_DEFINED ||
                    t == TokenType::UNKNOWN;
            }

            if (special_by_type || nx_token_looks_special(s)) {
                // Do not treat one-byte punctuation as a special boundary.
                // The string-shape check above avoids that, but keep this as
                // a final safety guard for malformed token_type metadata.
                if (s.size() >= 3) {
                    specials_cache.emplace_back(s, (int32_t)id);
                }
            }
        }

        // Longest first so e.g. "<|im_start|>" wins over "<|" or "<".
        std::sort(specials_cache.begin(), specials_cache.end(),
            [](const auto& a, const auto& b) {
                if (a.first.size() != b.first.size())
                    return a.first.size() > b.first.size();
                return a.first < b.first;
            });

        // De-duplicate identical token strings, preserving the first/lowest
        // sorted entry. Duplicate strings should be rare, but this keeps encode
        // deterministic if they appear.
        specials_cache.erase(
            std::unique(specials_cache.begin(), specials_cache.end(),
                [](const auto& a, const auto& b) { return a.first == b.first; }),
            specials_cache.end());

        cached_for = &vocab_;

        const int32_t im_start = token_to_id("<|im_start|>");
        const int32_t im_end   = token_to_id("<|im_end|>");
        if (im_start != vocab_.unk_id || im_end != vocab_.unk_id) {
            fprintf(stderr,
                "[tokenizer] special-token main route active: specials=%zu im_start=%d im_end=%d\n",
                specials_cache.size(), im_start, im_end);
        }
    }

    std::vector<int32_t> result;
    if (specials_cache.empty()) {
        if (is_sentencepiece_) return encode_spm(text);
        return encode_gpt2(text);
    }

    std::string segment;
    segment.reserve(text.size());

    auto flush_segment = [&]() {
        if (segment.empty()) return;
        std::vector<int32_t> seg_tokens = is_sentencepiece_
            ? encode_spm(segment)
            : encode_gpt2(segment);
        result.insert(result.end(), seg_tokens.begin(), seg_tokens.end());
        segment.clear();
    };

    size_t pos = 0;
    while (pos < text.size()) {
        int32_t matched_id  = -1;
        size_t  matched_len = 0;

        for (const auto& [s, id] : specials_cache) {
            if (s.size() > text.size() - pos) continue;
            if (std::memcmp(text.data() + pos, s.data(), s.size()) == 0) {
                matched_id  = id;
                matched_len = s.size();
                break;
            }
        }

        if (matched_id >= 0) {
            flush_segment();
            result.push_back(matched_id);
            pos += matched_len;
        } else {
            segment.push_back(text[pos]);
            pos += 1;
        }
    }

    flush_segment();
    return result;
}

// ─── Encode ───────────────────────────────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                        bool add_bos, bool add_eos) const {
    std::vector<int32_t> result;

    if (add_bos) result.push_back(vocab_.bos_id);

    if (text.empty()) {
        if (add_eos) result.push_back(vocab_.eos_id);
        return result;
    }

    // Route through the special-token-aware pre-pass. This guarantees
    // tokens like <|im_start|> / <|im_end|> are emitted as their direct
    // vocab IDs instead of being shredded by greedy longest-match in the
    // byte-level encoder. The pre-pass falls back to encode_spm/encode_gpt2
    // for the regular text segments between special tokens, so behavior on
    // strings without any special tokens is identical to before.
    std::vector<int32_t> tokens = encode_with_specials(text);

    result.insert(result.end(), tokens.begin(), tokens.end());

    if (add_eos) result.push_back(vocab_.eos_id);
    return result;
}

// ─── SentencePiece BPE encode ─────────────────────────────────────────────────
// SentencePiece works differently from GPT2:
//   1. Prepend ▁ to indicate word start (or after space)
//   2. Start with single-char tokens
//   3. Greedily merge by score (highest score = lowest negative log-prob)

std::vector<int32_t> Tokenizer::encode_spm(const std::string& text) const {
    // SentencePiece: convert spaces to ▁, then do BPE merges by score
    const std::string sp_space = "\xe2\x96\x81";  // UTF-8 for ▁

    // Pre-tokenize: split on whitespace, prepend ▁ to each word
    // (SentencePiece actually works on the whole string with ▁ replacing spaces)
    std::string normalized = text;

    // Replace spaces with ▁
    // But also prepend ▁ if text doesn't start with space
    std::string sp_text = sp_space;  // always start with word-start marker
    for (char c : text) {
        if (c == ' ') {
            sp_text += sp_space;
        } else {
            sp_text += c;
        }
    }

    // Initialize as individual characters (byte-level if no token found)
    // Each UTF-8 character or byte → token id
    std::vector<int32_t> tokens;

    // Walk through sp_text finding the longest matching token
    size_t pos = 0;
    while (pos < sp_text.size()) {
        // Try to find the longest matching token
        int best_id = -1;
        size_t best_len = 0;

        // Try from longest possible substring down to 1 byte
        // Limit search length to avoid O(n²) on long tokens
        size_t max_len = std::min(sp_text.size() - pos, (size_t)32);

        for (size_t len = max_len; len >= 1; --len) {
            std::string sub = sp_text.substr(pos, len);
            auto it = vocab_.token_to_id.find(sub);
            if (it != vocab_.token_to_id.end()) {
                best_id  = it->second;
                best_len = len;
                break;
            }
        }

        if (best_id >= 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            // Byte fallback
            uint8_t byte = (uint8_t)sp_text[pos];
            tokens.push_back(byte_token(byte));
            pos += 1;
        }
    }

    // BPE merge pass: greedily merge by score
    // Score: higher = better (SentencePiece uses negative log-prob)
    bool changed = true;
    while (changed) {
        changed = false;
        if (tokens.size() < 2) break;

        float best_score = -1e38f;
        size_t best_pos  = SIZE_MAX;
        int32_t best_result = -1;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            // Try merging tokens[i] + tokens[i+1]
            std::string merged = vocab_.tokens[tokens[i]] + vocab_.tokens[tokens[i+1]];
            auto it = vocab_.token_to_id.find(merged);
            if (it == vocab_.token_to_id.end()) continue;

            int32_t result_id = it->second;
            float score = (result_id < (int32_t)vocab_.scores.size())
                ? vocab_.scores[result_id] : -1e38f;

            if (score > best_score) {
                best_score  = score;
                best_pos    = i;
                best_result = result_id;
            }
        }

        if (best_pos != SIZE_MAX) {
            tokens[best_pos] = best_result;
            tokens.erase(tokens.begin() + best_pos + 1);
            changed = true;
        }
    }

    return tokens;
}

// ─── GPT-2/tiktoken BPE encode ───────────────────────────────────────────────
// GPT-2 BPE requires every byte of input to first be mapped through the
// bytes_to_unicode table (space→Ġ, newline→Ċ, etc.), THEN matched against
// the vocabulary.  Skipping this step means space-prefixed tokens like
// "Ġhello" are never found, causing byte-level fallback and broken output.

// Build the bytes→unicode forward map (same as openai/gpt-2 encoder.py)
static void build_bytes_to_unicode(uint32_t out[256]) {
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); cs.push_back(b); }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool found = false;
        for (int x : bs) if (x == b) { found = true; break; }
        if (!found) { bs.push_back(b); cs.push_back(256 + n++); }
    }
    for (int i = 0; i < 256; ++i) out[(uint8_t)bs[i]] = (uint32_t)cs[i];
}

// Encode a unicode codepoint as UTF-8
static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

std::vector<int32_t> Tokenizer::encode_gpt2(const std::string& text) const {
    if (text.empty()) return {};

    if (qwen_pretok_) {
        std::vector<int32_t> out;
        out.reserve(text.size() / 3 + 8);
        for (const auto& pr : qwen_pretok::split(text))
            qwen_pretok::bpe_piece(text.substr(pr.first, pr.second), byte_chars_,
                                   vocab_.token_to_id, merge_map_, vocab_.unk_id, out);
        return out;
    }

    // Step 1: Convert every byte of input through bytes_to_unicode
    // This turns space (0x20) → Ġ (U+0120), newline (0x0A) → Ċ (U+010A), etc.
    static bool b2u_built = false;
    static uint32_t b2u[256] = {};
    if (!b2u_built) { build_bytes_to_unicode(b2u); b2u_built = true; }

    // Build the unicode-encoded version of the input
    // Each byte of input becomes one unicode codepoint (possibly multi-byte UTF-8)
    std::string encoded;
    encoded.reserve(text.size() * 2);
    for (unsigned char c : text) {
        encoded += cp_to_utf8(b2u[c]);
    }

    // Step 2: Greedy longest-match against the vocabulary on the encoded string
    std::vector<int32_t> tokens;
    tokens.reserve(text.size());

    size_t pos = 0;
    while (pos < encoded.size()) {
        int best_id  = -1;
        size_t best_len = 0;
        size_t max_len = std::min(encoded.size() - pos, (size_t)64);

        for (size_t len = max_len; len >= 1; --len) {
            std::string sub = encoded.substr(pos, len);
            // Trim to valid UTF-8 boundary
            auto it = vocab_.token_to_id.find(sub);
            if (it != vocab_.token_to_id.end()) {
                best_id  = it->second;
                best_len = len;
                break;
            }
        }

        if (best_id >= 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            // Single codepoint fallback — take one UTF-8 codepoint
            unsigned char c = (unsigned char)encoded[pos];
            int bytes = 1;
            if      ((c & 0xE0) == 0xC0) bytes = 2;
            else if ((c & 0xF0) == 0xE0) bytes = 3;
            else if ((c & 0xF8) == 0xF0) bytes = 4;
            bytes = std::min(bytes, (int)(encoded.size() - pos));
            std::string single = encoded.substr(pos, bytes);
            auto it = vocab_.token_to_id.find(single);
            if (it != vocab_.token_to_id.end())
                tokens.push_back(it->second);
            else
                tokens.push_back(vocab_.unk_id);
            pos += bytes;
        }
    }

    // Step 3: BPE merge by priority
    if (merge_map_.empty()) return tokens;

    bool changed = true;
    while (changed && tokens.size() >= 2) {
        changed = false;
        float best_score  = 1e38f;
        size_t best_pos   = SIZE_MAX;
        int32_t best_result = -1;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            uint64_t key = merge_key(tokens[i], tokens[i+1]);
            auto it = merge_map_.find(key);
            if (it == merge_map_.end()) continue;
            if (it->second.score < best_score) {
                best_score  = it->second.score;
                best_pos    = i;
                best_result = it->second.result;
            }
        }

        if (best_pos != SIZE_MAX) {
            tokens[best_pos] = best_result;
            tokens.erase(tokens.begin() + best_pos + 1);
            changed = true;
        }
    }

    return tokens;
}

// ─── Byte fallback ────────────────────────────────────────────────────────────

int32_t Tokenizer::byte_token(uint8_t byte) const {
    // Try <0xXX> format (LLaMA)
    char buf[8];
    snprintf(buf, sizeof(buf), "<0x%02X>", byte);
    auto it = vocab_.token_to_id.find(buf);
    if (it != vocab_.token_to_id.end()) return it->second;

    // Try single character
    std::string c(1, (char)byte);
    it = vocab_.token_to_id.find(c);
    if (it != vocab_.token_to_id.end()) return it->second;

    return vocab_.unk_id;
}

// ─── Decode ───────────────────────────────────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int32_t>& tokens,
                               bool skip_special) const {
    std::string result;
    for (int32_t id : tokens) {
        if (skip_special && is_special(id)) continue;
        result += decode_token(id);  // decode_token already calls clean_token internally
    }
    return result;
}

std::string Tokenizer::decode_token(int32_t id) const {
    if (id < 0 || id >= (int32_t)vocab_.tokens.size()) return "";
    return clean_token(vocab_.tokens[id]);
}

// ─── GPT-2 byte-to-unicode reverse map ───────────────────────────────────────
// GPT-2 encodes every byte as a unicode character so the vocab is all-printable.
// The mapping encodes bytes 0-255 into a subset of unicode codepoints.
// We need to reverse this to get actual bytes out.
// Standard GPT-2 mapping (from openai/gpt-2 encoder.py):
//   Printable ASCII (33-126) + some Latin-1 (161-172, 174-255) → themselves
//   Everything else → chr(256 + offset) where offset is index in the remainder list
static std::string gpt2_byte_decode(const std::string& tok) {
    // Build reverse table on first call (256 entries: unicode cp → byte value)
    static bool built = false;
    static uint8_t rev[65536] = {};   // indexed by unicode codepoint, 0=unmapped
    static bool mapped[65536] = {};
    if (!built) {
        // Forward map: byte → codepoint
        int codepoint[256];
        // Pass 1: printable ASCII + selected Latin-1 → themselves
        std::vector<int> bs, cs;
        for (int b = '!'; b <= '~'; ++b) { bs.push_back(b); cs.push_back(b); }
        for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); cs.push_back(b); }
        for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); cs.push_back(b); }
        // Pass 2: remaining bytes → chr(256 + n)
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool found = false;
            for (int x : bs) if (x == b) { found = true; break; }
            if (!found) { bs.push_back(b); cs.push_back(256 + n++); }
        }
        for (int i = 0; i < 256; ++i) codepoint[bs[i]] = cs[i];
        // Populate reverse: codepoint → byte
        for (int b = 0; b < 256; ++b) {
            int cp = codepoint[b];
            if (cp < 65536) { rev[cp] = (uint8_t)b; mapped[cp] = true; }
        }
        built = true;
    }

    // Convert UTF-8 → codepoints → bytes
    std::string out;
    out.reserve(tok.size());
    size_t i = 0;
    while (i < tok.size()) {
        uint32_t cp = 0;
        unsigned char c = (unsigned char)tok[i];
        int bytes = 1;
        if      (c < 0x80)                        { cp = c; bytes = 1; }
        else if ((c & 0xE0) == 0xC0 && i+1 < tok.size()) { cp = (c & 0x1F) << 6 | ((unsigned char)tok[i+1] & 0x3F); bytes = 2; }
        else if ((c & 0xF0) == 0xE0 && i+2 < tok.size()) { cp = (c & 0x0F) << 12 | ((unsigned char)tok[i+1] & 0x3F) << 6 | ((unsigned char)tok[i+2] & 0x3F); bytes = 3; }
        else if ((c & 0xF8) == 0xF0 && i+3 < tok.size()) { cp = (c & 0x07) << 18 | ((unsigned char)tok[i+1] & 0x3F) << 12 | ((unsigned char)tok[i+2] & 0x3F) << 6 | ((unsigned char)tok[i+3] & 0x3F); bytes = 4; }
        i += bytes;

        if (cp < 65536 && mapped[cp]) {
            out += (char)rev[cp];
        } else {
            // Not a GPT-2 byte token — emit original UTF-8
            for (int j = 0; j < bytes; ++j) out += tok[i - bytes + j];
        }
    }
    return out;
}

std::string Tokenizer::clean_token(const std::string& tok) const {
    if (tok.empty()) return tok;

    // ── SentencePiece: ▁ (U+2581, UTF-8: E2 96 81) → space ──────────────────
    const std::string sp_marker = "\xe2\x96\x81";
    if (tok.find(sp_marker) != std::string::npos) {
        std::string result;
        result.reserve(tok.size());
        size_t pos = 0;
        while (pos < tok.size()) {
            if (tok.size() - pos >= 3 &&
                (unsigned char)tok[pos]   == 0xE2 &&
                (unsigned char)tok[pos+1] == 0x96 &&
                (unsigned char)tok[pos+2] == 0x81) {
                result += ' ';
                pos += 3;
            } else {
                result += tok[pos++];
            }
        }
        return result;
    }

    // ── SentencePiece byte tokens: <0xXX> → actual byte ──────────────────────
    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
        unsigned val = 0;
        if (sscanf(tok.c_str(), "<0x%02X>", &val) == 1)
            return std::string(1, (char)(uint8_t)val);
    }
    // Handle 5-char variant <0xX> (single hex digit)
    if (tok.size() == 5 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[4] == '>') {
        unsigned val = 0;
        if (sscanf(tok.c_str(), "<0x%X>", &val) == 1)
            return std::string(1, (char)(uint8_t)val);
    }

    // ── GPT-2/tiktoken: Ġ (U+0120) space, Ċ (U+010A) newline ────────────────
    // GPT-2 re-encodes bytes 0x00-0xFF as Unicode codepoints in range U+0021–U+00FF
    // plus U+0100–U+013F for the "extra" bytes. In UTF-8, U+0100+ starts at byte 0xC4.
    // ONLY apply this for non-SentencePiece tokenizers — SentencePiece models produce
    // real UTF-8 tokens that would be corrupted by the GPT-2 reverse mapping.
    if (!is_sentencepiece_) {
        bool has_gpt2 = false;
        for (unsigned char c : tok) {
            if (c >= 0xC4) { has_gpt2 = true; break; }
        }
        if (has_gpt2) {
            return gpt2_byte_decode(tok);
        }
    }

    return tok;
}

bool Tokenizer::is_special(int32_t id) const {
    if (id < 0) return false;
    if (id < (int32_t)vocab_.types.size()) {
        TokenType t = vocab_.types[id];
        if (t == TokenType::CONTROL ||
            t == TokenType::USER_DEFINED ||
            t == TokenType::UNKNOWN) {
            return true;
        }
    }
    if (id < (int32_t)vocab_.tokens.size()) {
        return nx_token_looks_special(vocab_.tokens[id]);
    }
    return false;
}

int32_t Tokenizer::token_to_id(const std::string& s) const {
    auto it = vocab_.token_to_id.find(s);
    return it != vocab_.token_to_id.end() ? it->second : vocab_.unk_id;
}

void Tokenizer::print_info() const {
    fprintf(stderr, "\n── Tokenizer ─────────────────────────────────\n");
    fprintf(stderr, "  Model:  %s\n",  vocab_.model_type.c_str());
    fprintf(stderr, "  Vocab:  %zu tokens\n", vocab_.tokens.size());
    fprintf(stderr, "  Merges: %zu rules\n",  merges_.size());
    fprintf(stderr, "  BOS:    %d (%s)\n",
            vocab_.bos_id,
            vocab_.bos_id < (int)vocab_.tokens.size()
                ? vocab_.tokens[vocab_.bos_id].c_str() : "?");
    fprintf(stderr, "  EOS:    %d (%s)\n",
            vocab_.eos_id,
            vocab_.eos_id < (int)vocab_.tokens.size()
                ? vocab_.tokens[vocab_.eos_id].c_str() : "?");
    fprintf(stderr, "──────────────────────────────────────────────\n\n");
}

} // namespace model
} // namespace nexus
