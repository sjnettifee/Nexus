/*
 * nexus/src/model/qwen_pretok.h
 *
 * Qwen2/2.5/3/3.5 byte-level BPE pre-tokenizer (the HF "Split" regex) plus
 * per-piece BPE. Header-only, no dependencies, so it can be unit-tested
 * against the reference HF tokenizer.
 *
 * Regex implemented (ordered alternation, exactly as in tokenizer.json):
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
 *   | \p{N}
 *   |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
 *   | \s*[\r\n]+
 *   | \s+(?!\S)
 *   | \s+
 *
 * Unicode classes: ASCII is exact. Non-ASCII uses range tables that cover
 * Latin/Greek/Cyrillic/Hebrew/Arabic/Indic/CJK/Hangul/etc. letters, common
 * marks, numbers, spaces and symbol/punctuation blocks (an approximation of
 * \p{L}/\p{M}/\p{N}; no ICU dependency).
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

namespace nexus { namespace model { namespace qwen_pretok {

inline uint32_t decode_cp(const std::string& s, size_t i, size_t& len) {
    const unsigned char c = (unsigned char)s[i];
    const size_t n = s.size();
    if (c < 0x80) { len = 1; return c; }
    if ((c & 0xE0) == 0xC0 && i + 1 < n) {
        len = 2; return ((c & 0x1Fu) << 6) | ((unsigned char)s[i+1] & 0x3Fu);
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < n) {
        len = 3; return ((c & 0x0Fu) << 12) | (((unsigned char)s[i+1] & 0x3Fu) << 6)
                      | ((unsigned char)s[i+2] & 0x3Fu);
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < n) {
        len = 4; return ((c & 0x07u) << 18) | (((unsigned char)s[i+1] & 0x3Fu) << 12)
                      | (((unsigned char)s[i+2] & 0x3Fu) << 6) | ((unsigned char)s[i+3] & 0x3Fu);
    }
    len = 1; return 0xFFFD;   // invalid byte: treated as a lone non-letter symbol
}

inline bool is_space(uint32_t c) {
    if (c == ' ' || (c >= 9 && c <= 13)) return true;
    if (c < 0x80) return false;
    return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
inline bool is_number(uint32_t c) {
    if (c < 0x80) return c >= '0' && c <= '9';
    return c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) ||
           (c >= 0x966 && c <= 0x96F) || (c >= 0x9E6 && c <= 0x9EF) ||
           (c >= 0xE50 && c <= 0xE59) || (c >= 0x2070 && c <= 0x2079 && c != 0x2071 && c != 0x2072 && c != 0x2073) ||
           (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x218B) ||
           (c >= 0x2460 && c <= 0x249B) || (c >= 0x24EA && c <= 0x24FF) ||
           (c >= 0x2776 && c <= 0x2793) || (c >= 0x3021 && c <= 0x3029) ||
           (c >= 0xFF10 && c <= 0xFF19);
}
inline bool is_mark(uint32_t c) {
    if (c < 0x300) return false;
    return (c <= 0x36F) || (c >= 0x483 && c <= 0x489) || (c >= 0x591 && c <= 0x5BD) ||
           c == 0x5BF || c == 0x5C1 || c == 0x5C2 || c == 0x5C4 || c == 0x5C5 || c == 0x5C7 ||
           (c >= 0x610 && c <= 0x61A) || (c >= 0x64B && c <= 0x65F) || c == 0x670 ||
           (c >= 0x6D6 && c <= 0x6DC) || (c >= 0x6DF && c <= 0x6E4) || c == 0x6E7 || c == 0x6E8 ||
           (c >= 0x6EA && c <= 0x6ED) || (c >= 0x900 && c <= 0x903) || (c >= 0x93A && c <= 0x94F && c != 0x93D) ||
           (c >= 0x951 && c <= 0x957) || (c >= 0x962 && c <= 0x963) ||
           (c >= 0xE31 && c <= 0xE3A && c != 0xE32 && c != 0xE33) || (c >= 0xE47 && c <= 0xE4E) ||
           (c >= 0x1AB0 && c <= 0x1AFF) || (c >= 0x1DC0 && c <= 0x1DFF) ||
           (c >= 0x20D0 && c <= 0x20FF) || (c >= 0x302A && c <= 0x302F) ||
           c == 0x3099 || c == 0x309A || (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFE20 && c <= 0xFE2F);
}
inline bool is_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    if (c < 0x100) {
        if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
        if (c < 0xC0) return false;               // Latin-1 punctuation / symbols / fractions
        return c != 0xD7 && c != 0xF7;
    }
    if (is_mark(c) || is_number(c) || is_space(c)) return false;
    if (c <= 0x2FF) return true;                  // Latin ext, IPA, modifier letters
    if (c <= 0x3FF) return !(c == 0x374 || c == 0x375 || c == 0x37E || c == 0x384 || c == 0x385 || c == 0x387);
    if (c <= 0x52F) return c != 0x482;            // Greek/Cyrillic
    if (c <= 0x58F) return (c >= 0x531 && c <= 0x556) || (c >= 0x561 && c <= 0x587);
    if (c <= 0x5FF) return c >= 0x5D0 && c <= 0x5F2;
    if (c <= 0x6FF) return (c >= 0x620 && c <= 0x64A) || (c >= 0x66E && c <= 0x6D3) || c == 0x6D5 || c >= 0x6FA;
    if (c <= 0x7FF) return true;                  // Syriac/Thaana/NKo
    if (c <= 0xDFF) return true;                  // Indic scripts (marks removed above)
    if (c <= 0xE7F) return !(c == 0xE3F || c == 0xE4F || c == 0xE5A || c == 0xE5B);   // Thai/Lao
    if (c <= 0x1FFF) return true;                 // Lao..Greek ext, Georgian, Hangul Jamo, etc.
    if (c <= 0x2BFF) return false;                // punctuation, symbols, arrows, math, boxes
    if (c <= 0x2DFF) return true;                 // Glagolitic, Coptic, Georgian sup, Ethiopic ext
    if (c <= 0x303F) return c == 0x3005 || c == 0x3006 || c == 0x3031 || c == 0x3032 ||
                            c == 0x3033 || c == 0x3034 || c == 0x3035 || c == 0x303B || c == 0x303C;
    if (c <= 0x30FF) return (c >= 0x3041 && c <= 0x3096) || (c >= 0x309D && c <= 0x309F) ||
                            (c >= 0x30A1 && c <= 0x30FA) || (c >= 0x30FC && c <= 0x30FF);
    if (c <= 0x33FF) return (c >= 0x3105 && c <= 0x312F) || (c >= 0x3131 && c <= 0x318E) || (c >= 0x31A0 && c <= 0x31BF) || (c >= 0x31F0 && c <= 0x31FF);
    if (c <= 0x4DBF) return c >= 0x3400;         // CJK ext A
    if (c <= 0x4DFF) return false;                // hexagram symbols
    if (c <= 0xA48F) return true;                 // CJK unified + Yi
    if (c <= 0xABFF) return c >= 0xA500 && !(c >= 0xA6F0 && c <= 0xA6F7) ;
    if (c <= 0xD7AF) return c >= 0xAC00;         // Hangul syllables
    if (c <= 0xF8FF) return false;                // surrogates / private use
    if (c <= 0xFAFF) return true;                 // CJK compat ideographs
    if (c <= 0xFDFF) return (c >= 0xFB1D && c <= 0xFDCF);
    if (c <= 0xFE6F) return false;
    if (c <= 0xFFEF) return (c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A) || (c >= 0xFF66 && c <= 0xFFDC);
    if (c >= 0x10000 && c <= 0x1F000) return !(c >= 0x1D000 && c <= 0x1D3FF);  // SMP scripts (approx.)
    if (c >= 0x20000 && c <= 0x323AF) return true;   // CJK ext B..H
    return false;                                  // emoji / symbols / tags / private
}

inline bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }

// Split `s` into Qwen pre-tokens. Returns (byte offset, byte length) pairs.
inline std::vector<std::pair<size_t, size_t>> split(const std::string& s) {
    std::vector<std::pair<size_t, size_t>> out;
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        size_t l0 = 0;
        const uint32_t c0 = decode_cp(s, i, l0);
        size_t end = 0;
        bool ok = false;

        // 1. contractions (case-insensitive)
        if (c0 == '\'' && i + 1 < n) {
            auto lc = [&](size_t k) { char ch = s[k]; return (char)((ch >= 'A' && ch <= 'Z') ? ch + 32 : ch); };
            const char a = lc(i + 1);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') { end = i + 2; ok = true; }
            else if (i + 2 < n) {
                const char b = lc(i + 2);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { end = i + 3; ok = true; }
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!ok) {
            if (!is_crlf(c0) && !is_letter(c0) && !is_number(c0)) {
                size_t q = i + l0;
                const size_t p = q;
                while (q < n) { size_t l; uint32_t c = decode_cp(s, q, l);
                                if (is_letter(c) || is_mark(c)) q += l; else break; }
                if (q > p) { end = q; ok = true; }
            }
            if (!ok) {
                size_t q = i;
                while (q < n) { size_t l; uint32_t c = decode_cp(s, q, l);
                                if (is_letter(c) || is_mark(c)) q += l; else break; }
                if (q > i) { end = q; ok = true; }
            }
        }

        // 3. single digit
        if (!ok && is_number(c0)) { end = i + l0; ok = true; }

        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (!ok) {
            size_t q = i;
            if (c0 == ' ') q += 1;
            const size_t p = q;
            while (q < n) { size_t l; uint32_t c = decode_cp(s, q, l);
                            if (!is_space(c) && !is_letter(c) && !is_mark(c) && !is_number(c)) q += l; else break; }
            if (q > p) {
                while (q < n && (s[q] == '\r' || s[q] == '\n')) ++q;
                end = q; ok = true;
            }
        }

        // 5/6/7: whitespace handling
        if (!ok && is_space(c0)) {
            size_t e = i;                    // end of maximal whitespace run
            size_t last_nl_end = 0;          // end offset of last \r|\n in the run
            while (e < n) { size_t l; uint32_t c = decode_cp(s, e, l);
                            if (!is_space(c)) break;
                            e += l;
                            if (is_crlf(c)) last_nl_end = e; }
            if (last_nl_end > 0) { end = last_nl_end; ok = true; }                 // \s*[\r\n]+
            else if (e >= n)     { end = e; ok = true; }                           // \s+(?!\S) at EOF
            else {
                // \s+(?!\S): leave the final whitespace char for the next token
                size_t prev = e - 1;
                while (prev > i && ((unsigned char)s[prev] & 0xC0) == 0x80) --prev;
                if (prev > i) { end = prev; ok = true; }
                else          { end = e;    ok = true; }                           // \s+
            }
        }

        if (!ok) end = i + l0;               // unreachable safety net
        out.emplace_back(i, end - i);
        i = end;
    }
    return out;
}

// Byte-level BPE of one pre-token. `piece` is raw UTF-8 bytes of the pre-token.
// MergeMap: unordered_map<uint64_t, M> where M has .score (rank, lower=first) and .result.
template <class MergeMap>
inline void bpe_piece(const std::string& piece,
                      const std::vector<std::string>& byte_char,       // 256 entries: byte -> vocab string
                      const std::unordered_map<std::string, int32_t>& token_to_id,
                      const MergeMap& merge_map, int32_t unk_id,
                      std::vector<int32_t>& out)
{
    std::vector<int32_t> t;
    t.reserve(piece.size());
    for (unsigned char b : piece) {
        auto it = token_to_id.find(byte_char[b]);
        t.push_back(it != token_to_id.end() ? it->second : unk_id);
    }
    while (t.size() >= 2) {
        float best = 1e38f; size_t bp = (size_t)-1; int32_t br = -1;
        for (size_t i = 0; i + 1 < t.size(); ++i) {
            const uint64_t key = ((uint64_t)(uint32_t)t[i] << 32) | (uint32_t)t[i+1];
            auto it = merge_map.find(key);
            if (it == merge_map.end()) continue;
            if (it->second.score < best) { best = it->second.score; bp = i; br = it->second.result; }
        }
        if (bp == (size_t)-1) break;
        t[bp] = br;
        t.erase(t.begin() + bp + 1);
    }
    out.insert(out.end(), t.begin(), t.end());
}

}}} // namespace
