// sw_render.h -- draws Dear ImGui draw lists into a plain 32-bit pixel buffer (0x00RRGGBB) on the CPU.
// No GPU, no OpenGL, no driver: this is what lets the launcher run on any machine that can show a window.
#pragma once
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace swr {

struct Tex { const uint32_t* px = nullptr; int w = 0, h = 0; };   // ImGui RGBA32 atlas (ABGR in memory order R,G,B,A)

static inline void blend_px(uint32_t& d, int r, int g, int b, int a) {
    if (a >= 255) { d = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b; return; }
    if (a <= 0) return;
    int dr = (d >> 16) & 255, dg = (d >> 8) & 255, db = d & 255;
    dr += ((r - dr) * a + 127) / 255; dg += ((g - dg) * a + 127) / 255; db += ((b - db) * a + 127) / 255;
    d = ((uint32_t)dr << 16) | ((uint32_t)dg << 8) | (uint32_t)db;
}

struct Fb { uint32_t* px; int w, h; };

static inline uint32_t tex_at(const Tex& t, float u, float v) {
    int x = (int)(u * t.w), y = (int)(v * t.h);
    x = x < 0 ? 0 : (x >= t.w ? t.w - 1 : x); y = y < 0 ? 0 : (y >= t.h ? t.h - 1 : y);
    return t.px[(size_t)y * t.w + x];
}

// axis-aligned rectangle, one colour, optional texture gradient in uv
static void fill_rect(Fb& fb, const ImVec4& clip, float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1,
                      uint32_t col, const Tex& tex, bool textured) {
    int ix0 = (int)std::ceil(std::max(x0, clip.x) - 0.5f), iy0 = (int)std::ceil(std::max(y0, clip.y) - 0.5f);
    int ix1 = (int)std::ceil(std::min(x1, clip.z) - 0.5f), iy1 = (int)std::ceil(std::min(y1, clip.w) - 0.5f);
    ix0 = std::max(ix0, 0); iy0 = std::max(iy0, 0); ix1 = std::min(ix1, fb.w); iy1 = std::min(iy1, fb.h);
    if (ix0 >= ix1 || iy0 >= iy1) return;
    const int cr = col & 255, cg = (col >> 8) & 255, cb = (col >> 16) & 255, ca = (col >> 24) & 255;
    const float iw = 1.0f / (x1 - x0), ih = 1.0f / (y1 - y0);
    for (int y = iy0; y < iy1; ++y) {
        uint32_t* row = fb.px + (size_t)y * fb.w;
        const float v = v0 + ((y + 0.5f - y0) * ih) * (v1 - v0);
        for (int x = ix0; x < ix1; ++x) {
            if (!textured) { blend_px(row[x], cr, cg, cb, ca); continue; }
            const float u = u0 + ((x + 0.5f - x0) * iw) * (u1 - u0);
            uint32_t t = tex_at(tex, u, v);
            int a = ((int)((t >> 24) & 255) * ca + 127) / 255;
            if (a == 0) continue;
            blend_px(row[x], (cr * (int)(t & 255) + 127) / 255, (cg * (int)((t >> 8) & 255) + 127) / 255, (cb * (int)((t >> 16) & 255) + 127) / 255, a);
        }
    }
}

// general triangle (vertex colours + uv interpolated)
static void fill_tri(Fb& fb, const ImVec4& clip, const ImDrawVert& A, const ImDrawVert& B, const ImDrawVert& C, const Tex& tex) {
    float area = (B.pos.x - A.pos.x) * (C.pos.y - A.pos.y) - (B.pos.y - A.pos.y) * (C.pos.x - A.pos.x);
    if (area == 0.0f) return;
    const ImDrawVert *a = &A, *b = &B, *c = &C;
    if (area < 0) { std::swap(b, c); area = -area; }
    float minx = std::min({a->pos.x, b->pos.x, c->pos.x}), maxx = std::max({a->pos.x, b->pos.x, c->pos.x});
    float miny = std::min({a->pos.y, b->pos.y, c->pos.y}), maxy = std::max({a->pos.y, b->pos.y, c->pos.y});
    int x0 = std::max((int)std::floor(std::max(minx, clip.x)), 0), x1 = std::min((int)std::ceil(std::min(maxx, clip.z)), fb.w);
    int y0 = std::max((int)std::floor(std::max(miny, clip.y)), 0), y1 = std::min((int)std::ceil(std::min(maxy, clip.w)), fb.h);
    if (x0 >= x1 || y0 >= y1) return;
    const float inv = 1.0f / area;
    auto ch = [](uint32_t col, int s) { return (float)((col >> s) & 255); };
    for (int y = y0; y < y1; ++y) {
        uint32_t* row = fb.px + (size_t)y * fb.w;
        const float py = y + 0.5f;
        for (int x = x0; x < x1; ++x) {
            const float px = x + 0.5f;
            float w0 = ((b->pos.x - px) * (c->pos.y - py) - (b->pos.y - py) * (c->pos.x - px));   // weight of a
            float w1 = ((c->pos.x - px) * (a->pos.y - py) - (c->pos.y - py) * (a->pos.x - px));   // weight of b
            float w2 = ((a->pos.x - px) * (b->pos.y - py) - (a->pos.y - py) * (b->pos.x - px));   // weight of c
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            w0 *= inv; w1 *= inv; w2 *= inv;
            float r = w0 * ch(a->col, 0) + w1 * ch(b->col, 0) + w2 * ch(c->col, 0);
            float g = w0 * ch(a->col, 8) + w1 * ch(b->col, 8) + w2 * ch(c->col, 8);
            float bl = w0 * ch(a->col, 16) + w1 * ch(b->col, 16) + w2 * ch(c->col, 16);
            float al = w0 * ch(a->col, 24) + w1 * ch(b->col, 24) + w2 * ch(c->col, 24);
            int ir = (int)(r + 0.5f), ig = (int)(g + 0.5f), ib = (int)(bl + 0.5f), ia = (int)(al + 0.5f);
            if (tex.px) {
                float u = w0 * a->uv.x + w1 * b->uv.x + w2 * c->uv.x, v = w0 * a->uv.y + w1 * b->uv.y + w2 * c->uv.y;
                uint32_t t = tex_at(tex, u, v);
                ir = (ir * (int)(t & 255) + 127) / 255; ig = (ig * (int)((t >> 8) & 255) + 127) / 255;
                ib = (ib * (int)((t >> 16) & 255) + 127) / 255; ia = (ia * (int)((t >> 24) & 255) + 127) / 255;
            }
            blend_px(row[x], ir, ig, ib, ia);
        }
    }
}

static inline bool feq(float a, float b) { return std::fabs(a - b) < 0.001f; }

// Render everything.  `clear` is the background colour (0x00RRGGBB).
static void render(const ImDrawData* dd, uint32_t* pixels, int w, int h, uint32_t clear, const Tex& font) {
    Fb fb{pixels, w, h};
    for (size_t i = 0, n = (size_t)w * h; i < n; ++i) pixels[i] = clear;
    for (int n = 0; n < dd->CmdListsCount; ++n) {
        const ImDrawList* dl = dd->CmdLists[n];
        const ImDrawVert* vb = dl->VtxBuffer.Data;
        const ImDrawIdx* ib = dl->IdxBuffer.Data;
        for (int ci = 0; ci < dl->CmdBuffer.Size; ++ci) {
            const ImDrawCmd& cmd = dl->CmdBuffer[ci];
            if (cmd.UserCallback) continue;
            ImVec4 clip(cmd.ClipRect.x, cmd.ClipRect.y, cmd.ClipRect.z, cmd.ClipRect.w);
            const ImDrawIdx* idx = ib + cmd.IdxOffset;
            const ImDrawVert* vtx = vb + cmd.VtxOffset;
            const Tex* tex = &font;                                       // the only texture we ever use
            for (unsigned i = 0; i + 3 <= cmd.ElemCount;) {
                // fast path: two triangles that form an axis-aligned rectangle of one colour (panels, glyphs)
                if (i + 6 <= cmd.ElemCount && idx[i] == idx[i + 3] && idx[i + 2] == idx[i + 4]) {
                    const ImDrawVert &p = vtx[idx[i]], &q = vtx[idx[i + 1]], &r = vtx[idx[i + 2]], &s = vtx[idx[i + 5]];
                    if (feq(q.pos.y, p.pos.y) && feq(q.pos.x, r.pos.x) && feq(s.pos.x, p.pos.x) && feq(s.pos.y, r.pos.y) &&
                        p.col == q.col && q.col == r.col && r.col == s.col && p.pos.x < r.pos.x && p.pos.y < r.pos.y) {
                        const bool same_uv = feq(p.uv.x, r.uv.x) && feq(p.uv.y, r.uv.y);
                        fill_rect(fb, clip, p.pos.x, p.pos.y, r.pos.x, r.pos.y, p.uv.x, p.uv.y, r.uv.x, r.uv.y, p.col, *tex, true);
                        (void)same_uv;
                        i += 6; continue;
                    }
                }
                fill_tri(fb, clip, vtx[idx[i]], vtx[idx[i + 1]], vtx[idx[i + 2]], *tex);
                i += 3;
            }
        }
    }
}

}  // namespace swr
