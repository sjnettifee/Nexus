// icon.h -- the Nexus window icon, drawn in code (no image file, no resource compiler, nothing to install).
// A violet diamond outline with a solid diamond inside, on a dark rounded square (same mark as the launcher header).
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

// n x n pixels, 0xAARRGGBB (straight alpha), 4x4 supersampled.
static std::vector<uint32_t> nx_icon_pixels(int n) {
    std::vector<uint32_t> out((size_t)n * n, 0);
    const float c = n * 0.5f, R = n * 0.5f;
    const float corner = n * 0.22f;
    auto inside_rr = [&](float x, float y) {                       // rounded square covering the whole icon
        float dx = std::fabs(x - c) - (R - corner), dy = std::fabs(y - c) - (R - corner);
        if (dx <= 0 || dy <= 0) return std::fabs(x - c) <= R && std::fabs(y - c) <= R;
        return dx * dx + dy * dy <= corner * corner;
    };
    for (int py = 0; py < n; ++py) for (int px = 0; px < n; ++px) {
        int a = 0, rs = 0, gs = 0, bs = 0, cnt = 0;
        for (int sy = 0; sy < 4; ++sy) for (int sx = 0; sx < 4; ++sx) {
            float x = px + (sx + 0.5f) / 4, y = py + (sy + 0.5f) / 4; ++cnt;
            if (!inside_rr(x, y)) continue;
            float d = (std::fabs(x - c) + std::fabs(y - c)) / R;      // diamond "radius": 1.0 touches the edge midpoints
            int r = 0x16, g = 0x1b, b = 0x26;                          // background
            bool ink = (d >= 0.62f && d <= 0.76f) || d <= 0.30f;       // outer ring + inner solid diamond
            if (ink) { r = 0x7c; g = 0x6a; b = 0xf7; }
            a += 255; rs += r; gs += g; bs += b;
        }
        if (a == 0) continue;
        int cov = a / cnt; int m = a / 255;                            // m = covered samples
        out[(size_t)py * n + px] = ((uint32_t)cov << 24) | ((uint32_t)(rs / m) << 16) | ((uint32_t)(gs / m) << 8) | (uint32_t)(bs / m);
    }
    return out;
}
