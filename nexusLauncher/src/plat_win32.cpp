// Windows window: plain Win32 + GDI blit.  No DirectX / OpenGL, so it works on any machine and in VMs.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "platform.h"
#include "imgui.h"
#include "icon.h"
#include <windows.h>
#include <windowsx.h>
#include <string>
#include <vector>

namespace {
HWND g_hwnd = nullptr; int g_w = 0, g_h = 0; float g_scale = 1.0f; bool g_closed = false; bool g_got = false;
std::vector<uint32_t> g_last; int g_lw = 0, g_lh = 0;
wchar_t g_hi_surrogate = 0;

ImGuiKey map_vk(WPARAM vk, LPARAM lp) {
    const bool ext = (lp & (1 << 24)) != 0;
    if (vk >= 'A' && vk <= 'Z') return (ImGuiKey)(ImGuiKey_A + (vk - 'A'));
    if (vk >= '0' && vk <= '9') return (ImGuiKey)(ImGuiKey_0 + (vk - '0'));
    if (vk >= VK_F1 && vk <= VK_F12) return (ImGuiKey)(ImGuiKey_F1 + (vk - VK_F1));
    switch (vk) {
        case VK_TAB: return ImGuiKey_Tab; case VK_LEFT: return ImGuiKey_LeftArrow; case VK_RIGHT: return ImGuiKey_RightArrow;
        case VK_UP: return ImGuiKey_UpArrow; case VK_DOWN: return ImGuiKey_DownArrow; case VK_PRIOR: return ImGuiKey_PageUp;
        case VK_NEXT: return ImGuiKey_PageDown; case VK_HOME: return ImGuiKey_Home; case VK_END: return ImGuiKey_End;
        case VK_INSERT: return ImGuiKey_Insert; case VK_DELETE: return ImGuiKey_Delete; case VK_BACK: return ImGuiKey_Backspace;
        case VK_SPACE: return ImGuiKey_Space; case VK_RETURN: return ext ? ImGuiKey_KeypadEnter : ImGuiKey_Enter;
        case VK_ESCAPE: return ImGuiKey_Escape; case VK_OEM_MINUS: return ImGuiKey_Minus; case VK_OEM_PERIOD: return ImGuiKey_Period;
        case VK_OEM_COMMA: return ImGuiKey_Comma; case VK_OEM_2: return ImGuiKey_Slash; case VK_OEM_5: return ImGuiKey_Backslash;
        case VK_OEM_1: return ImGuiKey_Semicolon; case VK_OEM_PLUS: return ImGuiKey_Equal;
        default: return ImGuiKey_None;
    }
}
void mods() {
    ImGuiIO& io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiMod_Ctrl, (GetKeyState(VK_CONTROL) & 0x8000) != 0); io.AddKeyEvent(ImGuiMod_Shift, (GetKeyState(VK_SHIFT) & 0x8000) != 0);
    io.AddKeyEvent(ImGuiMod_Alt, (GetKeyState(VK_MENU) & 0x8000) != 0);
}
void blit(HDC dc, const uint32_t* px, int w, int h) {
    BITMAPINFO bi{}; bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER); bi.bmiHeader.biWidth = w; bi.bmiHeader.biHeight = -h;
    bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
    SetDIBitsToDevice(dc, 0, 0, (DWORD)w, (DWORD)h, 0, 0, 0, (UINT)h, px, &bi, DIB_RGB_COLORS);
}

LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    ImGuiIO* io = ImGui::GetCurrentContext() ? &ImGui::GetIO() : nullptr;
    switch (m) {
        case WM_CLOSE: g_closed = true; g_got = true; return 0;
        case WM_DESTROY: return 0;
        case WM_ERASEBKGND: return 1;
        case WM_SIZE: if (wp != SIZE_MINIMIZED) { g_w = LOWORD(lp); g_h = HIWORD(lp); g_got = true; } return 0;
        case WM_GETMINMAXINFO: { MINMAXINFO* mi = (MINMAXINFO*)lp; mi->ptMinTrackSize.x = 640; mi->ptMinTrackSize.y = 460; return 0; }
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); if (!g_last.empty()) blit(dc, g_last.data(), g_lw, g_lh); EndPaint(h, &ps); return 0; }
        case WM_MOUSEMOVE: if (io) { io->AddMousePosEvent((float)GET_X_LPARAM(lp), (float)GET_Y_LPARAM(lp)); } g_got = true;
            { TRACKMOUSEEVENT t{sizeof t, TME_LEAVE, h, 0}; TrackMouseEvent(&t); } return 0;
        case WM_MOUSELEAVE: if (io) io->AddMousePosEvent(-FLT_MAX, -FLT_MAX); g_got = true; return 0;
        case WM_LBUTTONDOWN: case WM_LBUTTONDBLCLK: SetCapture(h); if (io) io->AddMouseButtonEvent(0, true); g_got = true; return 0;
        case WM_LBUTTONUP: ReleaseCapture(); if (io) io->AddMouseButtonEvent(0, false); g_got = true; return 0;
        case WM_RBUTTONDOWN: case WM_RBUTTONDBLCLK: if (io) io->AddMouseButtonEvent(1, true); g_got = true; return 0;
        case WM_RBUTTONUP: if (io) io->AddMouseButtonEvent(1, false); g_got = true; return 0;
        case WM_MBUTTONDOWN: if (io) io->AddMouseButtonEvent(2, true); g_got = true; return 0;
        case WM_MBUTTONUP: if (io) io->AddMouseButtonEvent(2, false); g_got = true; return 0;
        case WM_MOUSEWHEEL: if (io) io->AddMouseWheelEvent(0, (float)GET_WHEEL_DELTA_WPARAM(wp) / WHEEL_DELTA); g_got = true; return 0;
        case WM_MOUSEHWHEEL: if (io) io->AddMouseWheelEvent((float)GET_WHEEL_DELTA_WPARAM(wp) / WHEEL_DELTA, 0); g_got = true; return 0;
        case WM_KEYDOWN: case WM_SYSKEYDOWN: case WM_KEYUP: case WM_SYSKEYUP:
            if (io) { mods(); ImGuiKey k = map_vk(wp, lp); if (k != ImGuiKey_None) io->AddKeyEvent(k, m == WM_KEYDOWN || m == WM_SYSKEYDOWN); }
            g_got = true; if (m == WM_SYSKEYDOWN || m == WM_SYSKEYUP) { if (wp == VK_F10 || wp == VK_MENU) return 0; } return 0;
        case WM_CHAR:
            if (io) {
                wchar_t c = (wchar_t)wp;
                if (c >= 0xD800 && c <= 0xDBFF) g_hi_surrogate = c;
                else { unsigned cp = c; if (c >= 0xDC00 && c <= 0xDFFF && g_hi_surrogate) { cp = 0x10000 + (((unsigned)g_hi_surrogate - 0xD800) << 10) + ((unsigned)c - 0xDC00); } g_hi_surrogate = 0;
                    if (cp >= 32 && cp != 127) io->AddInputCharacter(cp); }
            }
            g_got = true; return 0;
        case WM_SETFOCUS: if (io) io->AddFocusEvent(true); g_got = true; return 0;
        case WM_KILLFOCUS: if (io) io->AddFocusEvent(false); g_got = true; return 0;
        case WM_SETCURSOR: if (LOWORD(lp) == HTCLIENT) { SetCursor(LoadCursor(nullptr, IDC_ARROW)); return TRUE; } break;
    }
    return DefWindowProcW(h, m, wp, lp);
}
std::wstring widen(const char* s) { int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0); std::wstring w((size_t)(n > 0 ? n : 1), 0); MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n); w.resize(n > 0 ? (size_t)n - 1 : 0); return w; }
}  // namespace

namespace plat {
bool create(const char* title, int w, int h) {
    // per-monitor DPI awareness (Win10+), falling back to system-wide, via GetProcAddress so old Windows still starts
    if (HMODULE u = GetModuleHandleW(L"user32.dll")) {
        typedef BOOL(WINAPI * F1)(HANDLE);
        typedef BOOL(WINAPI * F2)();
        void* p1 = (void*)GetProcAddress(u, "SetProcessDpiAwarenessContext");
        void* p2 = (void*)GetProcAddress(u, "SetProcessDPIAware");
        if (p1) ((F1)p1)((HANDLE)-4); else if (p2) ((F2)p2)();
    }
    HDC sdc = GetDC(nullptr); int dpi = GetDeviceCaps(sdc, LOGPIXELSX); ReleaseDC(nullptr, sdc);
    g_scale = dpi > 0 ? dpi / 96.0f : 1.0f;
    if (const char* s = getenv("NEXUS_UI_SCALE")) { float v = (float)atof(s); if (v >= 0.5f && v <= 4) g_scale = v; }
    WNDCLASSEXW wc{}; wc.cbSize = sizeof wc; wc.style = CS_DBLCLKS; wc.lpfnWndProc = wndproc; wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW); wc.lpszClassName = L"NexusLauncher";
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    if (!RegisterClassExW(&wc)) return false;
    RECT r{0, 0, (LONG)(w * g_scale), (LONG)(h * g_scale)}; AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, widen(title).c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                             nullptr, nullptr, wc.hInstance, nullptr);
    if (!g_hwnd) return false;
    for (int n : {32, 16}) {                                   // window / taskbar icon drawn in code (no .rc needed)
        auto px = nx_icon_pixels(n); BITMAPV5HEADER bi{}; bi.bV5Size = sizeof bi; bi.bV5Width = n; bi.bV5Height = -n; bi.bV5Planes = 1; bi.bV5BitCount = 32;
        bi.bV5Compression = BI_BITFIELDS; bi.bV5RedMask = 0x00FF0000; bi.bV5GreenMask = 0x0000FF00; bi.bV5BlueMask = 0x000000FF; bi.bV5AlphaMask = 0xFF000000;
        void* bits = nullptr; HDC dc = GetDC(nullptr);
        HBITMAP color = CreateDIBSection(dc, (BITMAPINFO*)&bi, DIB_RGB_COLORS, &bits, nullptr, 0);
        HBITMAP mask = CreateBitmap(n, n, 1, 1, nullptr); ReleaseDC(nullptr, dc);
        if (color && bits) {
            memcpy(bits, px.data(), px.size() * 4); ICONINFO ii{}; ii.fIcon = TRUE; ii.hbmMask = mask; ii.hbmColor = color;
            if (HICON hi = CreateIconIndirect(&ii)) SendMessageW(g_hwnd, WM_SETICON, n == 32 ? ICON_BIG : ICON_SMALL, (LPARAM)hi);
        }
        if (color) DeleteObject(color); if (mask) DeleteObject(mask);
    }
    ShowWindow(g_hwnd, SW_SHOWNORMAL);
    RECT cr; GetClientRect(g_hwnd, &cr); g_w = cr.right; g_h = cr.bottom;
    return true;
}
void size(int& w, int& h) { w = g_w; h = g_h; }
float dpi_scale() { return g_scale; }
bool pump(int wait_ms, bool& got_event) {
    g_got = false;
    MSG msg;
    if (!PeekMessageW(&msg, nullptr, 0, 0, PM_NOREMOVE) && wait_ms > 0) MsgWaitForMultipleObjects(0, nullptr, FALSE, (DWORD)wait_ms, QS_ALLINPUT);
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    got_event = g_got;
    return !g_closed;
}
void present(const uint32_t* px, int w, int h) {
    g_last.assign(px, px + (size_t)w * h); g_lw = w; g_lh = h;
    HDC dc = GetDC(g_hwnd); blit(dc, px, w, h); ReleaseDC(g_hwnd, dc);
}
void set_clipboard(const char* t) {
    if (!OpenClipboard(g_hwnd)) return;
    EmptyClipboard();
    std::wstring w = widen(t ? t : "");
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, (w.size() + 1) * sizeof(wchar_t));
    if (g) { memcpy(GlobalLock(g), w.c_str(), (w.size() + 1) * sizeof(wchar_t)); GlobalUnlock(g); SetClipboardData(CF_UNICODETEXT, g); }
    CloseClipboard();
}
std::string get_clipboard() {
    std::string out;
    if (!OpenClipboard(g_hwnd)) return out;
    if (HANDLE h = GetClipboardData(CF_UNICODETEXT)) {
        if (const wchar_t* w = (const wchar_t*)GlobalLock(h)) {
            int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
            if (n > 1) { out.resize((size_t)n - 1); WideCharToMultiByte(CP_UTF8, 0, w, -1, &out[0], n, nullptr, nullptr); }
            GlobalUnlock(h);
        }
    }
    CloseClipboard();
    return out;
}
void cancel_close() { g_closed = false; }
void destroy() { if (g_hwnd) DestroyWindow(g_hwnd); g_hwnd = nullptr; }
}  // namespace plat
