// Linux / BSD window: plain Xlib, loaded with dlopen so the program starts (and prints a clear message)
// on machines without X11 instead of failing to launch.  Build time only needs the X11 headers.
#include "platform.h"
#include "imgui.h"
#include "icon.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <dlfcn.h>
#include <poll.h>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

namespace {
void* g_lib = nullptr;
#define X_FUNCS(X) \
  X(Display*, XOpenDisplay, (const char*)) X(int, XCloseDisplay, (Display*)) \
  X(Window, XCreateSimpleWindow, (Display*, Window, int, int, unsigned, unsigned, unsigned, unsigned long, unsigned long)) \
  X(int, XStoreName, (Display*, Window, const char*)) X(int, XSelectInput, (Display*, Window, long)) \
  X(int, XMapWindow, (Display*, Window)) X(Atom, XInternAtom, (Display*, const char*, Bool)) \
  X(Status, XSetWMProtocols, (Display*, Window, Atom*, int)) X(int, XPending, (Display*)) \
  X(int, XNextEvent, (Display*, XEvent*)) X(int, XConnectionNumber, (Display*)) \
  X(GC, XCreateGC, (Display*, Drawable, unsigned long, XGCValues*)) X(int, XFreeGC, (Display*, GC)) \
  X(XImage*, XCreateImage, (Display*, Visual*, unsigned, int, int, char*, unsigned, unsigned, int, int)) \
  X(int, XPutImage, (Display*, Drawable, GC, XImage*, int, int, int, int, unsigned, unsigned)) \
  X(int, XFlush, (Display*)) X(int, XLookupString, (XKeyEvent*, char*, int, KeySym*, XComposeStatus*)) \
  X(char*, XResourceManagerString, (Display*)) X(int, XDestroyWindow, (Display*, Window)) \
  X(int, XChangeProperty, (Display*, Window, Atom, Atom, int, int, const unsigned char*, int)) \
  X(int, XSetSelectionOwner, (Display*, Atom, Window, Time)) X(Window, XGetSelectionOwner, (Display*, Atom)) \
  X(int, XConvertSelection, (Display*, Atom, Atom, Atom, Window, Time)) \
  X(int, XGetWindowProperty, (Display*, Window, Atom, long, long, Bool, Atom, Atom*, int*, unsigned long*, unsigned long*, unsigned char**)) \
  X(int, XFree, (void*)) X(int, XSendEvent, (Display*, Window, Bool, long, XEvent*)) \
  X(int, XDeleteProperty, (Display*, Window, Atom)) X(Bool, XFilterEvent, (XEvent*, Window)) \
  X(char*, XSetLocaleModifiers, (const char*)) X(XIM, XOpenIM, (Display*, struct _XrmHashBucketRec*, char*, char*)) \
  X(int, Xutf8LookupString, (XIC, XKeyPressedEvent*, char*, int, KeySym*, Status*)) \
  X(void, XSetWMNormalHints, (Display*, Window, XSizeHints*)) X(int, XCloseIM, (XIM)) X(void, XDestroyIC, (XIC)) \
  X(XSizeHints*, XAllocSizeHints, (void))
#define X_DECL(R, N, A) R(*p_##N) A = nullptr;
X_FUNCS(X_DECL)
XIC (*p_XCreateIC)(XIM, ...) = nullptr;                      // variadic: declared separately

Display* g_dpy = nullptr; Window g_win = 0; GC g_gc = 0; Visual* g_vis = nullptr; int g_depth = 24;
Atom a_wm_delete, a_clip, a_utf8, a_targets, a_prop, a_net_name;
XIM g_im = nullptr; XIC g_ic = nullptr;
int g_w = 0, g_h = 0; float g_scale = 1.0f; bool g_closed = false;
std::string g_clip_own;                                         // text we offer when we own CLIPBOARD
std::string g_last_clip;

bool load_x() {
    const char* names[] = {"libX11.so.6", "libX11.so", nullptr};
    for (int i = 0; names[i] && !g_lib; ++i) g_lib = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
    if (!g_lib) return false;
#define X_LOAD(R, N, A) p_##N = (R(*) A)dlsym(g_lib, #N);
    X_FUNCS(X_LOAD)
    p_XCreateIC = (XIC(*)(XIM, ...))dlsym(g_lib, "XCreateIC");
    return p_XOpenDisplay && p_XCreateSimpleWindow && p_XPutImage && p_XNextEvent && p_XCreateImage && p_XLookupString;
}

ImGuiKey map_key(KeySym ks) {
    if (ks >= XK_a && ks <= XK_z) return (ImGuiKey)(ImGuiKey_A + (ks - XK_a));
    if (ks >= XK_A && ks <= XK_Z) return (ImGuiKey)(ImGuiKey_A + (ks - XK_A));
    if (ks >= XK_0 && ks <= XK_9) return (ImGuiKey)(ImGuiKey_0 + (ks - XK_0));
    if (ks >= XK_F1 && ks <= XK_F12) return (ImGuiKey)(ImGuiKey_F1 + (ks - XK_F1));
    switch (ks) {
        case XK_Tab: return ImGuiKey_Tab; case XK_Left: return ImGuiKey_LeftArrow; case XK_Right: return ImGuiKey_RightArrow;
        case XK_Up: return ImGuiKey_UpArrow; case XK_Down: return ImGuiKey_DownArrow; case XK_Prior: return ImGuiKey_PageUp;
        case XK_Next: return ImGuiKey_PageDown; case XK_Home: return ImGuiKey_Home; case XK_End: return ImGuiKey_End;
        case XK_Insert: return ImGuiKey_Insert; case XK_Delete: return ImGuiKey_Delete; case XK_BackSpace: return ImGuiKey_Backspace;
        case XK_space: return ImGuiKey_Space; case XK_Return: return ImGuiKey_Enter; case XK_KP_Enter: return ImGuiKey_KeypadEnter;
        case XK_Escape: return ImGuiKey_Escape; case XK_minus: return ImGuiKey_Minus; case XK_period: return ImGuiKey_Period;
        case XK_comma: return ImGuiKey_Comma; case XK_slash: return ImGuiKey_Slash; case XK_backslash: return ImGuiKey_Backslash;
        case XK_semicolon: return ImGuiKey_Semicolon; case XK_equal: return ImGuiKey_Equal;
        default: return ImGuiKey_None;
    }
}

void mods(unsigned state) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiMod_Ctrl, state & ControlMask); io.AddKeyEvent(ImGuiMod_Shift, state & ShiftMask);
    io.AddKeyEvent(ImGuiMod_Alt, state & Mod1Mask); io.AddKeyEvent(ImGuiMod_Super, state & Mod4Mask);
}

void handle(XEvent& ev, bool& got) {
    ImGuiIO& io = ImGui::GetIO();
    if (p_XFilterEvent && p_XFilterEvent(&ev, None)) return;
    got = true;
    switch (ev.type) {
        case ClientMessage: if ((Atom)ev.xclient.data.l[0] == a_wm_delete) g_closed = true; break;
        case ConfigureNotify: g_w = ev.xconfigure.width; g_h = ev.xconfigure.height; break;
        case MotionNotify: io.AddMousePosEvent((float)ev.xmotion.x, (float)ev.xmotion.y); mods(ev.xmotion.state); break;
        case LeaveNotify: io.AddMousePosEvent(-FLT_MAX, -FLT_MAX); break;
        case ButtonPress: case ButtonRelease: {
            const bool down = ev.type == ButtonPress;
            io.AddMousePosEvent((float)ev.xbutton.x, (float)ev.xbutton.y); mods(ev.xbutton.state);
            switch (ev.xbutton.button) {
                case 1: io.AddMouseButtonEvent(0, down); break; case 3: io.AddMouseButtonEvent(1, down); break;
                case 2: io.AddMouseButtonEvent(2, down); break;
                case 4: if (down) io.AddMouseWheelEvent(0, 1.0f); break; case 5: if (down) io.AddMouseWheelEvent(0, -1.0f); break;
                case 6: if (down) io.AddMouseWheelEvent(1.0f, 0); break; case 7: if (down) io.AddMouseWheelEvent(-1.0f, 0); break;
            }
            break;
        }
        case KeyPress: case KeyRelease: {
            const bool down = ev.type == KeyPress;
            mods(ev.xkey.state);
            KeySym ks = 0; char buf[64] = {0};
            int n = p_XLookupString(&ev.xkey, buf, (int)sizeof(buf) - 1, &ks, nullptr);   // keysym (and Latin-1 text)
            if (down && g_ic && p_Xutf8LookupString) {                                      // full UTF-8 text when an input method exists
                Status st = 0; KeySym ks2 = 0; char ub[64] = {0};
                int m = p_Xutf8LookupString(g_ic, &ev.xkey, ub, (int)sizeof(ub) - 1, &ks2, &st);
                if (st != XBufferOverflow && m >= 0) { memcpy(buf, ub, (size_t)m); n = m; if (ks2) ks = ks2; }
            }
            ImGuiKey k = ks == 0xfe20 ? ImGuiKey_Tab : map_key(ks);                         // 0xfe20 = Shift+Tab
            if (k != ImGuiKey_None) io.AddKeyEvent(k, down);
            if (ks == XK_Control_L || ks == XK_Control_R) io.AddKeyEvent(ImGuiMod_Ctrl, down);
            if (ks == XK_Shift_L || ks == XK_Shift_R) io.AddKeyEvent(ImGuiMod_Shift, down);
            if (ks == XK_Alt_L || ks == XK_Alt_R) io.AddKeyEvent(ImGuiMod_Alt, down);
            if (down && n > 0 && !(ev.xkey.state & (ControlMask | Mod1Mask))) {
                buf[n] = 0;
                if ((unsigned char)buf[0] >= 32 && (unsigned char)buf[0] != 127) io.AddInputCharactersUTF8(buf);
            }
            break;
        }
        case FocusOut: io.AddFocusEvent(false); break;
        case FocusIn: io.AddFocusEvent(true); break;
        case SelectionRequest: {                                            // someone pastes what we copied
            XSelectionRequestEvent& rq = ev.xselectionrequest; XEvent r{}; r.xselection.type = SelectionNotify;
            r.xselection.display = rq.display; r.xselection.requestor = rq.requestor; r.xselection.selection = rq.selection;
            r.xselection.target = rq.target; r.xselection.time = rq.time; r.xselection.property = None;
            Atom prop = rq.property ? rq.property : rq.target;
            if (rq.target == a_targets) {
                Atom t[2] = {a_utf8, XA_STRING};
                p_XChangeProperty(g_dpy, rq.requestor, prop, XA_ATOM, 32, PropModeReplace, (unsigned char*)t, 2); r.xselection.property = prop;
            } else if (rq.target == a_utf8 || rq.target == XA_STRING) {
                p_XChangeProperty(g_dpy, rq.requestor, prop, rq.target, 8, PropModeReplace, (const unsigned char*)g_clip_own.data(), (int)g_clip_own.size());
                r.xselection.property = prop;
            }
            p_XSendEvent(g_dpy, rq.requestor, False, 0, &r);
            break;
        }
        default: got = false; break;
    }
}
}  // namespace

namespace plat {

bool create(const char* title, int w, int h) {
    if (!load_x()) { fprintf(stderr, "[launcher] libX11 not found (no graphical environment?)\n"); return false; }
    setlocale(LC_CTYPE, "");
    g_dpy = p_XOpenDisplay(nullptr);
    if (!g_dpy) { fprintf(stderr, "[launcher] cannot open the display (is DISPLAY set / is an X server running?)\n"); return false; }
    int scr = DefaultScreen(g_dpy);
    g_vis = DefaultVisual(g_dpy, scr); g_depth = DefaultDepth(g_dpy, scr);
    if (g_depth != 24 && g_depth != 32) { fprintf(stderr, "[launcher] unsupported colour depth %d\n", g_depth); return false; }
    g_win = p_XCreateSimpleWindow(g_dpy, RootWindow(g_dpy, scr), 0, 0, (unsigned)w, (unsigned)h, 0, 0, 0x12161c);
    p_XStoreName(g_dpy, g_win, title);
    a_net_name = p_XInternAtom(g_dpy, "_NET_WM_NAME", False); a_utf8 = p_XInternAtom(g_dpy, "UTF8_STRING", False);
    p_XChangeProperty(g_dpy, g_win, a_net_name, a_utf8, 8, PropModeReplace, (const unsigned char*)title, (int)strlen(title));
    p_XSelectInput(g_dpy, g_win, ExposureMask | StructureNotifyMask | PointerMotionMask | ButtonPressMask | ButtonReleaseMask |
                                 KeyPressMask | KeyReleaseMask | LeaveWindowMask | FocusChangeMask);
    a_wm_delete = p_XInternAtom(g_dpy, "WM_DELETE_WINDOW", False);
    p_XSetWMProtocols(g_dpy, g_win, &a_wm_delete, 1);
    a_clip = p_XInternAtom(g_dpy, "CLIPBOARD", False); a_targets = p_XInternAtom(g_dpy, "TARGETS", False);
    a_prop = p_XInternAtom(g_dpy, "NEXUS_PASTE", False);
    XSizeHints* sh = p_XAllocSizeHints ? p_XAllocSizeHints() : nullptr;
    if (sh) { sh->flags = PMinSize; sh->min_width = 640; sh->min_height = 420; p_XSetWMNormalHints(g_dpy, g_win, sh); p_XFree(sh); }
    {   // window / taskbar icon (works without installing anything: it is a window property)
        std::vector<long> ic; for (int n : {64, 32}) { auto px = nx_icon_pixels(n); ic.push_back(n); ic.push_back(n); for (uint32_t v : px) ic.push_back((long)v); }
        Atom a_icon = p_XInternAtom(g_dpy, "_NET_WM_ICON", False), a_card = p_XInternAtom(g_dpy, "CARDINAL", False);
        p_XChangeProperty(g_dpy, g_win, a_icon, a_card, 32, PropModeReplace, (const unsigned char*)ic.data(), (int)ic.size());
    }
    p_XMapWindow(g_dpy, g_win);
    g_gc = p_XCreateGC(g_dpy, g_win, 0, nullptr);
    g_w = w; g_h = h;
    // input method (accents, compose keys, IME); optional
    if (p_XSetLocaleModifiers && p_XOpenIM && p_XCreateIC) {
        p_XSetLocaleModifiers("");
        g_im = p_XOpenIM(g_dpy, nullptr, nullptr, nullptr);
        if (g_im) g_ic = p_XCreateIC(g_im, XNInputStyle, XIMPreeditNothing | XIMStatusNothing, XNClientWindow, g_win, XNFocusWindow, g_win, nullptr);
    }
    // DPI from Xft.dpi
    g_scale = 1.0f;
    if (p_XResourceManagerString) if (char* rm = p_XResourceManagerString(g_dpy)) {
        const char* p = strstr(rm, "Xft.dpi:"); if (p) { float d = (float)atof(p + 8); if (d >= 72 && d <= 400) g_scale = d / 96.0f; }
    }
    if (const char* s = getenv("NEXUS_UI_SCALE")) { float v = (float)atof(s); if (v >= 0.5f && v <= 4) g_scale = v; }
    return true;
}

void size(int& w, int& h) { w = g_w; h = g_h; }
float dpi_scale() { return g_scale; }

bool pump(int wait_ms, bool& got_event) {
    got_event = false;
    if (!p_XPending(g_dpy) && wait_ms > 0) {
        struct pollfd pf{p_XConnectionNumber(g_dpy), POLLIN, 0};
        poll(&pf, 1, wait_ms);
    }
    while (p_XPending(g_dpy)) { XEvent ev; p_XNextEvent(g_dpy, &ev); handle(ev, got_event); }
    return !g_closed;
}

void present(const uint32_t* px, int w, int h) {
    // XCreateImage wants a malloc'ed buffer it may own; we pass ours and detach before freeing.
    XImage* im = p_XCreateImage(g_dpy, g_vis, (unsigned)g_depth, ZPixmap, 0, (char*)px, (unsigned)w, (unsigned)h, 32, 0);
    if (!im) return;
    p_XPutImage(g_dpy, g_win, g_gc, im, 0, 0, 0, 0, (unsigned)w, (unsigned)h);
    im->data = nullptr; p_XFree(im);                      // do not free our pixels
    p_XFlush(g_dpy);
}

void set_clipboard(const char* t) {
    g_clip_own = t ? t : "";
    if (g_dpy) p_XSetSelectionOwner(g_dpy, a_clip, g_win, CurrentTime);
}

std::string get_clipboard() {
    if (!g_dpy) return "";
    Window owner = p_XGetSelectionOwner(g_dpy, a_clip);
    if (owner == g_win) return g_clip_own;
    if (owner == None) return "";
    p_XDeleteProperty(g_dpy, g_win, a_prop);
    p_XConvertSelection(g_dpy, a_clip, a_utf8, a_prop, g_win, CurrentTime);
    p_XFlush(g_dpy);
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.4) {
        while (p_XPending(g_dpy)) {
            XEvent ev; p_XNextEvent(g_dpy, &ev);
            if (ev.type == SelectionNotify && ev.xselection.selection == a_clip) {
                if (ev.xselection.property == None) return "";
                Atom type; int fmt; unsigned long n, after; unsigned char* data = nullptr; std::string out;
                if (p_XGetWindowProperty(g_dpy, g_win, a_prop, 0, 1 << 20, True, AnyPropertyType, &type, &fmt, &n, &after, &data) == Success && data) {
                    out.assign((char*)data, n * (size_t)(fmt / 8)); p_XFree(data);
                }
                return out;
            }
            bool g; handle(ev, g);
        }
        struct pollfd pf{p_XConnectionNumber(g_dpy), POLLIN, 0}; poll(&pf, 1, 20);
    }
    return "";
}

void cancel_close() { g_closed = false; }

void destroy() {
    if (!g_dpy) return;
    if (g_ic && p_XDestroyIC) p_XDestroyIC(g_ic);
    if (g_im && p_XCloseIM) p_XCloseIM(g_im);
    if (g_gc) p_XFreeGC(g_dpy, g_gc);
    if (g_win) p_XDestroyWindow(g_dpy, g_win);
    p_XCloseDisplay(g_dpy); g_dpy = nullptr;
}

}  // namespace plat
