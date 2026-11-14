// platform.h -- the only OS-specific surface of the launcher: a window we can draw pixels into and
// that feeds mouse / keyboard / clipboard into Dear ImGui.  Implemented in plat_x11.cpp (Linux, loads
// libX11 at run time -- nothing to install or link) and plat_win32.cpp (Windows).
#pragma once
#include <cstdint>
#include <string>

namespace plat {
bool create(const char* title, int w, int h);          // false: no display available
void size(int& w, int& h);                              // current client size in pixels
float dpi_scale();                                      // 1.0 = 96 dpi
// Handle pending OS events (feeding ImGuiIO).  Waits up to wait_ms for the first one.
// Returns false once the user asked to close the window.  got_event is set when anything happened.
bool pump(int wait_ms, bool& got_event);
void present(const uint32_t* pixels, int w, int h);     // 0x00RRGGBB, row-major
void set_clipboard(const char* utf8);
std::string get_clipboard();
void cancel_close();                                    // user pressed the close button but we want to stay open
void destroy();
}
