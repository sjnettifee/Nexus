// probe.h -- `nexus_server --probe`: describe this machine as JSON and exit.
//
// Used by the launcher / headless runner to choose defaults automatically
// (threads, which GPU, whether prefill acceleration makes sense).  It needs no
// license, loads no model, and never fails hard: every section reports what it
// could find, and a missing Vulkan loader is reported as data, not as an error.
//
// Place at src/api/probe.h; needs gpu/vk_loader.h (for -DNEXUS_VULKAN builds).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#endif
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64)) && !defined(_MSC_VER)
#  include <cpuid.h>
#endif
#ifdef NEXUS_VULKAN
#  include "gpu/vk_loader.h"
#endif

namespace nexus { namespace probe {

inline std::string jesc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

inline std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t')) s.pop_back();
    size_t i = 0; while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

inline std::string cpu_brand() {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0x80000000, nullptr) >= 0x80000004) {
        char buf[49] = {};
        for (unsigned i = 0; i < 3; ++i) {
            __get_cpuid(0x80000002 + i, &a, &b, &c, &d);
            memcpy(buf + 16 * i + 0, &a, 4); memcpy(buf + 16 * i + 4, &b, 4);
            memcpy(buf + 16 * i + 8, &c, 4); memcpy(buf + 16 * i + 12, &d, 4);
        }
        return trim(buf);
    }
#endif
#if defined(__linux__)
    if (FILE* f = fopen("/proc/cpuinfo", "r")) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "model name", 10) || !strncmp(line, "Model", 5) || !strncmp(line, "Hardware", 8)) {
                const char* p = strchr(line, ':');
                if (p) { std::string r = trim(p + 1); fclose(f); return r; }
            }
        }
        fclose(f);
    }
#elif defined(__APPLE__)
    char buf[256]; size_t n = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, nullptr, 0) == 0) return trim(buf);
#endif
    return "unknown";
}

inline const char* arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__)
    return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__)
    return "arm";
#else
    return "unknown";
#endif
}

inline const char* os_name() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "other";
#endif
}

// Physical core count (not hyper-threads); falls back to logical/2, then 1.
inline int physical_cores() {
    int logical = (int)std::thread::hardware_concurrency();
    if (logical < 1) logical = 1;
#if defined(__linux__)
    // unique (package_id, core_id) pairs
    std::vector<std::string> seen;
    for (int i = 0; i < logical * 4 && i < 1024; ++i) {
        char p1[160], p2[160];
        snprintf(p1, sizeof(p1), "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        snprintf(p2, sizeof(p2), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        FILE* f1 = fopen(p1, "r");
        if (!f1) { if (i >= logical) break; else continue; }
        char a[32] = {}, b[32] = {};
        if (!fgets(a, sizeof(a), f1)) a[0] = 0;
        fclose(f1);
        if (FILE* f2 = fopen(p2, "r")) { if (!fgets(b, sizeof(b), f2)) b[0] = 0; fclose(f2); }
        std::string key = trim(b) + ":" + trim(a);
        bool dup = false; for (auto& s : seen) if (s == key) { dup = true; break; }
        if (!dup) seen.push_back(key);
    }
    if (!seen.empty()) return (int)seen.size();
#elif defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> v(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(v.data(), &len)) {
            int n = 0; for (auto& e : v) if (e.Relationship == RelationProcessorCore) ++n;
            if (n > 0) return n;
        }
    }
#elif defined(__APPLE__)
    int n = 0; size_t sz = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0) return n;
#endif
    return logical > 1 ? logical / 2 : 1;
}

inline void ram_bytes(uint64_t& total, uint64_t& avail) {
    total = avail = 0;
#if defined(_WIN32)
    MEMORYSTATUSEX m; m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) { total = m.ullTotalPhys; avail = m.ullAvailPhys; }
#elif defined(__APPLE__)
    uint64_t t = 0; size_t sz = sizeof(t);
    if (sysctlbyname("hw.memsize", &t, &sz, nullptr, 0) == 0) total = avail = t;  // free RAM not cheap to get here
#else
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long kb;
            if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) total = kb * 1024ULL;
            else if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) avail = kb * 1024ULL;
        }
        fclose(f);
    }
#endif
}

inline std::string cpu_features_json() {
    std::string o = "{";
    auto add = [&](const char* n, bool v) { if (o.size() > 1) o += ","; o += "\""; o += n; o += "\":"; o += v ? "true" : "false"; };
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    add("sse4_1", __builtin_cpu_supports("sse4.1"));
    add("avx",    __builtin_cpu_supports("avx"));
    add("avx2",   __builtin_cpu_supports("avx2"));
    add("fma",    __builtin_cpu_supports("fma"));
    add("avx512f", __builtin_cpu_supports("avx512f"));
#elif defined(__aarch64__)
    add("neon", true);   // mandatory on AArch64
#endif
    return o + "}";
}

inline std::string vulkan_json() {
#ifndef NEXUS_VULKAN
    return "{\"built\":false,\"loaded\":false,\"devices\":[]}";
#else
    namespace v = ::nexus::gpu::vkl;
    std::string o = "{\"built\":true,";
    if (!v::load()) {
        o += "\"loaded\":false,\"error\":\"" + jesc(v::error()) + "\",\"devices\":[]}";
        return o;
    }
    o += "\"loaded\":true,\"loader\":\"" + jesc(v::lib_path()) + "\",";

    uint32_t ne = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ex(ne);
    if (ne) vkEnumerateInstanceExtensionProperties(nullptr, &ne, ex.data());

    static const char* pe = "VK_KHR_portability_enumeration";
    VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pApplicationInfo = &ai;
    for (auto& e : ex) if (!strcmp(e.extensionName, pe)) {
        ci.enabledExtensionCount = 1; ci.ppEnabledExtensionNames = &pe; ci.flags |= 0x1; break;
    }
    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, nullptr, &inst);
    if (r != VK_SUCCESS) {
        char b[160];
        snprintf(b, sizeof(b), "\"instance_error\":%d,\"hint\":\"%s\",\"devices\":[]}", (int)r,
                 r == VK_ERROR_INCOMPATIBLE_DRIVER ? "Vulkan loader present but no GPU driver (ICD) installed"
                                                   : "vkCreateInstance failed");
        return o + b;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    if (n) vkEnumeratePhysicalDevices(inst, &n, devs.data());
    o += "\"devices\":[";
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(devs[i], &p);
        VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
        uint64_t vram = 0;
        for (uint32_t j = 0; j < mp.memoryHeapCount; ++j)
            if (mp.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                if (mp.memoryHeaps[j].size > vram) vram = mp.memoryHeaps[j].size;
        // Does it have a compute-capable queue family?
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        if (nq) vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qf.data());
        bool compute = false; for (auto& q : qf) if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) compute = true;
        const char* dt = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete"
                       : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
                       : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "virtual"
                       : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "cpu" : "other";
        std::string why; bool meets = v::device_meets_requirements(p.limits, why);
        bool soft = v::device_is_software(p);
        // usable = what the engine would actually auto-select
        bool usable = compute && meets && (!soft || v::allow_software());
        if (!compute) why = "no compute queue";
        else if (meets && soft && !usable) why = "software renderer (slower than the CPU kernels)";
        char b[900];
        snprintf(b, sizeof(b),
            "%s{\"index\":%u,\"name\":\"%s\",\"type\":\"%s\",\"vendor_id\":%u,\"device_id\":%u,"
            "\"vram_bytes\":%llu,\"api_version\":\"%u.%u.%u\",\"driver_version\":%u,\"compute\":%s,"
            "\"max_workgroup_invocations\":%u,\"max_shared_memory\":%u,\"max_storage_buffer_range\":%u,"
            "\"software\":%s,\"usable\":%s,\"reason\":\"%s\"}",
            i ? "," : "", i, jesc(p.deviceName).c_str(), dt, p.vendorID, p.deviceID,
            (unsigned long long)vram,
            VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion),
            p.driverVersion, compute ? "true" : "false",
            p.limits.maxComputeWorkGroupInvocations, p.limits.maxComputeSharedMemorySize,
            p.limits.maxStorageBufferRange, soft ? "true" : "false", usable ? "true" : "false",
            jesc(why).c_str());
        o += b;
    }
    o += "]}";
    vkDestroyInstance(inst, nullptr);
    return o;
#endif
}

// Returns the full probe document (one JSON object, newline-terminated).
inline std::string run() {
    uint64_t rt, ra; ram_bytes(rt, ra);
    char head[256];
    snprintf(head, sizeof(head), "{\"probe_version\":1,\"os\":\"%s\",\"arch\":\"%s\",", os_name(), arch_name());
    std::string o = head;
    char cpu[512];
    snprintf(cpu, sizeof(cpu), "\"cpu\":{\"brand\":\"%s\",\"logical\":%u,\"physical\":%d,\"features\":",
             jesc(cpu_brand()).c_str(), std::thread::hardware_concurrency(), physical_cores());
    o += cpu; o += cpu_features_json(); o += "},";
    char mem[160];
    snprintf(mem, sizeof(mem), "\"ram\":{\"total_bytes\":%llu,\"available_bytes\":%llu},",
             (unsigned long long)rt, (unsigned long long)ra);
    o += mem;
    o += "\"vulkan\":" + vulkan_json() + "}\n";
    return o;
}

}} // namespace nexus::probe
