// vk_loader.h -- load the Vulkan loader at RUNTIME instead of linking it.
//
// Why: if the binary links libvulkan.so.1 / vulkan-1.dll directly, it refuses to
// start at all on a machine that has no Vulkan loader, even when the user only
// wants CPU.  With this header the loader is opened with dlopen/LoadLibrary at
// startup; if it is missing (or broken) vk::load() returns false, the GPU backend
// reports "unavailable" and the engine runs on CPU.
//
// How it plugs in: this header includes vulkan.h with VK_NO_PROTOTYPES (types only,
// nothing to link) and then #defines every vkXxx function the backend calls to a
// slot in a function-pointer table.  Existing call sites such as
//     vkQueueSubmit(q, 1, &si, fence);
// compile unchanged.  Every code path that reaches a vk* call must first succeed at
// nexus::gpu::vkl::load().
//
// Environment:
//   NEXUS_VULKAN_LIB=<path>   use exactly this loader library (nothing else is tried)
//   NEXUS_VULKAN_LIB=none     never touch Vulkan (forces CPU)
#pragma once

#ifdef NEXUS_VULKAN

#ifndef VK_NO_PROTOTYPES
#  define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <limits.h>
#  include <unistd.h>
#  ifdef __APPLE__
#    include <mach-o/dyld.h>
#  endif
#endif

// Every Vulkan entry point the backend calls.  Add new ones HERE (and nowhere else).
#define NEXUS_VK_FUNCS(X) \
    X(CreateInstance) X(DestroyInstance) X(EnumerateInstanceExtensionProperties) \
    X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
    X(GetPhysicalDeviceMemoryProperties) X(GetPhysicalDeviceQueueFamilyProperties) \
    X(EnumerateDeviceExtensionProperties) X(CreateDevice) X(DestroyDevice) \
    X(GetDeviceQueue) X(DeviceWaitIdle) X(QueueSubmit) \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) \
    X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
    X(CreateShaderModule) X(DestroyShaderModule) \
    X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
    X(CreatePipelineLayout) X(DestroyPipelineLayout) \
    X(CreateComputePipelines) X(DestroyPipeline) \
    X(CreateDescriptorPool) X(DestroyDescriptorPool) X(ResetDescriptorPool) X(CreateQueryPool) X(DestroyQueryPool) X(GetQueryPoolResults) X(CmdResetQueryPool) X(CmdWriteTimestamp) \
    X(AllocateDescriptorSets) X(FreeDescriptorSets) X(UpdateDescriptorSets) \
    X(CreateCommandPool) X(DestroyCommandPool) \
    X(AllocateCommandBuffers) X(FreeCommandBuffers) \
    X(BeginCommandBuffer) X(EndCommandBuffer) X(ResetCommandBuffer) \
    X(CreateFence) X(DestroyFence) X(ResetFences) X(WaitForFences) \
    X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) \
    X(CmdDispatch) X(CmdCopyBuffer) X(CmdPipelineBarrier)

namespace nexus { namespace gpu { namespace vkl {

struct Table {
#define X(n) PFN_vk##n n = nullptr;
    NEXUS_VK_FUNCS(X)
#undef X
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
};

inline Table& T() { static Table t; return t; }

namespace detail {

struct State {
    std::mutex  mu;
    bool        tried  = false;
    bool        ok     = false;
    std::string err;
    std::string lib_path;      // what we ended up loading
    void*       handle = nullptr;
};
inline State& S() { static State s; return s; }

#ifdef _WIN32
inline void* open_lib(const char* p) { return (void*)LoadLibraryA(p); }
inline void* sym(void* h, const char* n) { return (void*)GetProcAddress((HMODULE)h, n); }
inline void  close_lib(void* h) { if (h) FreeLibrary((HMODULE)h); }
#else
inline void* open_lib(const char* p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
inline void* sym(void* h, const char* n) { return dlsym(h, n); }
inline void  close_lib(void* h) { if (h) dlclose(h); }
#endif

inline std::string exe_dir() {
#ifdef _WIN32
    char b[MAX_PATH * 2];
    DWORD n = GetModuleFileNameA(nullptr, b, sizeof(b));
    std::string s(b, n);
    size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string(".") : s.substr(0, k);
#elif defined(__APPLE__)
    char b[PATH_MAX]; uint32_t n = sizeof(b);
    if (_NSGetExecutablePath(b, &n) != 0) return ".";
    std::string s(b); size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string(".") : s.substr(0, k);
#else
    char b[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", b, sizeof(b) - 1);
    if (n <= 0) return ".";
    b[n] = 0;
    std::string s(b); size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string(".") : s.substr(0, k);
#endif
}

// Resolve every entry point from an opened loader.  Returns the name of the first
// missing function, or "" on success.
inline std::string resolve_all(void* h) {
    Table& t = T();
    t.GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)sym(h, "vkGetInstanceProcAddr");
    if (!t.GetInstanceProcAddr) return "vkGetInstanceProcAddr";
#define X(n) \
    t.n = (PFN_vk##n)sym(h, "vk" #n); \
    if (!t.n) return "vk" #n;
    NEXUS_VK_FUNCS(X)
#undef X
    return "";
}

inline bool try_lib(const std::string& path, std::string& why) {
    State& s = S();
    void* h = open_lib(path.c_str());
    if (!h) { why += "  " + path + ": not found\n"; return false; }
    std::string miss = resolve_all(h);
    if (!miss.empty()) {
        why += "  " + path + ": loaded but missing " + miss + " (not a Vulkan 1.0+ loader?)\n";
        close_lib(h);
        T() = Table();
        return false;
    }
    s.handle = h;
    s.lib_path = path;
    return true;
}

} // namespace detail

// Idempotent and thread-safe.  Returns true if Vulkan entry points are usable.
inline bool load() {
    auto& s = detail::S();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.tried) return s.ok;
    s.tried = true;

    const char* ov = std::getenv("NEXUS_VULKAN_LIB");
    if (ov && *ov) {
        if (std::strcmp(ov, "none") == 0 || std::strcmp(ov, "off") == 0) {
            s.err = "Vulkan disabled by NEXUS_VULKAN_LIB=none";
            return s.ok = false;
        }
        std::string why;
        s.ok = detail::try_lib(ov, why);
        if (!s.ok) s.err = "NEXUS_VULKAN_LIB=" + std::string(ov) + " could not be used:\n" + why;
        return s.ok;
    }

    std::vector<std::string> cands;
#ifdef _WIN32
    cands = {"vulkan-1.dll"};
#elif defined(__APPLE__)
    // Vulkan on macOS = MoltenVK (Metal translation).  System/SDK loader first,
    // then a MoltenVK shipped next to the program.
    cands = {"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"};
#else
    cands = {"libvulkan.so.1", "libvulkan.so"};
#endif
    // Last resort: a loader shipped next to the executable (we never prefer it over
    // the system one: a system loader knows where the system's GPU drivers are).
    std::vector<std::string> local;
    {
        std::string d = detail::exe_dir();
        for (auto& c : cands) local.push_back(d + "/" + c);
#ifdef __APPLE__
        local.push_back(d + "/../Frameworks/libMoltenVK.dylib");
#endif
    }
    std::string why;
    for (auto& c : cands)  if (detail::try_lib(c, why))  return s.ok = true;
    for (auto& c : local)  if (detail::try_lib(c, why))  return s.ok = true;
    s.err = "no Vulkan loader found; tried:\n" + why;
    return s.ok = false;
}

inline const std::string& error()    { return detail::S().err; }
inline const std::string& lib_path() { return detail::S().lib_path; }

// ─── Device requirements ──────────────────────────────────────────────────────
// What our compute shaders need.  Used by the backend (device selection) and by `--probe`.
// A device that fails is never selected, so the engine stays on the CPU path instead of
// mis-computing.  Needs: 256-invocation workgroups (16x16 and 256x1 kernels), 16 KiB shared
// memory (mat_vec tiles), 9 storage buffers per stage (DeltaNet scan), 48 B push constants.
inline bool device_meets_requirements(const VkPhysicalDeviceLimits& l, std::string& why) {
    char b[200];
    if (l.maxComputeWorkGroupInvocations < 256) {
        snprintf(b, sizeof(b), "maxComputeWorkGroupInvocations=%u (<256)", l.maxComputeWorkGroupInvocations); why = b; return false; }
    if (l.maxComputeWorkGroupSize[0] < 256 || l.maxComputeWorkGroupSize[1] < 16) {
        snprintf(b, sizeof(b), "maxComputeWorkGroupSize=%ux%u (<256x16)", l.maxComputeWorkGroupSize[0], l.maxComputeWorkGroupSize[1]); why = b; return false; }
    if (l.maxComputeSharedMemorySize < 16384) {
        snprintf(b, sizeof(b), "maxComputeSharedMemorySize=%u (<16384)", l.maxComputeSharedMemorySize); why = b; return false; }
    if (l.maxPerStageDescriptorStorageBuffers < 9) {
        snprintf(b, sizeof(b), "maxPerStageDescriptorStorageBuffers=%u (<9)", l.maxPerStageDescriptorStorageBuffers); why = b; return false; }
    if (l.maxPushConstantsSize < 48) {
        snprintf(b, sizeof(b), "maxPushConstantsSize=%u (<48)", l.maxPushConstantsSize); why = b; return false; }
    if (l.maxBoundDescriptorSets < 1) { why = "maxBoundDescriptorSets=0"; return false; }
    why.clear();
    return true;
}

// Software rasterizers (llvmpipe/lavapipe, SwiftShader) report as Vulkan devices.  They are
// far slower than our CPU kernels, so they are never chosen automatically.
inline bool device_is_software(const VkPhysicalDeviceProperties& p) {
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) return true;
    std::string n = p.deviceName;
    return n.find("llvmpipe") != std::string::npos || n.find("lavapipe") != std::string::npos ||
           n.find("SwiftShader") != std::string::npos || n.find("softpipe") != std::string::npos;
}
inline bool allow_software() {
    const char* e = std::getenv("NEXUS_VULKAN_ALLOW_SOFTWARE");
    return e && *e && *e != '0';
}

}}} // namespace nexus::gpu::vkl

// Route the existing vkXxx(...) call sites through the table.
#define vkGetInstanceProcAddr ::nexus::gpu::vkl::T().GetInstanceProcAddr
#define NEXUS_VK_ALIAS(n) ::nexus::gpu::vkl::T().n
#define X(n)
#undef X
#define vkCreateInstance                      NEXUS_VK_ALIAS(CreateInstance)
#define vkDestroyInstance                     NEXUS_VK_ALIAS(DestroyInstance)
#define vkEnumerateInstanceExtensionProperties NEXUS_VK_ALIAS(EnumerateInstanceExtensionProperties)
#define vkEnumeratePhysicalDevices            NEXUS_VK_ALIAS(EnumeratePhysicalDevices)
#define vkGetPhysicalDeviceProperties         NEXUS_VK_ALIAS(GetPhysicalDeviceProperties)
#define vkGetPhysicalDeviceMemoryProperties   NEXUS_VK_ALIAS(GetPhysicalDeviceMemoryProperties)
#define vkGetPhysicalDeviceQueueFamilyProperties NEXUS_VK_ALIAS(GetPhysicalDeviceQueueFamilyProperties)
#define vkEnumerateDeviceExtensionProperties  NEXUS_VK_ALIAS(EnumerateDeviceExtensionProperties)
#define vkCreateDevice                        NEXUS_VK_ALIAS(CreateDevice)
#define vkDestroyDevice                       NEXUS_VK_ALIAS(DestroyDevice)
#define vkGetDeviceQueue                      NEXUS_VK_ALIAS(GetDeviceQueue)
#define vkDeviceWaitIdle                      NEXUS_VK_ALIAS(DeviceWaitIdle)
#define vkQueueSubmit                         NEXUS_VK_ALIAS(QueueSubmit)
#define vkCreateBuffer                        NEXUS_VK_ALIAS(CreateBuffer)
#define vkDestroyBuffer                       NEXUS_VK_ALIAS(DestroyBuffer)
#define vkGetBufferMemoryRequirements         NEXUS_VK_ALIAS(GetBufferMemoryRequirements)
#define vkAllocateMemory                      NEXUS_VK_ALIAS(AllocateMemory)
#define vkFreeMemory                          NEXUS_VK_ALIAS(FreeMemory)
#define vkBindBufferMemory                    NEXUS_VK_ALIAS(BindBufferMemory)
#define vkMapMemory                           NEXUS_VK_ALIAS(MapMemory)
#define vkUnmapMemory                         NEXUS_VK_ALIAS(UnmapMemory)
#define vkCreateShaderModule                  NEXUS_VK_ALIAS(CreateShaderModule)
#define vkDestroyShaderModule                 NEXUS_VK_ALIAS(DestroyShaderModule)
#define vkCreateDescriptorSetLayout           NEXUS_VK_ALIAS(CreateDescriptorSetLayout)
#define vkDestroyDescriptorSetLayout          NEXUS_VK_ALIAS(DestroyDescriptorSetLayout)
#define vkCreatePipelineLayout                NEXUS_VK_ALIAS(CreatePipelineLayout)
#define vkDestroyPipelineLayout               NEXUS_VK_ALIAS(DestroyPipelineLayout)
#define vkCreateComputePipelines              NEXUS_VK_ALIAS(CreateComputePipelines)
#define vkDestroyPipeline                     NEXUS_VK_ALIAS(DestroyPipeline)
#define vkCreateDescriptorPool                NEXUS_VK_ALIAS(CreateDescriptorPool)
#define vkDestroyDescriptorPool               NEXUS_VK_ALIAS(DestroyDescriptorPool)
#define vkResetDescriptorPool                 NEXUS_VK_ALIAS(ResetDescriptorPool)
#define vkCreateQueryPool                     NEXUS_VK_ALIAS(CreateQueryPool)
#define vkDestroyQueryPool                    NEXUS_VK_ALIAS(DestroyQueryPool)
#define vkGetQueryPoolResults                 NEXUS_VK_ALIAS(GetQueryPoolResults)
#define vkCmdResetQueryPool                   NEXUS_VK_ALIAS(CmdResetQueryPool)
#define vkCmdWriteTimestamp                   NEXUS_VK_ALIAS(CmdWriteTimestamp)
#define vkAllocateDescriptorSets              NEXUS_VK_ALIAS(AllocateDescriptorSets)
#define vkFreeDescriptorSets                  NEXUS_VK_ALIAS(FreeDescriptorSets)
#define vkUpdateDescriptorSets                NEXUS_VK_ALIAS(UpdateDescriptorSets)
#define vkCreateCommandPool                   NEXUS_VK_ALIAS(CreateCommandPool)
#define vkDestroyCommandPool                  NEXUS_VK_ALIAS(DestroyCommandPool)
#define vkAllocateCommandBuffers              NEXUS_VK_ALIAS(AllocateCommandBuffers)
#define vkFreeCommandBuffers                  NEXUS_VK_ALIAS(FreeCommandBuffers)
#define vkBeginCommandBuffer                  NEXUS_VK_ALIAS(BeginCommandBuffer)
#define vkEndCommandBuffer                    NEXUS_VK_ALIAS(EndCommandBuffer)
#define vkResetCommandBuffer                  NEXUS_VK_ALIAS(ResetCommandBuffer)
#define vkCreateFence                         NEXUS_VK_ALIAS(CreateFence)
#define vkDestroyFence                        NEXUS_VK_ALIAS(DestroyFence)
#define vkResetFences                         NEXUS_VK_ALIAS(ResetFences)
#define vkWaitForFences                       NEXUS_VK_ALIAS(WaitForFences)
#define vkCmdBindPipeline                     NEXUS_VK_ALIAS(CmdBindPipeline)
#define vkCmdBindDescriptorSets               NEXUS_VK_ALIAS(CmdBindDescriptorSets)
#define vkCmdPushConstants                    NEXUS_VK_ALIAS(CmdPushConstants)
#define vkCmdDispatch                         NEXUS_VK_ALIAS(CmdDispatch)
#define vkCmdCopyBuffer                       NEXUS_VK_ALIAS(CmdCopyBuffer)
#define vkCmdPipelineBarrier                  NEXUS_VK_ALIAS(CmdPipelineBarrier)

#endif // NEXUS_VULKAN
