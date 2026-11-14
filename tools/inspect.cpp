/*
 * nexus/tools/inspect.cpp
 *
 * First test of the real engine.
 * Opens a GGUF file, reads metadata, reports memory plan.
 * No llama.cpp. No wrappers. This is the start.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make nexus_inspect
 *
 * Run:
 *   ./nexus_inspect /path/to/model.gguf
 */

#include "core/memory.h"
#include "gguf/parser.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace nexus;
using namespace nexus::gguf;

static void print_banner() {
    printf("\n");
    printf("  ╔══════════════════════════════════════╗\n");
    printf("  ║   NEXUS  —  Open Inference Engine   ║\n");
    printf("  ║   Model Inspector                   ║\n");
    printf("  ╚══════════════════════════════════════╝\n\n");
}

static void print_memory_plan(const ModelMeta& meta, size_t file_size) {
    // Read actual system state
    FILE* f = fopen("/proc/meminfo", "r");
    size_t mem_available_kb = 0, mem_total_kb = 0;
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemAvailable: %zu kB", &mem_available_kb) == 1) continue;
            if (sscanf(line, "MemTotal: %zu kB", &mem_total_kb)          == 1) continue;
        }
        fclose(f);
    }

    size_t ram_avail = mem_available_kb * 1024;
    size_t ram_total = mem_total_kb * 1024;

    printf("── Memory Plan ─────────────────────────────────\n");
    printf("  System RAM: %s avail / %s total\n",
           MemoryManager::fmt_bytes(ram_avail).c_str(),
           MemoryManager::fmt_bytes(ram_total).c_str());
    printf("  Model file: %s\n", MemoryManager::fmt_bytes(file_size).c_str());

    // KV cache per context length
    // bytes = 2 (K+V) × 2 (f16) × n_kv_heads × head_dim × n_layers × n_ctx
    size_t kv_per_token = 2 * 2 * meta.n_kv_heads * meta.head_dim * meta.n_layers;
    printf("\n  KV Cache per context length:\n");
    for (int ctx : {1024, 2048, 4096, 8192, 16384, 32768}) {
        if ((uint32_t)ctx > meta.context_length) break;
        size_t kv = kv_per_token * ctx;
        size_t total_needed = file_size + kv;
        bool fits = total_needed < ram_avail * 9 / 10;
        printf("    ctx=%6d  KV=%s  total=%s  %s\n",
               ctx,
               MemoryManager::fmt_bytes(kv).c_str(),
               MemoryManager::fmt_bytes(total_needed).c_str(),
               fits ? "✓ fits" : "✗ tight");
    }

    printf("\n  Chunked KV page size (256 tokens):\n");
    size_t page_bytes = kv_per_token * 256;
    printf("    %s per page — allocate on demand, free instantly on clear\n",
           MemoryManager::fmt_bytes(page_bytes).c_str());

    printf("\n  Huge page allocation:\n");
    size_t layers = meta.n_layers;
    size_t bytes_per_layer = layers > 0 ? file_size / layers : 0;
    size_t huge_pages_per_layer = (bytes_per_layer + 2*1024*1024 - 1) / (2*1024*1024);
    printf("    %s per layer → %zu huge pages (2MB) per layer\n",
           MemoryManager::fmt_bytes(bytes_per_layer).c_str(),
           huge_pages_per_layer);
    printf("    TLB entries needed: %zu (vs %zu with 4KB pages — %.0fx fewer)\n",
           huge_pages_per_layer * meta.n_layers,
           (file_size + 4095) / 4096,
           (float)((file_size + 4095) / 4096) / (huge_pages_per_layer * meta.n_layers));

    printf("────────────────────────────────────────────────\n\n");
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [--tensors]\n", argv[0]);
        return 1;
    }

    bool show_tensors = false;
    bool show_kv      = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--tensors") show_tensors = true;
        if (std::string(argv[i]) == "--kv")      show_kv      = true;
    }
    std::string path = argv[1];

    printf("[open] %s\n", path.c_str());

    auto t0 = std::chrono::steady_clock::now();

    Parser parser;
    if (!parser.open(path)) {
        fprintf(stderr, "Failed to open model.\n");
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("[open] Header read in %.1f ms\n\n", ms);

    parser.print_meta();

    // File size
    FILE* ftest = fopen(path.c_str(), "rb");
    size_t file_size = 0;
    if (ftest) {
        fseek(ftest, 0, SEEK_END);
        file_size = ftell(ftest);
        fclose(ftest);
    }

    print_memory_plan(parser.meta(), file_size);

    if (show_kv) {
        // Dump all raw GGUF key-value pairs for diagnosis
        printf("── Raw GGUF Metadata (all keys) ─────────────────\n");
        const auto& kv = parser.kv();
        // Print in sorted order
        std::vector<std::string> keys;
        for (const auto& [k, v] : kv) keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (const auto& k : keys) {
            const auto& val = kv.at(k);
            printf("  %-55s = ", k.c_str());
            if (const auto* v = std::get_if<uint32_t>(&val))  printf("%u\n",  *v);
            else if (const auto* v = std::get_if<uint64_t>(&val)) printf("%llu\n", (unsigned long long)*v);
            else if (const auto* v = std::get_if<int32_t>(&val))  printf("%d\n",  *v);
            else if (const auto* v = std::get_if<float>(&val))    printf("%.6g\n", *v);
            else if (const auto* v = std::get_if<double>(&val))   printf("%.6g\n", *v);
            else if (const auto* v = std::get_if<bool>(&val))     printf("%s\n",   *v ? "true" : "false");
            else if (const auto* v = std::get_if<std::string>(&val)) printf("\"%s\"\n", v->c_str());
            else printf("[array/other]\n");
        }
        printf("─────────────────────────────────────────────────\n\n");
    }

    if (show_tensors) {
        parser.print_tensors();
    }

    printf("Run with --tensors to see tensor list, --kv to dump all metadata.\n\n");
    printf("Next step: nexus_run --model %s --prompt \"Hello\" --debug\n\n", path.c_str());

    return 0;
}
