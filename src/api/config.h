/*
 * nexus/src/api/config.h
 *
 * Simple INI-style config file parser.
 * Reads nexus.conf and provides typed getters.
 * CLI flags always take precedence — config is just defaults.
 *
 * Format:
 *   [section]
 *   key = value    # comments after values are stripped
 *   # full-line comments
 *   ; also comments
 */

#pragma once
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdlib>      // getenv
#include <algorithm>

namespace nexus {

class Config {
public:
    // Load from file. Returns true if file was found and parsed.
    bool load(const std::string& path) {
        std::string expanded = expand_home(path);
        std::ifstream f(expanded);
        if (!f.is_open()) return false;

        std::string section;
        std::string line;
        while (std::getline(f, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;

            if (line[0] == '[') {
                auto end = line.find(']');
                if (end != std::string::npos)
                    section = trim(line.substr(1, end - 1));
                continue;
            }

            auto eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key   = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));

            // Strip inline comments
            for (char c : {'#', ';'}) {
                auto pos = value.find(c);
                if (pos != std::string::npos)
                    value = trim(value.substr(0, pos));
            }

            std::string full_key = section.empty() ? key : section + "." + key;
            data_[full_key] = value;
        }
        loaded_ = true;
        return true;
    }

    // Try standard locations: ./nexus.conf, ~/.nexus.conf, ~/.config/nexus/nexus.conf
    bool load_default() {
        const char* home = getenv("HOME");
        std::vector<std::string> candidates = {
            "./nexus.conf",
        };
        if (home) {
            candidates.push_back(std::string(home) + "/.nexus.conf");
            candidates.push_back(std::string(home) + "/.config/nexus/nexus.conf");
        }
        for (const auto& p : candidates) {
            if (load(p)) {
                loaded_path_ = p;
                return true;
            }
        }
        return false;
    }

    bool loaded() const { return loaded_; }
    const std::string& loaded_path() const { return loaded_path_; }

    // ── Typed getters (section.key or just key) ──────────────────────────────

    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = data_.find(key);
        if (it == data_.end()) return def;
        return expand_home(it->second);
    }

    int get_int(const std::string& key, int def = 0) const {
        auto it = data_.find(key);
        if (it == data_.end()) return def;
        try { return std::stoi(it->second); } catch (...) { return def; }
    }

    float get_float(const std::string& key, float def = 0.0f) const {
        auto it = data_.find(key);
        if (it == data_.end()) return def;
        try { return std::stof(it->second); } catch (...) { return def; }
    }

    bool get_bool(const std::string& key, bool def = false) const {
        auto it = data_.find(key);
        if (it == data_.end()) return def;
        const std::string& v = it->second;
        return v == "true" || v == "yes" || v == "1" || v == "on";
    }

    // Check if key exists
    bool has(const std::string& key) const {
        return data_.count(key) > 0;
    }

    // Dump all loaded keys (useful for --show-config)
    void print() const {
        if (!loaded_) { printf("  (no config file loaded)\n"); return; }
        printf("  Config: %s\n", loaded_path_.c_str());
        for (const auto& [k, v] : data_)
            printf("  %-40s = %s\n", k.c_str(), v.c_str());
    }

private:
    std::unordered_map<std::string, std::string> data_;
    bool        loaded_      = false;
    std::string loaded_path_;

    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

    static std::string expand_home(const std::string& s) {
        if (s.empty() || s[0] != '~') return s;
        const char* home = getenv("HOME");
        if (!home) return s;
        return std::string(home) + s.substr(1);
    }
};

} // namespace nexus
