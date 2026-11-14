/*
 * nexus/src/api/python.cpp
 *
 * Python bindings via pybind11.
 * Exposes the Transformer engine to Python with a clean, ergonomic API.
 *
 * Usage from Python:
 *
 *   import nexus_py as nexus
 *
 *   model = nexus.Model("/path/to/model.gguf", n_threads=8)
 *   print(model.info())
 *
 *   # Stream tokens
 *   for token in model.generate("Hello, how are you?", max_tokens=200):
 *       print(token, end="", flush=True)
 *
 *   # Or get full response
 *   response = model.generate_str("Explain quantum computing", max_tokens=512)
 *
 *   model.reset()   # clear KV cache
 *   model.memory_report()
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "../model/transformer.h"

namespace py = pybind11;
using namespace nexus::model;

// ─── Python-friendly wrapper ──────────────────────────────────────────────────

class PyModel {
public:
    PyModel(const std::string& path, int n_threads = 1,
            py::object progress_cb = py::none()) {

        std::function<void(const std::string&, float)> cb;
        if (!progress_cb.is_none()) {
            cb = [progress_cb](const std::string& msg, float frac) {
                py::gil_scoped_acquire gil;
                try { progress_cb(msg, frac); } catch (...) {}
            };
        }

        transformer_ = std::make_unique<Transformer>();
        if (!transformer_->load(path, n_threads, cb)) {
            throw std::runtime_error("Failed to load model: " + path);
        }
    }

    // Model info as dict
    py::dict info() const {
        const auto& m = transformer_->meta();
        py::dict d;
        d["arch"]           = m.arch;
        d["name"]           = m.name;
        d["n_layers"]       = m.n_layers;
        d["n_heads"]        = m.n_heads;
        d["n_kv_heads"]     = m.n_kv_heads;
        d["head_dim"]       = m.head_dim;
        d["hidden_dim"]     = m.hidden_dim;
        d["ffn_dim"]        = m.ffn_dim;
        d["vocab_size"]     = m.vocab_size;
        d["context_length"] = m.context_length;
        d["rope_base"]      = m.rope_freq_base;
        return d;
    }

    // Tokenize text → list of token IDs
    std::vector<int32_t> tokenize(const std::string& text, bool add_bos = true) {
        return transformer_->tokenize(text, add_bos);
    }

    // Detokenize ID → string
    std::string detokenize_token(int32_t token_id) {
        return transformer_->detokenize(token_id);
    }

    // Detokenize list → string
    std::string detokenize(const std::vector<int32_t>& tokens) {
        return transformer_->detokenize(tokens);
    }

    // Generate: returns list of token strings (streaming via Python generator)
    py::object generate(
        const std::string& prompt,
        int   max_tokens   = 512,
        float temperature  = 0.7f,
        float top_p        = 0.9f,
        int   top_k        = 50,
        float repeat_penalty = 1.1f,
        const std::vector<std::string>& stop_strings = {}
    ) {
        // Tokenize prompt
        std::vector<int32_t> tokens = transformer_->tokenize(prompt, true);

        SamplerConfig scfg;
        scfg.temperature    = temperature;
        scfg.top_p          = top_p;
        scfg.top_k          = top_k;
        scfg.repeat_penalty = repeat_penalty;

        // Run prompt through model (prefill)
        for (int i = 0; i < (int)tokens.size(); ++i) {
            py::gil_scoped_release no_gil;
            transformer_->forward(tokens, i);
        }

        // Generate tokens
        int pos = (int)tokens.size();
        std::vector<std::string> token_strs;
        std::string accumulated;

        for (int i = 0; i < max_tokens; ++i) {
            int32_t next_tok;
            {
                py::gil_scoped_release no_gil;
                next_tok = transformer_->sample(scfg);
                tokens.push_back(next_tok);
                transformer_->forward(tokens, pos++);
            }

            // EOS?
            if (next_tok == 2 || next_tok == 1) break;  // EOS/BOS token IDs

            std::string tok_str = transformer_->detokenize(next_tok);
            token_strs.push_back(tok_str);
            accumulated += tok_str;

            // Stop string check
            bool stop = false;
            for (const auto& s : stop_strings) {
                if (accumulated.size() >= s.size() &&
                    accumulated.substr(accumulated.size() - s.size()) == s) {
                    // Trim stop string from output
                    token_strs.back() = token_strs.back().substr(
                        0, token_strs.back().size() - s.size());
                    stop = true;
                    break;
                }
            }
            if (stop) break;
        }

        return py::cast(token_strs);
    }

    // Generate complete string (blocking)
    std::string generate_str(
        const std::string& prompt,
        int   max_tokens  = 512,
        float temperature = 0.7f,
        float top_p       = 0.9f,
        int   top_k       = 50,
        const std::vector<std::string>& stop_strings = {}
    ) {
        py::object tok_list = generate(prompt, max_tokens, temperature,
                                        top_p, top_k, 1.1f, stop_strings);
        std::string result;
        for (auto t : tok_list) result += py::cast<std::string>(t);
        return result;
    }

    // Reset KV cache (new conversation)
    void reset() {
        transformer_->reset_context();
    }

    // Print memory report to stderr
    void memory_report() {
        transformer_->print_memory_report();
    }

    // Forward a single token position (low-level, for custom generation loops)
    void forward_token(const std::vector<int32_t>& tokens, int pos) {
        transformer_->forward(tokens, pos);
    }

    // Get logits from last forward pass
    std::vector<float> logits() const {
        // Access via the transformer's logits_ (we need to expose it)
        // For now return empty — expose in next iteration
        return {};
    }

    bool is_loaded() const { return transformer_->is_loaded(); }
    int  vocab_size() const { return transformer_->vocab_size(); }
    int  n_layers()   const { return transformer_->n_layers(); }
    int  context_length() const { return transformer_->context_length(); }

private:
    std::unique_ptr<Transformer> transformer_;
};

// ─── Module definition ────────────────────────────────────────────────────────

PYBIND11_MODULE(nexus_py, m) {
    m.doc() = "Nexus — Open Inference Engine Python bindings";

    py::class_<PyModel>(m, "Model")
        .def(py::init<const std::string&, int, py::object>(),
             py::arg("path"),
             py::arg("n_threads") = 1,
             py::arg("progress_cb") = py::none(),
             R"doc(
Load a GGUF model.

Args:
    path:        Path to .gguf model file
    n_threads:   CPU threads for computation (default 1, use cpu_count-1)
    progress_cb: Optional callback(msg: str, fraction: float) during loading

Raises:
    RuntimeError: if model cannot be loaded
)doc")

        .def("info",      &PyModel::info,
             "Returns model metadata as a dict")

        .def("tokenize",  &PyModel::tokenize,
             py::arg("text"), py::arg("add_bos") = true,
             "Encode text string to list of token IDs")

        .def("detokenize", py::overload_cast<int32_t>(&PyModel::detokenize_token),
             py::arg("token_id"),
             "Decode a single token ID to string")

        .def("detokenize", py::overload_cast<const std::vector<int32_t>&>(&PyModel::detokenize),
             py::arg("tokens"),
             "Decode a list of token IDs to string")

        .def("generate",  &PyModel::generate,
             py::arg("prompt"),
             py::arg("max_tokens")      = 512,
             py::arg("temperature")     = 0.7f,
             py::arg("top_p")           = 0.9f,
             py::arg("top_k")           = 50,
             py::arg("repeat_penalty")  = 1.1f,
             py::arg("stop_strings")    = std::vector<std::string>{},
             R"doc(
Generate tokens from a prompt.

Returns a list of token strings. For streaming, iterate over the list
or use a Python generator wrapper around this.

Args:
    prompt:         Input text
    max_tokens:     Maximum tokens to generate
    temperature:    Sampling temperature (0 = greedy, 1 = normal, 2 = creative)
    top_p:          Nucleus sampling threshold (0.9 = default)
    top_k:          Top-K sampling (0 = disabled)
    repeat_penalty: Penalize repeated tokens (1.0 = no penalty)
    stop_strings:   Stop generation when any of these strings appear
)doc")

        .def("generate_str", &PyModel::generate_str,
             py::arg("prompt"),
             py::arg("max_tokens")  = 512,
             py::arg("temperature") = 0.7f,
             py::arg("top_p")       = 0.9f,
             py::arg("top_k")       = 50,
             py::arg("stop_strings") = std::vector<std::string>{},
             "Generate and return complete response string")

        .def("reset",         &PyModel::reset,
             "Clear KV cache and start a new conversation")

        .def("memory_report", &PyModel::memory_report,
             "Print memory usage report to stderr")

        .def("forward_token", &PyModel::forward_token,
             py::arg("tokens"), py::arg("pos"),
             "Low-level: run one forward pass step")

        .def("is_loaded",      &PyModel::is_loaded)
        .def("vocab_size",     &PyModel::vocab_size)
        .def("n_layers",       &PyModel::n_layers)
        .def("context_length", &PyModel::context_length)

        .def("__repr__", [](const PyModel& m) {
            return "<nexus.Model layers=" + std::to_string(m.n_layers()) +
                   " vocab=" + std::to_string(m.vocab_size()) + ">";
        });

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("__author__")  = "Nexus Project";
}
