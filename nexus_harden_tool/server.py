#!/usr/bin/env python3
"""
Nexus Code-Hardening Companion -- standalone app, talks to running Nexus server(s) over HTTP.

Pipeline:
  1. Scan a project folder for source files and get an EXACT token count for each from the chunk
     model's own tokenizer (/api/tokenize) -- never an estimate.
  2. Files that alone exceed the per-chunk cap are flagged BEFORE chunking. For each one you choose:
       exclude | split into line-range parts | run whole on a large-context model | force it on the
       chunk model as its own over-cap chunk.
  3. Remaining files are packed into chunks of whole files, each at or under the cap.
  4. Each chunk runs an editable chain of prompts (one or many) as successive turns of one
     conversation. Every step's answer is kept and streamed live to the UI.
     Chunk-model chunks run first, then the tool switches to the large-context model (starting it
     itself if you gave a launch command) for the large ones.
  5. Optional final pass: the chunk results are packed and run through a separate prompt chain,
     on its own model if you want (e.g. 9B for the sweep, 27B for the master report).

Every step's answer is written to disk the moment it finishes. Snapshots ("saves") of a project's
results can be taken at any time (and automatically when a run finishes) and browsed later.

Models: each phase (chunk / large-context / final) points at a Nexus server URL plus a model id
("1", or "2" for a server started with --model2). An optional launch command lets this tool start
that server itself, wait for it to come up, and shut it down again when switching to another model
(so a big model and a long-context model never have to fit in RAM at the same time).

Stopping: Nexus deliberately keeps generating after a client disconnects (so replies can be
resumed), so "stop now", "stop & skip this chunk" and the per-answer output limit all call the
server's /api/stop -- just dropping the connection would leave it generating, and the next chunk
would queue behind it.

Usage:
    python3 server.py [--port 8787]
Then open http://127.0.0.1:8787/ in a browser.
"""
import json, os, re, sys, time, math, bisect, threading, hashlib, traceback, subprocess, signal, atexit, shlex, struct
import http.client, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
DATA_ROOT = os.path.join(HERE, "harden_data")   # one subfolder per project: state.json, saves/, model logs
os.makedirs(DATA_ROOT, exist_ok=True)

DEFAULT_EXTENSIONS = [
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hh", ".inc",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".php", ".cs", ".swift", ".m", ".mm", ".sh", ".sql", ".proto",
    ".html", ".css", ".yaml", ".yml", ".json", ".toml", ".cmake", ".txt",
]
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "third_party", "build",
    "dist", "__pycache__", ".venv", "venv", "target", "out", ".idea", ".vscode",
    "harden_data",   # this tool's own saved state -- never scan it (matters if the tool lives inside the scanned tree)
}
RESERVE_TOKENS = 4096        # context headroom kept free for the answer (or max output, if larger)
ROUTE_LABEL = {"main": "chunk model", "large": "large-context model", "final": "final-pass model"}

class NexusError(Exception):
    pass

class Cancelled(Exception):
    pass

# --------------------------------------------------------------------------------------------
# Nexus HTTP client
# --------------------------------------------------------------------------------------------
def m_url(cfg):
    if cfg.get("source") == "file":
        return f"http://127.0.0.1:{int(cfg.get('port') or 0)}" if cfg.get("file") else ""
    u = (cfg.get("url") or "").strip().rstrip("/")
    if u and "://" not in u:
        u = "http://" + u                          # "127.0.0.1:8080" typed without http://
    # "localhost" can resolve to IPv6 (::1) first on Windows while Nexus listens on 127.0.0.1 only:
    # every request would wait for the IPv6 attempt to fail (and short probes time out).
    return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", u, flags=re.I)
def m_id(cfg):   return "1" if cfg.get("source") == "file" else (str(cfg.get("model") or "1").strip() or "1")
def m_key(cfg):  return (m_url(cfg), launch_spec(cfg)[0])

# Sampling: sent with every request (works for any server, including one started in the launcher)
# and also given as the server's own defaults when this tool starts the model. Names = launcher's.
DEFAULT_SAMPLING = {"temp": 0.2, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "tfs_z": 1.0,
                    "rep": 1.1, "rep_n": 128, "freq": 0.0, "pres": 0.0}
SAMPLING_FIELDS = (  # our key, /api/chat field, server CLI flag, launcher settings key, type
    ("temp", "temperature", "--temperature", "temp", float), ("top_p", "top_p", "--top-p", "topp", float),
    ("top_k", "top_k", "--top-k", "topk", int), ("min_p", "min_p", "--min-p", "minp", float),
    ("tfs_z", "tfs_z", "--tfs-z", "tfsz", float), ("rep", "repeat_penalty", "--rep-penalty", "rep", float),
    ("rep_n", "repeat_last_n", "--repeat-last-n", "repn", int), ("freq", "frequency_penalty", "--freq-penalty", "freq", float),
    ("pres", "presence_penalty", "--presence-penalty", "pres", float))

def sampling_values(cfg):
    out = {}
    s = cfg.get("sampling") or {}
    for k, req, flag, lk, typ in SAMPLING_FIELDS:
        v = s.get(k)
        if v in (None, ""):
            continue
        try:
            out[k] = typ(float(v)) if typ is int else typ(v)
        except (TypeError, ValueError):
            pass
    return out

# --------------------------------------------------------------------------------------------
# Finding Nexus: when this tool sits next to (or inside a folder next to) the launcher, it uses
# the same server program, the same model folders, the launcher's saved per-model settings and
# the server's own hardware probe for the graphics-card list -- exactly what the launcher does.
# --------------------------------------------------------------------------------------------
GLOBAL_PATH = os.path.join(DATA_ROOT, "global.json")
SERVER_NAMES = ("nexus_server.exe", "nexus_server") if os.name == "nt" else ("nexus_server",)
MODEL_EXTS = (".gguf", ".nxm")
NEXUS = {"dir": None, "server": None, "probe": None, "probe_error": "", "probed_for": None}
_nexus_lock = threading.Lock()

def load_global():
    try:
        with open(GLOBAL_PATH, "r", encoding="utf-8") as f:
            v = json.load(f)
            return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}

def save_global(g):
    with open(GLOBAL_PATH + ".tmp", "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1)
    os.replace(GLOBAL_PATH + ".tmp", GLOBAL_PATH)

def server_in(d):
    """Same places the launcher looks: right in the folder, or in a 'server' folder beside it."""
    for sub in ("", "server"):
        for n in SERVER_NAMES:
            p = os.path.join(d, sub, n)
            if os.path.isfile(p):
                return p
    return None

def nexus_candidates():
    g = load_global()
    out = [g["nexus_dir"]] if g.get("nexus_dir") else []
    d = HERE
    for _ in range(3):                       # this folder, its parent, and one more up (e.g. <nexus>/accessories/harden)
        out.append(d)
        d = os.path.dirname(d)
    return out

def find_nexus():
    for c in nexus_candidates():
        if c and os.path.isdir(c):
            p = server_in(c)
            if p:
                return os.path.abspath(c), p
    return None, None

def launcher_cfg(d):
    for p in (os.path.join(d, "nexus-data", "nexus_launcher.json"), os.path.join(d, "nexus_launcher.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                v = json.load(f)
                if isinstance(v, dict):
                    return v
        except (OSError, ValueError):
            continue
    return {}

def scan_model_files(d, lcfg):
    """The launcher's model list: folders you added in the launcher first, then <nexus>/models and
    <nexus>/../models, two folder levels deep, .gguf and .nxm (vision encoders left out)."""
    roots = [x for x in ((lcfg.get("global") or {}).get("model_dirs") or []) if isinstance(x, str)]
    roots += [os.path.join(d, "models"), os.path.join(d, "..", "models")]
    seen, out = set(), []
    for r in roots:
        r = os.path.abspath(r)
        if not os.path.isdir(r) or r in seen:
            continue
        seen.add(r)
        base = r.rstrip(os.sep).count(os.sep)
        found = []
        for dp, dns, fns in os.walk(r):
            if dp.rstrip(os.sep).count(os.sep) - base >= 2:
                dns[:] = []
            dns[:] = [x for x in dns if not x.startswith(".") and not x.lower().endswith(".nxmodel")]
            for fn in fns:
                if fn.lower().endswith(MODEL_EXTS) and "mmproj" not in fn.lower():
                    p = os.path.join(dp, fn)
                    try:
                        found.append({"path": p, "name": fn, "dir": dp, "size": os.path.getsize(p)})
                    except OSError:
                        pass
        out += sorted(found, key=lambda m: (m["dir"].lower(), m["name"].lower()))
    return out

def run_probe(server, d):
    try:
        r = subprocess.run([server, "--probe"], cwd=d, capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"hardware probe failed: {e}"
    out = r.stdout.decode("utf-8", "replace")
    i, j = out.find("{"), out.rfind("}")
    if i < 0 or j < i:
        return None, "the hardware probe printed no JSON"
    try:
        return json.loads(out[i:j + 1]), ""
    except ValueError as e:
        return None, f"could not read the hardware probe: {e}"

def gpu_list(probe):
    """Cards from the server's own probe, named for --gpu-device the way the launcher does it
    (name:<card name>#<n>, which stays right even if device numbering changes)."""
    out, count = [], {}
    devs = (((probe or {}).get("vulkan") or {}).get("devices")) or []
    for x in devs:
        idx, nm = str(x.get("index", "")), str(x.get("name") or "")
        arg = idx
        if nm:
            count[nm.lower()] = count.get(nm.lower(), 0) + 1
            arg = f"name:{nm}#{count[nm.lower()]}"
        vb = x.get("vram_bytes")
        out.append({"value": idx, "name": nm, "arg": arg,
                    "vram_mb": int(vb / 1048576) if isinstance(vb, (int, float)) else None,
                    "usable": bool(x.get("usable")), "reason": str(x.get("reason") or ""),
                    "integrated": "discrete" not in str(x.get("type") or "").lower()})
    return out

def nexus_refresh(force_probe=False):
    with _nexus_lock:
        d, srv = find_nexus()
        NEXUS["dir"], NEXUS["server"] = d, srv
        if srv and (force_probe or NEXUS["probed_for"] != srv):
            NEXUS["probe"], NEXUS["probe_error"] = run_probe(srv, d)
            NEXUS["probed_for"] = srv
        if not srv:
            NEXUS["probe"], NEXUS["probe_error"], NEXUS["probed_for"] = None, "", None
        return d, srv

def physical_cores():
    cpu = (NEXUS["probe"] or {}).get("cpu") or {}
    for k in ("physical", "logical"):
        if isinstance(cpu.get(k), (int, float)) and cpu[k] > 0:
            return int(cpu[k])
    return os.cpu_count() or 4

def nexus_view(force_probe=False):
    d, srv = nexus_refresh(force_probe)
    lcfg = launcher_cfg(d) if d else {}
    models = scan_model_files(d, lcfg) if d else []
    saved = lcfg.get("models") or {}
    for m in models:
        sv = saved.get(m["name"])
        if isinstance(sv, dict):
            m["launcher"] = {"sampling": {k: sv[lk] for k, req, flag, lk, typ in SAMPLING_FIELDS if str(sv.get(lk, "")).strip()},
                             "chat_format": str(sv.get("chatFmt") or ""),
                             "rope": str(sv.get("rope") or ""), "norm": str(sv.get("norm") or "")}
    roster = []
    if d:
        try:
            with open(os.path.join(d, "nexus-data", "nexus-roster.json"), "r", encoding="utf-8") as f:
                roster = [r for r in json.load(f) if isinstance(r, dict)]
        except (OSError, ValueError, TypeError):
            roster = []
    return {"found": bool(srv), "dir": d, "server": srv, "models": models, "gpus": gpu_list(NEXUS["probe"]),
            "cores": physical_cores(), "roster": roster, "probe_error": NEXUS["probe_error"],
            "looked_in": [c for c in nexus_candidates() if c], "here": HERE}

def gguf_layers(path):
    """Layer count from a GGUF header (<arch>.block_count), so 'full GPU' asks for exactly the
    model's layers like the launcher does. None if it can't be read (e.g. .nxm)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            ver, = struct.unpack("<I", f.read(4))
            if ver < 2:
                return None
            _, n_kv = struct.unpack("<QQ", f.read(16))

            def rstr():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n)

            def skip(t):
                if t in sizes:
                    f.seek(sizes[t], 1)
                elif t == 8:
                    n, = struct.unpack("<Q", f.read(8)); f.seek(n, 1)
                elif t == 9:
                    at, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
                    if at in sizes:
                        f.seek(sizes[at] * n, 1)
                    else:
                        for _ in range(n):
                            skip(at)
                else:
                    raise ValueError("bad gguf type")

            for _ in range(min(n_kv, 4096)):
                key = rstr().decode("utf-8", "replace")
                t, = struct.unpack("<I", f.read(4))
                if key.endswith(".block_count") and t in (4, 5, 10, 11):
                    return int(struct.unpack({4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[t], f.read(sizes[t]))[0])
                skip(t)
    except (OSError, ValueError, struct.error):
        return None
    return None

def launch_spec(cfg):
    """How this tool would start the model: (display text, argv or None, extra env, cwd, error).
    'file' models get the same command line the launcher would build for that choice of card,
    full GPU and prompt acceleration; other models use their raw launch command (run by the shell)."""
    if cfg.get("source") != "file":
        return (cfg.get("launch") or "").strip(), None, None, None, ""
    d, srv = NEXUS["dir"], NEXUS["server"]
    if not srv:
        d, srv = nexus_refresh()
    if not srv:
        return "", None, None, None, ("the Nexus server program wasn't found. Put this tool's folder next to the "
                                      "launcher (or set the Nexus folder in the Models card) so it can find it.")
    path = cfg.get("file") or ""
    if not os.path.isfile(path):
        return "", None, None, None, f"model file not found: {path or '(none picked)'}"
    port = int(cfg.get("port") or 0)
    if not 1 <= port <= 65535:
        return "", None, None, None, "pick a port (1-65535) for this model"
    threads = int(cfg.get("threads") or 0) or physical_cores()
    argv = [srv, "--model", path, "--port", str(port), "--host", "127.0.0.1", "--threads", str(threads)]
    env = {}
    gpus = {g["value"]: g for g in gpu_list(NEXUS["probe"])}
    sel = str(cfg.get("gpu") or "cpu")
    g = gpus.get(sel)
    garg = g["arg"] if g else sel
    full = sel != "cpu" and bool(cfg.get("full_gpu"))
    turbo = sel != "cpu" and not full and bool(cfg.get("turbo"))
    if full:
        n = gguf_layers(path) if path.lower().endswith(".gguf") else None
        argv += ["--gpu-layers", str(n or 999), "--gpu-device", garg]
        if os.name != "nt" and n and not (g and g["integrated"]):
            env["NEXUS_DIRECT_GPU_LAYERS"] = str(n)              # launcher: layers live in VRAM only, no RAM copy
            if g and g.get("vram_mb"):
                env["NEXUS_DIRECT_VRAM_MB"] = str(g["vram_mb"])
    else:
        argv += ["--gpu-layers", "0"]
        if turbo:
            argv += ["--gpu-device", garg, "--turbo-prefill", "--turbo-gpu-stream"]
    if (cfg.get("chat_format") or "").strip():
        argv += ["--chat-format", cfg["chat_format"].strip()]
    # Same overrides as the launcher's "RoPE layout" / "Norm convention" (empty = the model file decides).
    if cfg.get("rope") in ("interleaved", "half_split"):
        env["NEXUS_ROPE_LAYOUT"] = cfg["rope"]
    if cfg.get("norm") in ("direct", "zero_centered"):
        env["NEXUS_NORM_CONVENTION"] = cfg["norm"]
    sv = sampling_values(cfg)
    for k, req, flag, lk, typ in SAMPLING_FIELDS:
        if k in sv:
            argv += [flag, str(sv[k])]
    try:
        argv += shlex.split(cfg.get("extra_args") or "", posix=os.name != "nt")
    except ValueError as e:
        return "", None, None, None, f"extra arguments: {e}"
    shown = " ".join(f"{k}={v}" for k, v in env.items())
    shown = (shown + " " if shown else "") + " ".join(shlex.quote(a) for a in argv)
    return shown, argv, env, d, ""

# Model servers are reached DIRECTLY, never through a system proxy. Python picks up the proxy set in
# Windows' Internet Options / HTTP_PROXY (VPNs, antivirus web shields set one); a browser always skips it
# for 127.0.0.1, Python only if the proxy's bypass list says so -- so the browser could open the model
# while this tool got proxy errors.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# API keys for model servers on other computers that were given one in their launcher (Network card).
# Registered per server address whenever a model config is resolved; sent as "Authorization: Bearer <key>".
API_KEYS = {}

def _register_key(cfg):
    u, k = m_url(cfg), str(cfg.get("api_key") or "").strip()
    if u and cfg.get("source") != "file":
        if k: API_KEYS[u] = k
        else: API_KEYS.pop(u, None)

def _headers(url, extra=None):
    h = {"Content-Type": "application/json"}
    for base, k in API_KEYS.items():
        if url.startswith(base + "/") or url == base:
            h["Authorization"] = "Bearer " + k
            break
    if extra: h.update(extra)
    return h

def _http_err(url, e):
    if e.code == 401:
        return NexusError(f"{url}: that server needs its API key (HTTP 401) -- enter it in the Models card")
    if e.code == 403:
        return NexusError(f"{url}: that server refused this request (HTTP 403): {e.read()[:200]!r}")
    return NexusError(f"{url}: HTTP {e.code} {e.read()[:200]!r}")

def _req(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(url),
                                 method="POST" if body is not None else "GET")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise _http_err(url, e)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise NexusError(f"could not reach {url}: {e}")
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        raise NexusError(f"{url}: not JSON: {raw[:200]!r}")

def nexus_info(cfg, timeout=10):
    if not m_url(cfg):
        raise NexusError("no URL set")
    # Plain /api/info, no query string: Nexus matches the path exactly, so "/api/info?model=1" was an
    # unknown page to it. The reply already carries model 2's fields (model2_*), so nothing is lost.
    return _req(f"{m_url(cfg)}/api/info", timeout=timeout)

def nexus_tokenize(cfg, text):
    j = _req(m_url(cfg) + "/api/tokenize", {"text": text, "add_bos": "false", "model": m_id(cfg)}, timeout=300)
    if not j.get("ok"):
        raise NexusError(f"tokenize failed: {str(j)[:300]}")
    return int(j["count"])

def nexus_context(cfg):
    """Tokens the model holds in memory right now (None if it can't say)."""
    try:
        j = _req(f"{m_url(cfg)}/api/context?model={m_id(cfg)}", timeout=10)
        v = j.get("context_size") if isinstance(j, dict) else None
        return v if isinstance(v, int) else None
    except NexusError:
        return None

def nexus_add_turn(cfg, text, cancel=None, on_stat=None):
    """Adds a user turn to what the model already remembers (it reads only the new text) -- the way the
    Nexus chat page continues a conversation, which also works after a saved memory is loaded."""
    body = {"model": m_id(cfg), "text": text, "turn_id": hashlib.sha1(os.urandom(16)).hexdigest()}
    think = (cfg.get("think") or "").strip()
    if think in ("on", "off"):
        body["think"] = think
    req = urllib.request.Request(m_url(cfg) + "/api/session/extend_turn", data=json.dumps(body).encode("utf-8"),
                                 headers=_headers(m_url(cfg)), method="POST")
    # This request only answers when the model has read the message, so ask the server how far it is
    # meanwhile (GET /api/progress) to show the percentage and speed, like the first prompt of a chain.
    done = threading.Event()
    def poll():
        t0 = time.time()
        while not done.wait(1.0):
            try:
                pg = _req(f"{m_url(cfg)}/api/progress?model={m_id(cfg)}", timeout=5)
            except NexusError:
                continue                                  # older server without /api/progress: no numbers, still works
            if on_stat and isinstance(pg, dict) and pg.get("active"):
                on_stat({"phase": "reading", "pct": pg.get("pct", 0), "prompt_tokens": pg.get("tokens", 0),
                         "prompt_tps": pg.get("tps", 0), "eta": pg.get("eta", -1), "reading_s": round(time.time() - t0)})
    if on_stat:
        threading.Thread(target=poll, daemon=True).start()
    try:
        with _OPENER.open(req, timeout=6 * 3600) as resp:
            raw = resp.read().decode("utf-8", "replace").strip()     # keep-alive spaces, then the JSON
    except urllib.error.HTTPError as e:
        raise _http_err(f"{m_url(cfg)}/api/session/extend_turn", e)
    except (urllib.error.URLError, OSError) as e:
        raise NexusError(f"{m_url(cfg)}/api/session/extend_turn: {e}")
    finally:
        done.set()
    try:
        j = json.loads(raw or "{}")
    except ValueError:
        raise NexusError("the model server gave an unreadable answer while reading the message")
    if not j.get("ok"):
        raise NexusError(j.get("error") or "the model server would not take the message")
    return j

def nexus_session(cfg, action, path):
    j = _req(f"{m_url(cfg)}/api/session/{action}", {"path": path, "model": m_id(cfg)}, timeout=3600)
    if not j.get("ok"):
        raise NexusError(j.get("error") or f"could not {action} the model's memory")
    return j

def nexus_stop(cfg):
    if not cfg or not m_url(cfg):
        return
    try:
        _req(m_url(cfg) + "/api/stop", {"model": m_id(cfg)}, timeout=15)
    except NexusError:
        pass

def ctx_of(info, model="1"):
    """The model's context LENGTH. (Nexus's "context_size" is how much of it is in use right now --
    0 on a freshly started model -- so it must not be read as the size.)"""
    if not isinstance(info, dict):
        return None
    keys = ("model2_context_length",) if str(model) == "2" else ("max_context", "context_length")
    for k in keys + ("context_size",):          # context_size: only for servers that report nothing else
        v = info.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None

def nexus_chat(cfg, messages, conv_id=None, cancel=None, on_token=None, max_out=0, timeout=6 * 3600, on_stat=None,
               endpoint="/api/chat"):
    """Streams one /api/chat reply. Returns {text, notice, conv_id, truncated, cancelled}.

    Reads line by line (resp.readline) so each SSE event surfaces as soon as it arrives; read(N)
    under chunked encoding waits for N bytes and would hide tokens until the reply was over.
    When cancel() turns true or max_out tokens have been generated, /api/stop is sent and the
    stream is drained to its end (tokens after the stop are discarded), so the server is idle
    again before the caller sends the next request."""
    body = {"model": m_id(cfg), "temperature": 0, "turn_id": hashlib.sha1(os.urandom(16)).hexdigest()}
    if endpoint == "/api/chat":
        body["messages"] = messages
    sv = sampling_values(cfg)
    for k, req, flag, lk, typ in SAMPLING_FIELDS:
        if k in sv:
            body[req] = sv[k]
    if conv_id:
        body["conv_id"] = conv_id           # warm-KV prefix reuse: later chain steps skip re-reading the code
    think = (cfg.get("think") or "").strip()
    if think in ("on", "off"):
        body["think"] = think
    req = urllib.request.Request(m_url(cfg) + endpoint, data=json.dumps(body).encode("utf-8"),
                                 headers=_headers(m_url(cfg) + endpoint), method="POST")
    parts, notice = [], None
    gen, base = 0, None
    # speed: prompt reading from the server's progress events, writing measured here from the first token
    stat = {"phase": "waiting", "prompt_tokens": 0, "prompt_tps": 0.0, "out_tps": 0.0}
    t_req, t_first, g_first = time.time(), None, 0
    def push_stat(**kw):
        stat.update(kw)
        if on_stat:
            on_stat(dict(stat))
    stop_sent, deadline = False, 0.0
    truncated = cancelled = False
    try:
        resp = _OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise _http_err(f"{m_url(cfg)}{endpoint}", e)
    except (urllib.error.URLError, OSError) as e:
        raise NexusError(f"could not reach {m_url(cfg)}: {e}")
    with resp:
        while True:
            if not stop_sent:
                if cancel and cancel():
                    cancelled = True
                elif max_out and gen >= max_out:
                    truncated = True
                if cancelled or truncated:
                    nexus_stop(cfg)
                    stop_sent, deadline = True, time.time() + 30
            elif time.time() > deadline:
                break
            try:
                raw = resp.readline()
            except (OSError, http.client.HTTPException) as e:
                if stop_sent:
                    break
                raise NexusError(f"stream from {m_url(cfg)} broke: {e}")
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue                     # blank separators and ": ping" keep-alives
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("conv_id"):
                conv_id = ev["conv_id"]
            if "notice" in ev:
                notice = ev["notice"]        # e.g. memory guard intervened -- surfaced, never hidden
            pg = ev.get("progress")
            if isinstance(pg, dict) and t_first is None:   # the server is still reading the prompt
                push_stat(phase="reading", pct=pg.get("pct", 0), prompt_tokens=pg.get("tokens", 0),
                          prompt_tps=pg.get("tps", 0) or stat["prompt_tps"], eta=pg.get("eta", -1))
            if ev.get("final") and isinstance(ev.get("gen"), int):
                gen = ev["gen"]              # the engine's exact count for the whole reply
            if "token" in ev and not stop_sent:
                tok = ev["token"]
                parts.append(tok)
                if isinstance(ev.get("gen"), int):
                    gen = ev["gen"]          # real tokens from the engine (MTP can commit several per piece)
                else:                        # older server: fall back to its context counter / one per piece
                    ct = ev.get("context_tokens")
                    if isinstance(ct, int):
                        if base is None:
                            base = ct - 1
                        gen = ct - base
                    else:
                        gen += 1
                now = time.time()
                if t_first is None:
                    t_first, g_first = now, gen
                    push_stat(phase="writing", pct=100, eta=-1, first_token_s=round(now - t_req, 1))
                elif now - t_first > 0.5 and gen > g_first:
                    push_stat(out_tps=round((gen - g_first) / (now - t_first), 1), gen=gen)
                if on_token:
                    on_token(tok, gen)
    if not stop_sent and cancel and cancel():
        # The stop landed while we were blocked waiting for the next line, and the server ended
        # the stream before we looked again -- it was still a stop, not a natural finish.
        cancelled = True
    return {"text": "".join(parts), "notice": notice, "conv_id": conv_id,
            "truncated": truncated, "cancelled": cancelled, "gen": gen, "stat": stat}

# --------------------------------------------------------------------------------------------
# Model processes this tool started itself (from a launch command)
# --------------------------------------------------------------------------------------------
MANAGED = {}                  # (url, launch) -> {"proc", "log", "root", "launch", "url"}
_proc_lock = threading.Lock()

def load_progress(path):
    """Last model-loading line of a server log this tool started, e.g. '45%  Loading weights into memory'."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 16384))
            tail = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return ""
    for line in reversed(tail):
        line = line.strip()
        if line.startswith("[load]"):
            return line[6:].strip()[:120]
    return ""

def managed_alive(url):
    """A model process this tool started on this URL that is still running (whatever its settings)."""
    for e in list(MANAGED.values()):
        if e["url"] == url and e["proc"].poll() is None:
            return e
    return None

def start_process(root, key, exclusive=False, argv=None, env=None, cwd=None):
    url, launch = key
    logp = os.path.join(project_dir(root), "model_" + hashlib.sha1(launch.encode()).hexdigest()[:8] + ".log")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with open(logp, "ab") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {launch}\n".encode())
        lf.flush()
        if argv:
            proc = subprocess.Popen(argv, cwd=cwd or None, env=dict(os.environ, **(env or {})), stdout=lf,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
        else:
            proc = subprocess.Popen(launch, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, **kw)
    MANAGED[key] = {"proc": proc, "log": logp, "root": root, "launch": launch, "url": url, "exclusive": bool(exclusive)}
    return MANAGED[key]

def stop_process(key):
    ent = MANAGED.pop(key, None)
    if not ent:
        return
    p = ent["proc"]
    if p.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGTERM)    # whole process group: the shell AND the server it started
    except Exception:
        pass
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                p.kill()
            else:
                os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass

def stop_all_managed(root=None):
    with _proc_lock:
        for k in [k for k, e in MANAGED.items() if root is None or e["root"] == root]:
            stop_process(k)

atexit.register(stop_all_managed)

def route_status(st, root):
    """For each phase: is the model this tool would use for it running as a process this tool started?"""
    out = {}
    for r in ("main", "large", "final"):
        try:
            e = MANAGED.get(m_key(mcfg(st, r)))
        except Exception:
            e = None
        if e and e["root"] == root:
            out[r] = {"alive": e["proc"].poll() is None, "pid": e["proc"].pid, "log": e["log"]}
    return out

def managed_view(root):
    out = []
    for (url, launch), e in MANAGED.items():
        if e["root"] == root:
            out.append({"url": url, "launch": launch, "pid": e["proc"].pid,
                        "alive": e["proc"].poll() is None, "log": e["log"]})
    return out

# --------------------------------------------------------------------------------------------
# Live output buffer (what's streaming right now), polled by the UI. Keyed by project root.
# --------------------------------------------------------------------------------------------
LIVE = {}
_live_lock = threading.Lock()

def live_set(root, **kw):
    with _live_lock:
        LIVE.setdefault(root, {"answers": [], "text": "", "gen": 0}).update(kw)

def live_token(root, tok, gen):
    with _live_lock:
        d = LIVE.setdefault(root, {"answers": [], "text": "", "gen": 0})
        d["text"] = d.get("text", "") + tok
        d["gen"] = gen

def live_stat(root, stat):
    with _live_lock:
        LIVE.setdefault(root, {"answers": [], "text": "", "gen": 0})["stat"] = stat

def live_push_answer(root, text):
    with _live_lock:
        d = LIVE.setdefault(root, {"answers": [], "text": "", "gen": 0})
        d.setdefault("answers", []).append(text)
        d["text"], d["gen"] = "", 0
        d.pop("stat", None)

def live_clear(root):
    with _live_lock:
        LIVE.pop(root, None)

# --------------------------------------------------------------------------------------------
# Project state: ONE in-memory object per project, shared by the HTTP handlers and the run thread
# (so stopping, reordering, deleting and resetting work while a run is going), always mutated
# under _lock, written to disk after every change that matters.
# --------------------------------------------------------------------------------------------
_lock = threading.RLock()
STATES = {}

def project_key(root):
    return hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:16]

def project_dir(root):
    d = os.path.join(DATA_ROOT, project_key(root))
    os.makedirs(d, exist_ok=True)
    return d

def state_path(root):
    return os.path.join(project_dir(root), "state.json")

DEFAULT_CHUNK_PROMPTS = [
    "Find security vulnerabilities in the code above that could lead to code execution, privilege "
    "escalation, or data exposure. For each one, give the exact file and function, quote the relevant "
    "lines, and explain the concrete attacker input or sequence of calls that would trigger it. If you "
    "find nothing, say so plainly rather than inventing a finding.",
    "Review your own findings above. Which of them are you least sure about, and why? Drop anything "
    "that was really just a style issue or a theoretical concern with no real trigger path. Keep only "
    "what you would stand behind as a real, exploitable issue.",
    "For what remains: could any of these, alone or combined, actually be used to gain code execution "
    "or shell access? Walk through the worst realistic case step by step. If nothing here rises to "
    "that level, say that clearly instead of overstating it.",
]
DEFAULT_FINAL_PROMPTS = [
    "Below are the findings from every section of this codebase, reviewed independently. "
    "Identify findings from different sections that interact -- where one section's issue becomes "
    "worse, or only becomes exploitable, in combination with another's. List concrete combinations, "
    "not vague overlaps.",
    "Rank every distinct issue (from a single section or a combination) by how bad it would be if "
    "exploited and how easy it would be to trigger, highest risk first. For each, name the file(s) "
    "and give one sentence on what you'd fix first.",
]

# Settings that carry over to NEW projects: every "Save settings" / "Save prompts" also stores them
# here (global.json "defaults"), so a new project starts the way the last one was set up.
DEFAULTS_KEYS = ("cap_tokens", "large_cap_tokens", "extensions", "exclude_dirs", "models", "auto_final", "unload_on_switch",
                 "unload_at_end", "keep_final_loaded", "autosave", "chunk_prompts", "final_prompts")

def saved_defaults():
    d = load_global().get("defaults")
    return d if isinstance(d, dict) else {}

def store_defaults(st):
    g = load_global()
    g["defaults"] = json.loads(json.dumps({k: st[k] for k in DEFAULTS_KEYS if k in st}))
    save_global(g)

def apply_defaults(st, d):
    """Copy saved defaults into a project state (models merged field by field, so newer fields keep their built-in values)."""
    d = json.loads(json.dumps(d))
    for k in DEFAULTS_KEYS:
        if k not in d:
            continue
        if k == "models" and isinstance(d[k], dict):
            for route, m in d[k].items():
                if route in st["models"] and isinstance(m, dict):
                    st["models"][route].update(m)
        else:
            st[k] = d[k]

def default_models():
    def blank(port):
        return {"source": "url", "url": "", "model": "1", "launch": "", "file": "", "port": port, "gpu": "cpu",
                "full_gpu": False, "turbo": False, "threads": 0, "chat_format": "", "extra_args": "", "rope": "", "norm": "",
                "sampling": dict(DEFAULT_SAMPLING), "max_out": 0, "think": "", "exclusive": False}
    return {"main": blank(8091), "large": blank(8092), "final": blank(8093)}

def default_state(root):
    return {
        "root": os.path.abspath(root),
        "cap_tokens": 150000,
        "large_cap_tokens": 250000,  # part size when an oversized file is split for the LARGE-context model
        "extensions": list(DEFAULT_EXTENSIONS),
        "exclude_dirs": sorted(DEFAULT_EXCLUDE_DIRS),
        "models": default_models(),
        "auto_final": False,        # run the final pass automatically when the chunks finish
        "unload_on_switch": True,   # stop a model this tool started before starting a different one
        "unload_at_end": True,      # stop models this tool started when a run finishes
        "keep_final_loaded": True,  # ...except the final-pass model, so you can ask it follow-up questions
        "autosave": True,           # take a snapshot automatically when a run finishes
        "chunk_prompts": list(DEFAULT_CHUNK_PROMPTS),
        "final_prompts": list(DEFAULT_FINAL_PROMPTS),
        "files": [],
        "oversized": [],            # files over the cap, each with decision: exclude|split|large|force
        "chunks": [],
        "final_chunks": [],
        "run_log": [],
        "running": False,
        "run_mode": "",
        "phase_msg": "",
    }

def _migrate(raw, root):
    st = default_state(root)
    st.update(raw)
    models = default_models()
    saved = raw.get("models") or {}
    for k in models:
        models[k].update(saved.get(k) or {})
    if "models" not in raw and raw.get("nexus_url"):
        models["main"]["url"] = raw["nexus_url"]          # older single-URL state
    st["models"] = models
    for k in ("nexus_url", "final_running", "queue_built"):
        st.pop(k, None)
    if "harden_data" not in st["exclude_dirs"]:
        st["exclude_dirs"] = sorted(set(st["exclude_dirs"]) | {"harden_data"})
    for o in st["oversized"]:
        if o.get("decision") == "force_include":
            o["decision"] = "force"
        o.setdefault("decision", "exclude")
    for lst, route in (("chunks", "main"), ("final_chunks", "final")):
        for c in st[lst]:
            if "steps" not in c:
                answers = c.pop("all_answers", None) or ([c["result"]] if c.get("result") else [])
                c["steps"] = [{"prompt": "", "text": t} for t in answers]
            c.setdefault("route", route)
            c.setdefault("kind", "forced" if c.get("forced") else ("final" if lst == "final_chunks" else "files"))
            if c.get("status") == "running":
                c["status"] = "pending"                    # left over from a crash / restart
    st["running"], st["phase_msg"], st["run_mode"] = False, "", ""
    return st

def get_state(root):
    root = os.path.abspath(root)
    with _lock:
        st = STATES.get(root)
        if st is None:
            p = state_path(root)
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    st = _migrate(json.load(f), root)
            else:
                st = default_state(root)
                apply_defaults(st, saved_defaults())       # a new project starts from the last saved setup
            STATES[root] = st
        return st

def save_state(st):
    with _lock:
        p = state_path(st["root"])
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, p)

def log(st, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _lock:
        st["run_log"].append(line)
        st["run_log"] = st["run_log"][-800:]
    print(line, flush=True)

def set_phase(st, msg):
    with _lock:
        st["phase_msg"] = msg

def mcfg(st, route):
    """Resolved model config for a phase. An empty final-pass URL means 'same as the chunk model'
    (keeping the final pass's own output limit and think setting)."""
    m = dict(st["models"].get(route) or {})
    if route == "final" and m.get("source") != "file" and not m_url(m):
        base = dict(st["models"]["main"])
        base["max_out"], base["think"] = m.get("max_out", 0), m.get("think", "")
        _register_key(base)
        return base
    _register_key(m)
    return m

# --------------------------------------------------------------------------------------------
# Scanning, exact token counts, packing, splitting
# --------------------------------------------------------------------------------------------
TOKEN_CACHE = {}   # (path, mtime, size, url, model) -> tokens

def scan_files(st):
    root, exts, excl = st["root"], set(st["extensions"]), set(st["exclude_dirs"])
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excl and not d.startswith(".")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            full = os.path.join(dirpath, fn)
            try:
                sz, mt = os.path.getsize(full), os.path.getmtime(full)
            except OSError:
                continue
            if sz > 8 * 1024 * 1024:   # 8 MB: not hand-written source; skip rather than hang tokenizing it
                continue
            found.append((os.path.relpath(full, root), full, sz, mt))
    return sorted(found)

def read_file_text(root, rel):
    with open(os.path.join(root, rel), "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def pack_chunks(items, cap):
    groups, cur, cur_tok = [], [], 0
    for it in items:
        if cur and cur_tok + it["tokens"] > cap:
            groups.append(cur); cur, cur_tok = [], 0
        cur.append(it); cur_tok += it["tokens"]
    if cur:
        groups.append(cur)
    return groups

def split_file(root, cfg, path, cap):
    """Splits one file into consecutive line ranges, each measured with the real tokenizer and at
    or under the cap (a single line longer than the cap is the only exception -- it can't be cut)."""
    text = read_file_text(root, path)
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    total = nexus_tokenize(cfg, text)
    n = max(2, math.ceil(total / max(1, int(cap * 0.85))))
    cum = [0]
    for ln in lines:
        cum.append(cum[-1] + len(ln))
    bounds = [0]
    for k in range(1, n):
        i = bisect.bisect_left(cum, cum[-1] * k / n)
        if bounds[-1] < i < len(lines):
            bounds.append(i)
    bounds.append(len(lines))
    out = []

    def handle(a, b):
        t = nexus_tokenize(cfg, "".join(lines[a:b]))
        if t > cap and b - a > 1:
            mid = bisect.bisect_left(cum, (cum[a] + cum[b]) / 2)
            mid = min(max(mid, a + 1), b - 1)
            handle(a, mid); handle(mid, b)
        else:
            out.append((a, b, t))

    for a, b in zip(bounds, bounds[1:]):
        if b > a:
            handle(a, b)
    return [{"path": path, "tokens": t, "lines": [a + 1, b], "part": i + 1, "parts": len(out)}
            for i, (a, b, t) in enumerate(out)]

def member_text(root, m):
    text = read_file_text(root, m["path"])
    if m.get("lines"):
        a, b = m["lines"]
        return "".join(text.splitlines(keepends=True)[a - 1:b])
    return text

def build_chunk_text(root, item):
    parts = []
    for m in item["members"]:
        hdr = f"=== FILE: {m['path']}"
        if m.get("lines"):
            hdr += f" (part {m['part']}/{m['parts']}, lines {m['lines'][0]}-{m['lines'][1]})"
        parts.append(f"{hdr} ({m['tokens']} tokens) ===\n{member_text(root, m)}")
    return "\n\n".join(parts)

def build_final_text(item):
    return "\n\n".join(f"=== RESULT OF {m['chunk_id']} (files: {', '.join(m['files'])}) ===\n{m['text']}"
                       for m in item["members"])

# --------------------------------------------------------------------------------------------
# Runs: one worker thread per project at a time
# --------------------------------------------------------------------------------------------
class RunCtl:
    def __init__(self):
        self.cancel_now = False     # stop immediately (current answer is stopped mid-output)
        self.stop_after = False     # finish the current chunk, then stop
        self.skip_ids = set()       # "stop & skip" requested for these items
        self.current_id = None
        self.cfg_active = None
        self.finished = 0
        self.ran_final = False      # a final pass ran in this run (on its own, or automatically after the chunks)

RUNS = {}

def runnable(c):
    return c["status"] in ("pending", "error") or (c["status"] == "stopped" and c.get("stop_reason") == "run")

def ensure_model(root, st, route, ctl):
    """Makes sure the model for this phase is up, then returns (cfg, info).
    Before that it stops other models this tool started when any of these is true:
      - this model is marked "run alone" (exclusive) -- e.g. a 27B that needs all the RAM,
      - the other model was marked "run alone" (it was hogging the RAM, so make room),
      - "swap models one at a time" (unload_on_switch) is on,
      - the other one uses the same URL (same port -- they can't both run).
    Otherwise models stay loaded side by side, so two small models can both stay resident."""
    cfg = mcfg(st, route)
    label = ROUTE_LABEL[route]
    shown, argv, env, cwd, err = launch_spec(cfg)
    if err:
        raise NexusError(f"{label}: {err}")
    if not m_url(cfg):
        raise NexusError(f"no {label} configured -- set it up in the Models card")
    key = (m_url(cfg), shown)
    with _proc_lock:
        for k in list(MANAGED):
            e = MANAGED[k]
            if e["root"] == root and k != key and (cfg.get("exclusive") or e.get("exclusive")
                                                    or st.get("unload_on_switch", True) or k[0] == key[0]):
                log(st, f"stopping model this tool started: {e['launch']}")
                stop_process(k)

    def probe():
        try:
            return nexus_info(cfg, timeout=5)
        except NexusError:
            return None

    info = probe()
    if info is not None:
        mine = managed_alive(m_url(cfg))
        if mine and mine["launch"] != shown:
            log(st, f"note: the {label} was started with different settings; they apply after it is restarted (Stop, then Start)")
        if cfg.get("source") == "file" and not mine:
            raise NexusError(f"port {cfg.get('port')} is already in use by another Nexus server (maybe one started in "
                             f"the launcher), so the {label} can't start there. Pick a different port for it, or stop that one.")
        return cfg, info
    ent = None
    if key[1]:
        with _proc_lock:
            ent = managed_alive(key[0])                # already starting (maybe with older settings): wait for that one
            if not ent:
                log(st, f"starting {label}: {key[1]}")
                ent = start_process(root, key, cfg.get("exclusive"), argv, env, cwd)
        set_phase(st, f"Starting the {label} ({m_url(cfg)}) and waiting for it to load...")
    else:
        set_phase(st, f"Waiting for the {label} at {m_url(cfg)} -- start it in the launcher, "
                      f"or give it a launch command so this tool can start it.")
        log(st, f"waiting for {label} at {m_url(cfg)}")
    while True:
        if ctl.cancel_now or ctl.stop_after:
            set_phase(st, "")
            raise Cancelled()
        if ent is not None and ent["proc"].poll() is not None:
            set_phase(st, "")
            raise NexusError(f"the {label} process exited (code {ent['proc'].returncode}) before it came up "
                             f"-- see {ent['log']}")
        time.sleep(2)
        if ent is not None:
            lp = load_progress(ent["log"])
            if lp:
                set_phase(st, f"Starting the {label} ({m_url(cfg)}): {lp}")
        info = probe()
        if info is not None:
            set_phase(st, "")
            log(st, f"{label} is up ({m_url(cfg)} model {m_id(cfg)}, context {ctx_of(info, m_id(cfg)) or '?'})")
            return cfg, info

def run_item(root, st, ctl, list_key, item, route, cfg, info):
    kind = "final" if list_key == "final_chunks" else "chunk"
    prompts = list(st["final_prompts"] if kind == "final" else st["chunk_prompts"])
    max_out = int(cfg.get("max_out") or 0)
    ctl.current_id, ctl.cfg_active = item["id"], cfg
    name = (info.get("name") or info.get("model") or "") if isinstance(info, dict) else ""
    with _lock:
        item["model"] = f"{ROUTE_LABEL[route]}: {name or m_url(cfg)}" + (f" (model {m_id(cfg)})" if m_id(cfg) != "1" else "")
    live_set(root, chunk_id=item["id"], kind=kind, answers=[], text="", gen=0,
             prompt_index=0, prompt_total=len(prompts), max_out=max_out)
    log(st, f"running {item['id']} ({len(item['members'])} member(s), {item['tokens']:,} tok) on the {ROUTE_LABEL[route]}")
    try:
        if not prompts:
            raise NexusError("the prompt chain is empty -- add at least one prompt")
        content = build_final_text(item) if kind == "final" else build_chunk_text(root, item)
        first = prompts[0] + "\n\n" + content
        ctx = ctx_of(info, m_id(cfg))
        if ctx:
            n = nexus_tokenize(cfg, first)
            budget = ctx - max(RESERVE_TOKENS, max_out)
            if n > budget:
                raise NexusError(f"too big for this model: the prompt is {n:,} tokens and the model's context is "
                                 f"{ctx:,} (with {ctx - budget:,} kept free for the answer). Reset it and send it to "
                                 f"the large-context model, or rebuild with this file set to Split.")
        messages = [{"role": "user", "content": first}]
        conv = None
        kv_ctx = None          # tokens in the model's memory right after our last answer (None = not known to be ours)
        for i, p in enumerate(prompts):
            if i > 0:
                messages.append({"role": "user", "content": p})
            live_set(root, prompt_index=i, text="", gen=0, stat={"phase": "waiting"})
            stop_fn = lambda: ctl.cancel_now or item["id"] in ctl.skip_ids
            memory = "new"
            if i > 0 and kv_ctx is not None and nexus_context(cfg) == kv_ctx:
                # The model still holds the chunk and every answer so far: add ONLY this prompt to its memory and
                # let it answer. Nothing is re-read, and it answers with everything it already worked out.
                memory = "kept"
                live_stat(root, {"phase": "reading", "pct": 0, "prompt_tps": 0, "eta": -1})
                rd = {}
                def _rs(d, rd=rd):
                    rd.update(d); live_stat(root, d)
                nexus_add_turn(cfg, p, on_stat=_rs)
                r = nexus_chat(cfg, None, cancel=stop_fn, endpoint="/api/continue",
                               on_token=lambda t, g: live_token(root, t, g), max_out=max_out,
                               on_stat=lambda d, rd=rd: live_stat(root, dict(d, prompt_tps=d.get("prompt_tps") or rd.get("prompt_tps", 0),
                                                                             prompt_tokens=d.get("prompt_tokens") or rd.get("prompt_tokens", 0))))
                if rd.get("prompt_tps"):
                    r.setdefault("stat", {}).update(prompt_tps=rd["prompt_tps"], prompt_tokens=rd.get("prompt_tokens", 0))
            else:
                if i > 0:
                    memory = "re-read"
                    log(st, f"  {item['id']}: the model no longer holds this chunk in memory (restarted, or used by "
                            f"something else) -- reading the chunk and the earlier answers again for step {i+1}")
                r = nexus_chat(cfg, messages, conv_id=conv, cancel=stop_fn,
                               on_token=lambda t, g: live_token(root, t, g), max_out=max_out,
                               on_stat=lambda d: live_stat(root, d))
            conv = r["conv_id"] or conv
            kv_ctx = nexus_context(cfg)
            step = {"prompt": p, "text": r["text"], "memory": memory}
            if r["truncated"]:
                step["truncated"] = max_out
            if r["cancelled"]:
                step["stopped"] = True
            if r["notice"]:
                step["notice"] = r["notice"]
            sr = r.get("stat") or {}
            step["speed"] = {"prompt_tokens": sr.get("prompt_tokens", 0), "prompt_tps": sr.get("prompt_tps", 0),
                             "out_tokens": r.get("gen", 0), "out_tps": sr.get("out_tps", 0),
                             "first_token_s": sr.get("first_token_s")}
            messages.append({"role": "assistant", "content": r["text"]})
            with _lock:
                item["steps"].append(step)
                save_state(st)
            live_push_answer(root, r["text"])
            sp = step["speed"]
            log(st, f"  {item['id']}: step {i+1}/{len(prompts)} done ({len(r['text'])} chars, {r.get('gen', 0)} tok"
                    + (", memory kept: only the prompt was read" if memory == "kept" else "")
                    + (f", read {sp['prompt_tokens']:,} tok at {sp['prompt_tps']:.1f} tok/s" if sp["prompt_tps"] else "")
                    + (f", wrote at {sp['out_tps']:.1f} tok/s" if sp["out_tps"] else "")
                    + (f", cut at the {max_out}-token output limit" if r["truncated"] else "")
                    + (", STOPPED" if r["cancelled"] else "") + ")")
            if r["cancelled"]:
                skipped = item["id"] in ctl.skip_ids
                ctl.skip_ids.discard(item["id"])
                with _lock:
                    item["status"] = "stopped"
                    item["stop_reason"] = "chunk" if skipped else "run"
                    item["finished"] = time.time()
                    save_state(st)
                log(st, f"  {item['id']}: stopped" + (" and skipped" if skipped else ""))
                return
        if kind == "final":
            # The final model now holds this conversation in memory: keep it for follow-up questions
            # (only the LAST final chunk stays in memory when there are several).
            ctx_now = nexus_context(cfg)
            with _lock:
                st["followup"] = {"chunk": item["id"], "model": item.get("model", ""), "ctx": ctx_now,
                                  "chain": [{"prompt": stp["prompt"], "text": stp["text"]} for stp in item["steps"]],
                                  "turns": [], "busy": False, "error": "", "kv": None}
        with _lock:
            item["result"] = item["steps"][-1]["text"]
            item["status"] = "done"
            item["finished"] = time.time()
            save_state(st)
        ctl.finished += 1
    except NexusError as e:
        with _lock:
            item["status"], item["error"], item["finished"] = "error", str(e), time.time()
            save_state(st)
        log(st, f"  {item['id']}: FAILED - {e}")
    finally:
        ctl.current_id = None
        live_clear(root)

def run_phase(root, st, ctl, list_key, route):
    """Runs every runnable item on one route, always picking the next one from the live list, so
    items moved, deleted or reset mid-run are honoured."""
    cfg = info = None
    while not (ctl.cancel_now or ctl.stop_after):
        with _lock:
            item = next((c for c in st[list_key] if c.get("route") == route and runnable(c)), None)
            if item is None:
                return
            item["status"] = "running"
            item["steps"] = []
            item["started"] = time.time()
            for k in ("error", "notice", "stop_reason", "result", "finished", "result_tokens"):
                item.pop(k, None)
            save_state(st)
        if cfg is None:
            try:
                cfg, info = ensure_model(root, st, route, ctl)
            except Cancelled:
                with _lock:
                    item["status"] = "pending"
                    save_state(st)
                return
            except NexusError as e:
                with _lock:
                    for c in st[list_key]:
                        if c.get("route") == route and (c is item or runnable(c)):
                            c["status"], c["error"] = "error", str(e)
                    save_state(st)
                log(st, f"{ROUTE_LABEL[route]} unavailable: {e}")
                return
        run_item(root, st, ctl, list_key, item, route, cfg, info)
        with _lock:
            if item.pop("restart_pending", False):
                item["status"], item["steps"] = "pending", []
                for k in ("result", "error", "notice", "stop_reason", "started", "finished", "result_tokens", "model"):
                    item.pop(k, None)
                ctl.skip_ids.discard(item["id"])
                save_state(st)
                restarted = True
            else:
                restarted = False
        if restarted:
            log(st, f"{item['id']}: restarting with the current settings")

def build_final(root, st, cfg=None):
    with _lock:
        done = [c for c in st["chunks"] if c["status"] == "done"]
        not_done = [c["id"] for c in st["chunks"] if c["status"] != "done"]
        src = [(c["id"], [m["path"] + (f" (lines {m['lines'][0]}-{m['lines'][1]})" if m.get("lines") else "")
                          for m in c["members"]], c.get("result") or "") for c in done]
        cap = st["cap_tokens"]
    items = []
    for cid, files, text in src:
        try:
            n = nexus_tokenize(cfg, text) if cfg else None
        except NexusError:
            n = None
        items.append({"chunk_id": cid, "files": files, "text": text, "tokens": n if n else max(1, len(text) // 3)})
    groups = pack_chunks(items, cap)
    final_chunks = [{"id": f"final{i+1}", "route": "final", "kind": "final", "members": g,
                     "tokens": sum(m["tokens"] for m in g), "status": "pending", "steps": []}
                    for i, g in enumerate(groups)]
    with _lock:
        st["final_chunks"] = final_chunks
        save_state(st)
    return final_chunks, not_done

def worker(root, mode):
    ctl = RUNS[root]
    st = get_state(root)
    with _lock:
        st["running"], st["run_mode"] = True, mode
        save_state(st)
    try:
        if mode == "chunks":
            for _pass in range(20):
                for route in ("main", "large"):
                    run_phase(root, st, ctl, "chunks", route)
                    if ctl.cancel_now or ctl.stop_after:
                        break
                if ctl.cancel_now or ctl.stop_after:
                    break
                with _lock:   # chunks reset while the run went on (e.g. to redo them on another model): one more pass
                    again = any(c["status"] == "pending" and c.get("route") in ("main", "large") for c in st["chunks"])
                if not again:
                    break
            if not (ctl.cancel_now or ctl.stop_after):
                with _lock:
                    counts = {}
                    for c in st["chunks"]:
                        counts[c["status"]] = counts.get(c["status"], 0) + 1
                log(st, "chunks finished: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
                if st.get("auto_final") and counts.get("done"):
                    cfg, _ = ensure_model(root, st, "final", ctl)
                    fc, _ = build_final(root, st, cfg)
                    log(st, f"final pass: {len(fc)} final chunk(s) built, running")
                    ctl.ran_final = True
                    run_phase(root, st, ctl, "final_chunks", "final")
        else:
            ctl.ran_final = True
            run_phase(root, st, ctl, "final_chunks", "final")
        if ctl.cancel_now or ctl.stop_after:
            log(st, "run stopped")
    except Cancelled:
        log(st, "run stopped")
    except NexusError as e:
        log(st, f"run ended: {e}")
    except Exception:
        log(st, "run crashed:\n" + traceback.format_exc())
    finally:
        with _lock:
            for lst in ("chunks", "final_chunks"):
                for c in st[lst]:
                    if c["status"] == "running":
                        c["status"] = "pending"
            st["running"], st["run_mode"], st["phase_msg"] = False, "", ""
            save_state(st)
        # The final pass may have run on its own OR automatically at the end of a chunk run: either way, keep the
        # final-pass model (and the final pass in its memory) loaded for follow-up questions.
        keep_final = ctl.ran_final and st.get("keep_final_loaded", True) and bool(st.get("followup"))
        if st.get("unload_at_end", True) and managed_view(root):
            if keep_final:
                keep_url = m_url(mcfg(st, "final"))
                with _proc_lock:
                    for k in list(MANAGED):
                        if MANAGED[k]["root"] == root and k[0] != keep_url:
                            stop_process(k)
                log(st, "kept the final-pass model loaded for follow-up questions (it still remembers the final pass)")
            else:
                log(st, "stopping the model(s) this tool started")
                stop_all_managed(root)
        if st.get("autosave", True) and ctl.finished:
            try:
                meta = create_save(root, "auto: after " + ("final pass" if mode == "final" else "run"))
                log(st, f"snapshot saved: {meta['name']}")
                if ctl.ran_final and st.get("followup"):
                    with _lock:
                        st["followup"]["autosave"] = {"local": meta["id"].split("/", 1)[1], "base": meta["name"]}
            except Exception as e:
                log(st, f"auto-save failed: {e}")
        with _lock:
            save_state(st)
        live_clear(root)
        RUNS.pop(root, None)

# --------------------------------------------------------------------------------------------
# Saves (snapshots of a project's results, browsable later -- no KV, just text)
# --------------------------------------------------------------------------------------------
SAVE_ID_RE = re.compile(r"^[0-9a-f]{16}/[0-9A-Za-z_-]{1,80}$")

def create_save(root, name="", local=None):
    """Snapshot of the project's results (and the follow-up conversation). With `local`, that snapshot is
    rewritten in place (the follow-up auto-save keeps one snapshot up to date instead of piling up new ones)."""
    st = get_state(root)
    with _lock:
        snap = {k: json.loads(json.dumps(st[k])) for k in
                ("root", "cap_tokens", "models", "chunk_prompts", "final_prompts", "chunks", "final_chunks")}
        if st.get("followup"):
            snap["followup"] = json.loads(json.dumps(st["followup"]))
            snap["followup"]["busy"] = False
            kv = snap["followup"].get("kv")
            if kv and kv.get("ctx") != snap["followup"].get("ctx"):
                # that saved memory is from earlier in the conversation: loading it would not match these turns
                snap["followup"]["kv"] = None
    if not (local and re.fullmatch(r"[0-9A-Za-z_-]{1,80}", local)):
        local = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
    sid = f"{project_key(root)}/{local}"
    meta = {"id": sid, "name": (name or "").strip() or time.strftime("snapshot %Y-%m-%d %H:%M"),
            "root": snap["root"], "created": time.time(),
            "chunks": len(snap["chunks"]), "chunks_done": sum(c["status"] == "done" for c in snap["chunks"]),
            "final": len(snap["final_chunks"]), "final_done": sum(c["status"] == "done" for c in snap["final_chunks"]),
            "followup_turns": sum(1 for t in (snap.get("followup") or {}).get("turns", []) if t["role"] == "assistant"),
            "kv": (snap.get("followup") or {}).get("kv")}
    d = os.path.join(project_dir(root), "saves")
    os.makedirs(d, exist_ok=True)
    for fn, obj in ((local + ".json", dict(snap, meta=meta)), (local + ".meta.json", meta)):
        p = os.path.join(d, fn)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(p + ".tmp", p)
    return meta

def list_saves():
    out = []
    for key in os.listdir(DATA_ROOT):
        d = os.path.join(DATA_ROOT, key, "saves")
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".meta.json"):
                try:
                    with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                        out.append(json.load(f))
                except (OSError, ValueError):
                    pass
    return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

def save_paths(sid):
    if not SAVE_ID_RE.match(sid or ""):
        raise ValueError("bad save id")
    key, local = sid.split("/")
    d = os.path.join(DATA_ROOT, key, "saves")
    return os.path.join(d, local + ".json"), os.path.join(d, local + ".meta.json")

# --------------------------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------------------------
ALLOWED_HOSTS = set()   # filled in main(): 127.0.0.1:<port>, localhost:<port>

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body, code=200, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        return self._send(json.dumps(obj).encode("utf-8"), code)

    def _trusted(self):
        """This tool can start processes (launch commands) and read any folder you point it at, so
        only accept requests addressed to this server by its own name (blocks DNS rebinding) and,
        if a browser says where a request came from, only from this page itself (blocks other
        websites from driving it)."""
        host = (self.headers.get("Host") or "").lower()
        if host not in ALLOWED_HOSTS:
            return False
        origin = (self.headers.get("Origin") or "").lower()
        return not origin or origin in {"http://" + h for h in ALLOWED_HOSTS}

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            v = json.loads(raw or b"{}")
            return v if isinstance(v, dict) else {}
        except ValueError:
            return {}

    def _q(self, name):
        return parse_qs(urlparse(self.path).query).get(name, [None])[0]

    def _root(self, body):
        root = body.get("root") or self._q("root")
        if not root:
            raise ValueError("missing 'root'")
        return os.path.abspath(root)

    def do_GET(self):
        if not self._trusted():
            return self._json({"error": "forbidden"}, 403)
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                return self._serve_static("index.html")
            if u.path.startswith("/static/"):
                return self._serve_static(u.path[len("/static/"):])
            if u.path == "/api/state":
                root = self._root({})
                status = route_status(get_state(root), root)
                with _lock:
                    st = get_state(root)
                    v = {k: val for k, val in st.items() if not (self._q("lite") and k == "files")}
                    v["file_count"] = len(st["files"])
                    v["managed"] = managed_view(root)
                    v["model_status"] = status
                    body = json.dumps(v).encode("utf-8")
                return self._send(body)
            if u.path == "/api/live":
                root = self._root({})
                with _live_lock:
                    d = json.loads(json.dumps(LIVE.get(root, {})))
                return self._json(d)
            if u.path == "/api/nexus":
                return self._json(nexus_view(force_probe=bool(self._q("refresh"))))
            if u.path == "/api/library":
                g = load_global()
                return self._json({"prompt_sets": prompt_sets(), "recent": [r for r in (g.get("recent") or []) if os.path.isdir(r)],
                                   "has_defaults": bool(g.get("defaults")),
                                   "default_cap": (g.get("defaults") or {}).get("cap_tokens")})
            if u.path == "/api/browse":
                return self._json(browse_dir(self._q("path") or "", bool(self._q("hidden"))))
            if u.path == "/api/launch_preview":
                root = self._root({})
                st = get_state(root)
                shown, argv, env, cwd, err = launch_spec(mcfg(st, self._q("route") or "main"))
                return self._json({"command": shown, "error": err})
            if u.path == "/api/saves":
                return self._json({"saves": list_saves()})
            if u.path == "/api/saves/get":
                data_p, _ = save_paths(self._q("id"))
                with open(data_p, "r", encoding="utf-8") as f:
                    return self._send(f.read().encode("utf-8"))
        except (ValueError, OSError) as e:
            return self._json({"error": str(e)}, 400)
        return self._json({"error": "not found"}, 404)

    def _serve_static(self, name):
        full = os.path.normpath(os.path.join(STATIC, name.lstrip("/")))
        if not full.startswith(STATIC + os.sep) or not os.path.isfile(full):
            return self._json({"error": "not found"}, 404)
        ctype = {"js": "application/javascript", "css": "text/css", "html": "text/html; charset=utf-8"}.get(
            full.rsplit(".", 1)[-1], "application/octet-stream")
        with open(full, "rb") as f:
            return self._send(f.read(), 200, ctype)

    def do_POST(self):
        if not self._trusted() or "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json({"error": "forbidden"}, 403)
        u = urlparse(self.path)
        body = self._body()
        try:
            handler = POST_ROUTES.get(u.path)
            if not handler:
                return self._json({"error": "not found"}, 404)
            return handler(self, body)
        except NexusError as e:
            return self._json({"error": str(e)}, 502)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}, 500)

def _find(st, list_key, cid):
    if list_key not in ("chunks", "final_chunks"):
        raise ValueError("list must be 'chunks' or 'final_chunks'")
    for i, c in enumerate(st[list_key]):
        if c["id"] == cid:
            return i, c
    raise ValueError(f"no item '{cid}'")

def h_project(self, body):
    root = self._root(body)
    if not os.path.isdir(root):
        return self._json({"error": f"not a folder: {root}"}, 400)
    g = load_global()
    g["recent"] = [root] + [r for r in (g.get("recent") or []) if r != root][:14]
    save_global(g)
    st = get_state(root)
    status = route_status(st, root)
    with _lock:
        save_state(st)
        v = {k: val for k, val in st.items() if k != "files"}
        v["file_count"] = len(st["files"])
        v["managed"] = managed_view(root)
        v["model_status"] = status
        return self._json(v)

def h_project_delete(self, body):
    root = self._root(body)
    with _lock:
        st = get_state(root)
        if st["running"]:
            return self._json({"error": "stop the run first"}, 409)
        p = state_path(root)
        if os.path.isfile(p):
            os.remove(p)
        STATES.pop(root, None)
    stop_all_managed(root)
    live_clear(root)
    return self._json({"ok": True, "note": "saved snapshots of this project were kept"})

def h_settings(self, body):
    root = self._root(body)
    with _lock:
        st = get_state(root)
        if "cap_tokens" in body:
            st["cap_tokens"] = max(256, int(body["cap_tokens"]))
        if "large_cap_tokens" in body:
            st["large_cap_tokens"] = max(256, int(body["large_cap_tokens"]))
        for k in ("auto_final", "unload_on_switch", "unload_at_end", "autosave", "keep_final_loaded"):
            if k in body:
                st[k] = bool(body[k])
        for route, m in (body.get("models") or {}).items():
            if route in st["models"] and isinstance(m, dict):
                cur = st["models"][route]
                for f in ("url", "model", "launch", "think", "file", "gpu", "chat_format", "extra_args", "rope", "norm"):
                    if f in m:
                        cur[f] = str(m[f] or "").strip()
                if m.get("source") in ("file", "url", "custom"):
                    cur["source"] = m["source"]
                for f in ("port", "threads"):
                    if f in m:
                        cur[f] = max(0, min(65535, int(float(m[f] or 0))))
                for f in ("full_gpu", "turbo"):
                    if f in m:
                        cur[f] = bool(m[f])
                if isinstance(m.get("sampling"), dict):
                    cur["sampling"] = {k: m["sampling"][k] for k in DEFAULT_SAMPLING if str(m["sampling"].get(k, "")).strip() != ""}
                if "max_out" in m:
                    cur["max_out"] = max(0, int(m["max_out"] or 0))
                if "exclusive" in m:
                    cur["exclusive"] = bool(m["exclusive"])
        for k in ("chunk_prompts", "final_prompts"):
            if k in body:
                st[k] = [p for p in body[k] if isinstance(p, str) and p.strip()]
        save_state(st)
        store_defaults(st)                                  # ...and the starting point for every new project
    return self._json({"ok": True})

def h_defaults_apply(self, body):
    """Use the saved defaults in this (older) project too."""
    root = self._root(body)
    d = saved_defaults()
    if not d:
        return self._json({"error": "no saved defaults yet -- they are stored the first time you press Save settings or Save prompts"}, 400)
    with _lock:
        st = get_state(root)
        if st["running"]:
            return self._json({"error": "stop the run first"}, 409)
        apply_defaults(st, d)
        save_state(st)
    return h_project(self, body)

# ---- saved prompt chains: built-in ones plus your own, for the chunk chain and the final chain
NOTES_SWEEP_PROMPTS = [
    "Go through the code above and list everything that should not ship in a public release: personal "
    "information (names, e-mail addresses, home paths, machine names, IP addresses, keys, tokens, passwords), "
    "leftover notes to self, TODO/FIXME/HACK comments, debugging leftovers, commented-out experiments, and "
    "anything unprofessional or embarrassing. For each one give the file, the line or function, and quote it. "
    "If a file is clean, say so plainly.",
]
SECOND_LOOK_PROMPT = (
    "Now go through the code above one more time, looking only for what you MISSED: places where input from "
    "outside (network requests, files, command-line arguments, environment variables, model output) reaches a "
    "shell, a file path, memory sizes or indexes, or another process without being checked. Add only real, new "
    "findings with file, function and quoted lines -- or say there is nothing to add.")
CONSOLIDATE_PROMPTS = [
    "These are individual reports across the entire codebase. Consolidate them into one report: every distinct "
    "issue once, with the file(s) it is in, how serious it is, and what to do about it. Keep the exact file and "
    "function names the reports give. Mark anything that only one report mentions and that you cannot judge "
    "from the reports alone as \"needs checking\".",
]
BUILTIN_PROMPT_SETS = {"chunk": {"Built-in: security audit": DEFAULT_CHUNK_PROMPTS,
                                 "Built-in: security audit + second look": DEFAULT_CHUNK_PROMPTS + [SECOND_LOOK_PROMPT],
                                 "Built-in: notes & personal-info sweep": NOTES_SWEEP_PROMPTS},
                       "final": {"Built-in: combine & rank findings": DEFAULT_FINAL_PROMPTS,
                                 "Built-in: consolidate into one report": CONSOLIDATE_PROMPTS}}

def prompt_sets():
    mine = load_global().get("prompt_sets") or {}
    out = {}
    for kind in ("chunk", "final"):
        own = mine.get(kind) if isinstance(mine.get(kind), dict) else {}
        out[kind] = [{"name": n, "prompts": list(p), "builtin": True} for n, p in BUILTIN_PROMPT_SETS[kind].items()] + \
                    [{"name": n, "prompts": list(p), "builtin": False} for n, p in sorted(own.items(), key=lambda kv: kv[0].lower())]
    return out

def h_prompt_set_save(self, body):
    kind, name = body.get("kind"), str(body.get("name") or "").strip()[:80]
    prompts = [p for p in (body.get("prompts") or []) if isinstance(p, str) and p.strip()]
    if kind not in ("chunk", "final") or not name:
        return self._json({"error": "need kind (chunk/final) and a name"}, 400)
    if name in BUILTIN_PROMPT_SETS[kind]:
        return self._json({"error": "that name belongs to a built-in set -- pick another"}, 400)
    if not prompts:
        return self._json({"error": "the chain is empty"}, 400)
    g = load_global()
    g.setdefault("prompt_sets", {}).setdefault(kind, {})[name] = prompts
    save_global(g)
    return self._json({"ok": True, "sets": prompt_sets()})

def h_prompt_set_delete(self, body):
    kind, name = body.get("kind"), str(body.get("name") or "")
    g = load_global()
    (g.get("prompt_sets") or {}).get(kind, {}).pop(name, None)
    save_global(g)
    return self._json({"ok": True, "sets": prompt_sets()})

def browse_dir(path, hidden=False):
    """Folders inside `path` for the folder picker (this tool only listens on 127.0.0.1)."""
    home = os.path.expanduser("~")
    p = os.path.abspath(os.path.expanduser(path or home))
    while not os.path.isdir(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)                              # a typed path that doesn't exist: nearest folder above it
    dirs, err = [], ""
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    if e.is_dir() and (hidden or not e.name.startswith(".")):
                        dirs.append(e.name)
                except OSError:
                    pass
    except OSError as e:
        err = str(e)
    dirs.sort(key=str.lower)
    parent = os.path.dirname(p)
    roots = []
    if os.name == "nt":
        roots = [f"{c}:\\" for c in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.isdir(f"{c}:\\")]
    return {"path": p, "parent": parent if parent != p else "", "dirs": dirs, "home": home, "roots": roots, "error": err,
            "is_project": os.path.isfile(os.path.join(DATA_ROOT, project_key(p), "state.json"))}   # no folder created just by looking

def h_nexus_dir(self, body):
    d = os.path.abspath(os.path.expanduser(str(body.get("dir") or "").strip())) if body.get("dir") else ""
    if d and not server_in(d):
        return self._json({"error": f"no nexus_server in {d} (or in a 'server' folder inside it)"}, 400)
    g = load_global()
    if d:
        g["nexus_dir"] = d
    else:
        g.pop("nexus_dir", None)
    save_global(g)
    return self._json(nexus_view(force_probe=True))

def h_model_test(self, body):
    cfg = {"url": body.get("url", ""), "model": body.get("model", "1"), "api_key": body.get("api_key", "")}
    _register_key(cfg)
    info = nexus_info(cfg, timeout=8)
    return self._json({"ok": True, "info": info, "context_size": ctx_of(info, cfg["model"])})

def h_model_start(self, body):
    root = self._root(body)
    st = get_state(root)
    cfg = mcfg(st, body.get("route", "main"))
    shown, argv, env, cwd, err = launch_spec(cfg)
    if err:
        return self._json({"error": err}, 400)
    key = (m_url(cfg), shown)
    if not key[1]:
        return self._json({"error": "this model has no way to be started here -- pick a model file, or give it a launch command"}, 400)
    with _proc_lock:
        ent = managed_alive(key[0])
        if ent:
            return self._json({"ok": True, "note": "already running" + ("" if ent["launch"] == shown else
                               " (with older settings -- Stop and Start it to apply the new ones)"), "pid": ent["proc"].pid})
        try:
            nexus_info(cfg, timeout=3)
            if cfg.get("source") == "file":
                return self._json({"error": f"port {cfg.get('port')} is already used by another server -- pick another port"}, 409)
        except NexusError:
            pass
        ent = start_process(root, key, cfg.get("exclusive"), argv, env, cwd)
    log(st, f"started model by hand: {key[1]}")
    return self._json({"ok": True, "pid": ent["proc"].pid, "log": ent["log"]})

def h_model_stop(self, body):
    root = self._root(body)
    st = get_state(root)
    if body.get("route"):
        with _proc_lock:
            stop_process(m_key(mcfg(st, body["route"])))
    else:
        stop_all_managed(root)
    return self._json({"ok": True})

def h_scan(self, body):
    root = self._root(body)
    st = get_state(root)
    cfg = mcfg(st, "main")
    info, err, t0 = None, None, time.time()
    while info is None:
        try:
            info = nexus_info(cfg)
        except NexusError as e:
            err = e
            ent = managed_alive(m_url(cfg))
            if not ent or time.time() - t0 > 1800:   # only wait for a model this tool started and that is still loading
                break
            time.sleep(2)
    if info is None:
        return self._json({"error": f"the chunk model isn't reachable at {m_url(cfg) or '(no address set)'} ({err}). "
                                    f"Start it (launcher, or the Start button in the Models card) and scan again."}, 502)
    with _lock:
        if st["running"]:
            return self._json({"error": "a run is in progress"}, 409)
    files = []
    for rel, full, sz, mt in scan_files(st):
        key = (full, mt, sz, m_url(cfg), m_id(cfg))
        if key not in TOKEN_CACHE:
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            TOKEN_CACHE[key] = nexus_tokenize(cfg, text)
        files.append({"path": rel, "tokens": TOKEN_CACHE[key], "size": sz, "mtime": mt})
    with _lock:
        cap = st["cap_tokens"]
        old = {o["path"]: o.get("decision", "exclude") for o in st["oversized"]}
        st["files"] = files
        st["oversized"] = [dict(f, decision=old.get(f["path"], "exclude")) for f in files if f["tokens"] > cap]
        save_state(st)
        return self._json({"file_count": len(files), "oversized": st["oversized"], "context_size": ctx_of(info, m_id(cfg)),
                           "total_tokens": sum(f["tokens"] for f in files)})

def h_oversized_decision(self, body):
    root = self._root(body)
    if body.get("decision") not in ("exclude", "split", "split_large", "large", "force"):
        return self._json({"error": "decision must be exclude | split | split_large | large | force"}, 400)
    with _lock:
        st = get_state(root)
        for o in st["oversized"]:
            if o["path"] == body.get("path"):
                o["decision"] = body["decision"]
                save_state(st)
                return self._json({"ok": True})
    return self._json({"error": "not in the oversized list"}, 400)

def h_build_chunks(self, body):
    root = self._root(body)
    st = get_state(root)
    with _lock:
        if st["running"]:
            return self._json({"error": "a run is in progress"}, 409)
        cap = st["cap_tokens"]
        dec = {o["path"]: o["decision"] for o in st["oversized"]}
        over = {o["path"]: o for o in st["oversized"]}
        normal = [f for f in st["files"] if f["path"] not in dec]
        large_cap = int(st.get("large_cap_tokens") or 250000)
        to_split = [p for p, d in dec.items() if d in ("split", "split_large")]
    split_parts = {}
    if to_split:
        cfg = mcfg(st, "main")
        try:
            for p in to_split:   # parts sized for the model that will read them
                split_parts[p] = split_file(root, cfg, p, large_cap if dec[p] == "split_large" else cap)
        except NexusError as e:
            return self._json({"error": f"splitting uses the chunk model's tokenizer, which isn't reachable: {e}"}, 502)
    chunks = []
    for i, g in enumerate(pack_chunks([{"path": f["path"], "tokens": f["tokens"]} for f in normal], cap)):
        chunks.append({"id": f"chunk{i+1}", "kind": "files", "route": "main", "members": g,
                       "tokens": sum(m["tokens"] for m in g)})
    n = nl = 0
    for p in to_split:
        big = dec[p] == "split_large"
        for part in split_parts[p]:
            if big:
                nl += 1
                chunks.append({"id": f"lsplit{nl}", "kind": "split", "route": "large", "members": [part], "tokens": part["tokens"]})
            else:
                n += 1
                chunks.append({"id": f"split{n}", "kind": "split", "route": "main", "members": [part], "tokens": part["tokens"]})
    n = 0
    for p, d in dec.items():
        if d in ("force", "large"):
            n += 1
            chunks.append({"id": ("forced" if d == "force" else "large") + str(n), "kind": "forced" if d == "force" else "large",
                           "route": "main" if d == "force" else "large",
                           "members": [{"path": p, "tokens": over[p]["tokens"]}], "tokens": over[p]["tokens"]})
    for c in chunks:
        c.update(status="pending", steps=[])
    with _lock:
        st["chunks"], st["final_chunks"] = chunks, []
        save_state(st)
    return self._json({"chunks": len(chunks), "excluded": sorted(p for p, d in dec.items() if d == "exclude"),
                       "split_parts": sum(len(v) for v in split_parts.values())})

def h_item(self, body):
    """Per-item actions: reset | delete | move (dir: up/down/top/bottom) | route (main/large) | skip."""
    root = self._root(body)
    action, list_key, cid = body.get("action"), body.get("list", "chunks"), body.get("id")
    st = get_state(root)
    with _lock:
        i, c = _find(st, list_key, cid)
        lst = st[list_key]
        if action == "skip":
            ctl = RUNS.get(root)
            if c["status"] != "running" or not ctl:
                return self._json({"error": "that item isn't running"}, 409)
            ctl.skip_ids.add(cid)
            threading.Thread(target=nexus_stop, args=(ctl.cfg_active,), daemon=True).start()
            return self._json({"ok": True})
        if action == "restart":
            # stop it now and run it again from scratch -- with whatever model / prompts are set NOW
            ctl = RUNS.get(root)
            if c["status"] != "running" or not ctl:
                return self._json({"error": "that item isn't running"}, 409)
            c["restart_pending"] = True
            ctl.skip_ids.add(cid)
            threading.Thread(target=nexus_stop, args=(ctl.cfg_active,), daemon=True).start()
            return self._json({"ok": True})
        if action == "move":
            j = {"up": i - 1, "down": i + 1, "top": 0, "bottom": len(lst) - 1}.get(body.get("dir"), i)
            j = max(0, min(len(lst) - 1, j))
            lst.insert(j, lst.pop(i))
        elif c["status"] == "running":
            return self._json({"error": "stop it first (Stop & skip)"}, 409)
        elif action == "delete":
            lst.pop(i)
        elif action == "reset":
            c["status"], c["steps"] = "pending", []
            for k in ("result", "error", "notice", "stop_reason", "started", "finished", "result_tokens", "model"):
                c.pop(k, None)
        elif action == "route" and list_key == "chunks" and body.get("route") in ("main", "large"):
            c["route"] = body["route"]
            if c["status"] in ("error", "stopped"):
                c["status"], c["steps"] = "pending", []
                c.pop("error", None); c.pop("stop_reason", None)
        else:
            return self._json({"error": "unknown action"}, 400)
        save_state(st)
    return self._json({"ok": True})

# ---- follow-up chat with the final-pass model (it still remembers the final pass) -----------------
FU_THREADS = {}

def _fu_files_text(root, paths, uploads):
    parts, names = [], []
    for p in paths or []:
        rel = str(p).replace("\\", "/").lstrip("/")
        full = os.path.realpath(os.path.join(root, rel))
        if not full.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(full):
            raise NexusError(f"not a file in this project: {p}")
        parts.append(f"=== FILE: {rel} ===\n{read_file_text(root, rel)}")
        names.append(rel)
    for u in uploads or []:
        n, t = str(u.get("name") or "file"), str(u.get("text") or "")
        parts.append(f"=== FILE: {n} (attached) ===\n{t}")
        names.append(n)
    return parts, names

def _fu_worker(root, text, names, reread):
    st = get_state(root)
    fu = st["followup"]
    cfg = mcfg(st, "final")
    ctl = RunCtl()
    FU_THREADS[root] = ctl
    # Your message shows in the conversation right away, like a chat; the answer follows it.
    uturn = {"role": "user", "text": text if not names else text.split("\n\n=== FILE: ")[0], "files": names,
             "sent": text, "pending": True}
    with _lock:
        fu["turns"].append(uturn)
        save_state(st)
    answered = False
    reread = reread or bool(fu.get("reread_next"))
    try:
        cfg, info = ensure_model(root, st, "final", ctl)
        ctl.cfg_active = cfg
        live_set(root, chunk_id="followup", kind="followup", answers=[], text="", gen=0, prompt_index=0, max_out=0,
                 stat={"phase": "reading"})
        now = nexus_context(cfg)
        if reread or (fu.get("ctx") is not None and now != fu["ctx"]):
            if not reread:
                raise NexusError(f"the final model no longer holds this conversation in memory (it has {now if now is not None else '?'} "
                                 f"tokens, expected {fu['ctx']}): it was restarted or used by something else. Load the saved memory "
                                 f"(if you saved one), or send again with \"read it all again\" ticked.")
            # rebuild the whole conversation by reading it again (slow for a big final chunk, but exact)
            with _lock:
                item = next((c for c in st["final_chunks"] if c["id"] == fu["chunk"]), None)
            if not item:
                raise NexusError("the final chunk this conversation started from is gone (rebuilt?)")
            msgs = [{"role": "user", "content": fu["chain"][0]["prompt"] + "\n\n" + build_final_text(item)}]   # as run_item builds it
            msgs.append({"role": "assistant", "content": fu["chain"][0]["text"]})
            for stp in fu["chain"][1:]:
                msgs += [{"role": "user", "content": stp["prompt"]}, {"role": "assistant", "content": stp["text"]}]
            prev = [t for t in fu["turns"] if t is not uturn]
            for i, t in enumerate(prev):
                if t["role"] == "user" and not (i + 1 < len(prev) and prev[i + 1]["role"] == "assistant"):
                    continue                                   # a message that never got an answer
                msgs.append({"role": t["role"], "content": t.get("sent", t["text"])})
            msgs.append({"role": "user", "content": text})
            r = nexus_chat(cfg, msgs, cancel=lambda: ctl.cancel_now,
                           on_token=lambda t, g: live_token(root, t, g), on_stat=lambda d: live_stat(root, d))
        else:
            live_stat(root, {"phase": "reading", "pct": 0, "prompt_tps": 0, "eta": -1})
            rd = {}
            def _rs(d):
                rd.update(d); live_stat(root, d)
            nexus_add_turn(cfg, text, on_stat=_rs)
            r = nexus_chat(cfg, None, cancel=lambda: ctl.cancel_now, endpoint="/api/continue",
                           on_token=lambda t, g: live_token(root, t, g),
                           on_stat=lambda d: live_stat(root, dict(d, prompt_tps=d.get("prompt_tps") or rd.get("prompt_tps", 0))))
            if rd.get("prompt_tps"):
                r.setdefault("stat", {}).update(prompt_tps=rd["prompt_tps"], prompt_tokens=rd.get("prompt_tokens", 0))
        sr = r.get("stat") or {}
        with _lock:
            uturn.pop("pending", None)
            fu["turns"].append({"role": "assistant", "text": r["text"], "stopped": bool(r["cancelled"]),
                                "speed": {"out_tokens": r.get("gen", 0), "out_tps": sr.get("out_tps", 0),
                                          "prompt_tokens": sr.get("prompt_tokens", 0), "prompt_tps": sr.get("prompt_tps", 0)}})
            fu["ctx"] = nexus_context(cfg)
            fu["error"] = ""
            fu.pop("reread_next", None)
            save_state(st)
        answered = True
        log(st, f"follow-up answered ({r.get('gen', 0)} tok" + (f" at {sr.get('out_tps', 0):.1f} tok/s" if sr.get("out_tps") else "") + ")")
    except NexusError as e:
        with _lock:
            fu["error"] = str(e)
            save_state(st)
    except Exception:
        with _lock:
            fu["error"] = "follow-up crashed:\n" + traceback.format_exc()
            save_state(st)
    finally:
        with _lock:
            if not answered:
                uturn.pop("pending", None); uturn["unanswered"] = True
            fu["busy"] = False
            save_state(st)
        live_clear(root)
        FU_THREADS.pop(root, None)
        if answered and st.get("autosave", True):
            _fu_autosave(root, st)

def _fu_autosave(root, st):
    """After every answered follow-up: keep one snapshot of this conversation up to date (the run's auto-save,
    or a new one), so nothing is lost without saving by hand. It's text only -- model memory is saved with
    'Save report with final KV'."""
    with _lock:
        fu = st.get("followup") or {}
        a = dict(fu.get("autosave") or {})
        n = sum(1 for t in fu.get("turns", []) if t["role"] == "assistant")
    try:
        meta = create_save(root, f"{a.get('base') or 'auto: follow-up'} + {n} follow-up" + ("s" if n != 1 else ""), local=a.get("local"))
        with _lock:
            if st.get("followup") is fu:
                fu["autosave"] = {"local": meta["id"].split("/", 1)[1], "base": a.get("base") or "auto: follow-up"}
                save_state(st)
    except Exception as e:
        log(st, f"follow-up auto-save failed: {e}")

def h_followup_send(self, body):
    root = self._root(body)
    st = get_state(root)
    text = str(body.get("text") or "").strip()
    with _lock:
        fu = st.get("followup")
        if not fu:
            return self._json({"error": "run the final pass first -- follow-ups talk to the final-pass model"}, 400)
        if st["running"] or fu.get("busy"):
            return self._json({"error": "busy -- wait for the run or the current answer to finish"}, 409)
    try:
        parts, names = _fu_files_text(root, body.get("files"), body.get("uploads"))
    except NexusError as e:
        return self._json({"error": str(e)}, 400)
    if not text and not parts:
        return self._json({"error": "type a message or add a file"}, 400)
    full = "\n\n".join(([text] if text else []) + parts)
    with _lock:
        fu["busy"], fu["error"] = True, ""
        save_state(st)
    threading.Thread(target=_fu_worker, args=(root, full, names, bool(body.get("reread"))), daemon=True).start()
    return self._json({"ok": True})

def h_followup_stop(self, body):
    root = self._root(body)
    ctl = FU_THREADS.get(root)
    if ctl:
        ctl.cancel_now = True
        if ctl.cfg_active:
            threading.Thread(target=nexus_stop, args=(ctl.cfg_active,), daemon=True).start()
    return self._json({"ok": True})

def h_followup_save(self, body):
    """Save report with final KV: the model's memory goes to Nexus's sessions folder, the report (with this
    conversation) to the saved reports, linked together."""
    root = self._root(body)
    st = get_state(root)
    with _lock:
        fu = st.get("followup")
        if not fu or fu.get("busy") or st["running"]:
            return self._json({"error": "nothing to save yet, or busy"}, 409)
    cfg = mcfg(st, "final")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(body.get("name") or "").strip())[:60].strip("-") or "report"
    kvp = f"nexus-data/sessions/harden-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    now = nexus_context(cfg)
    if fu.get("ctx") is not None and now != fu["ctx"]:
        return self._json({"error": "the final model no longer holds this conversation in memory, so there is nothing to save"}, 409)
    try:
        nexus_session(cfg, "save", kvp)
    except NexusError as e:
        return self._json({"error": f"saving the model's memory failed: {e}"}, 502)
    with _lock:
        fu["kv"] = {"path": kvp, "ctx": now, "model": fu.get("model", ""), "saved": time.time()}
        save_state(st)
    meta = create_save(root, (str(body.get("name") or "").strip() or "final pass") + " (with model memory)")
    log(st, f"saved the report and the final model's memory ({now} tokens) as {kvp}")
    return self._json({"ok": True, "save": meta, "kv": fu["kv"]})

def h_followup_resume(self, body):
    """Continue a saved report's conversation: load its saved memory into the final model, then pick up where it was."""
    root = self._root(body)
    st = get_state(root)
    sid = str(body.get("id") or "")
    try:
        key, local = sid.split("/", 1)
        with open(os.path.join(DATA_ROOT, key, "saves", os.path.basename(local) + ".json"), "r", encoding="utf-8") as f:
            snap = json.load(f)
    except (ValueError, OSError):
        return self._json({"error": "saved report not found"}, 404)
    fu = snap.get("followup")
    if not fu:
        return self._json({"error": "this saved report has no follow-up conversation"}, 400)
    if os.path.realpath(snap.get("root", "")) != os.path.realpath(root):
        return self._json({"error": f"open its project first ({snap.get('root')})"}, 400)
    with _lock:
        if st["running"] or (st.get("followup") or {}).get("busy"):
            return self._json({"error": "busy -- wait for the run or the current answer to finish"}, 409)
    sname = re.sub(r" \+ \d+ follow-ups?$", "", (snap.get("meta") or {}).get("name", "")).strip()
    if sname.startswith("auto:"):
        fu["autosave"] = {"local": os.path.basename(local), "base": sname}        # keep that auto-snapshot up to date
    else:
        fu["autosave"] = {"base": "auto: " + (sname or "follow-up")}             # a save you named stays as you saved it
    if not fu.get("kv"):
        # no saved memory: the next message has the model read the final chunk and the conversation again
        fu["busy"], fu["ctx"], fu["reread_next"] = False, None, True
        fu["error"] = ("This save has no model memory, so your next message makes the final model read the report and "
                       "this conversation again first (slow for a big report, but exact). After that it's quick again.")
        with _lock:
            st["followup"] = fu
            if snap.get("final_chunks") and not any(c["id"] == fu["chunk"] for c in st["final_chunks"]):
                st["final_chunks"] = snap["final_chunks"]
            save_state(st)
        log(st, "reopened a saved follow-up conversation without model memory: the next message re-reads it")
        return self._json({"ok": True, "reread": True})
    try:
        cfg, info = ensure_model(root, st, "final", RunCtl())
        nexus_session(cfg, "load", fu["kv"]["path"])
    except NexusError as e:
        return self._json({"error": str(e)}, 502)
    now = nexus_context(cfg)
    fu["busy"], fu["error"], fu["ctx"] = False, "", now
    fu.pop("reread_next", None)
    with _lock:
        st["followup"] = fu
        if snap.get("final_chunks") and not any(c["id"] == fu["chunk"] for c in st["final_chunks"]):
            st["final_chunks"] = snap["final_chunks"]
        save_state(st)
    log(st, f"loaded the saved model memory {fu['kv']['path']} ({now} tokens): the follow-up conversation can continue")
    return self._json({"ok": True})

def h_items_rerun(self, body):
    """Reset every finished/failed/stopped item on one route so it runs again (with the current settings)."""
    root = self._root(body)
    route, list_key = body.get("route"), body.get("list", "chunks")
    if list_key not in ("chunks", "final_chunks"):
        return self._json({"error": "bad list"}, 400)
    st = get_state(root)
    n = 0
    with _lock:
        for c in st[list_key]:
            if (route in (None, "", c.get("route"))) and c["status"] in ("done", "error", "stopped"):
                c["status"], c["steps"] = "pending", []
                for k in ("result", "error", "notice", "stop_reason", "started", "finished", "result_tokens", "model"):
                    c.pop(k, None)
                n += 1
        save_state(st)
    return self._json({"ok": True, "reset": n})

def _start_run(self, root, mode):
    st = get_state(root)
    with _lock:
        if st["running"] or root in RUNS:
            return self._json({"error": "already running"}, 409)
        key = "chunks" if mode == "chunks" else "final_chunks"
        if not st[key]:
            return self._json({"error": "nothing built yet -- build " + ("chunks" if mode == "chunks" else "the final chunk(s)") + " first"}, 400)
        RUNS[root] = RunCtl()
    threading.Thread(target=worker, args=(root, mode), daemon=True).start()
    return self._json({"ok": True})

def h_run_start(self, body):
    return _start_run(self, self._root(body), "chunks")

def h_final_run_start(self, body):
    return _start_run(self, self._root(body), "final")

def h_run_stop(self, body):
    """mode: 'now' (stop mid-output) or 'after' (finish the current item first)."""
    root = self._root(body)
    ctl = RUNS.get(root)
    if not ctl:
        return self._json({"ok": True, "note": "not running"})
    if body.get("mode") == "after":
        ctl.stop_after = True
    else:
        ctl.cancel_now = True
        threading.Thread(target=nexus_stop, args=(ctl.cfg_active,), daemon=True).start()
    return self._json({"ok": True})

def h_build_final(self, body):
    root = self._root(body)
    st = get_state(root)
    with _lock:
        if st["running"]:
            return self._json({"error": "a run is in progress"}, 409)
    cfg = mcfg(st, "final")
    try:
        nexus_info(cfg, timeout=4)
    except NexusError:
        cfg = None      # not up: pack with a rough estimate (exact counts when the auto final pass builds it)
    fc, not_done = build_final(root, st, cfg)
    return self._json({"final_chunks": len(fc), "chunks_not_done": not_done, "exact": cfg is not None})

def h_save_create(self, body):
    return self._json(create_save(self._root(body), body.get("name", "")))

def h_save_delete(self, body):
    data_p, meta_p = save_paths(body.get("id"))
    for p in (data_p, meta_p):
        if os.path.isfile(p):
            os.remove(p)
    return self._json({"ok": True})

POST_ROUTES = {
    "/api/project": h_project,
    "/api/project/delete": h_project_delete,
    "/api/settings": h_settings,
    "/api/model/test": h_model_test,
    "/api/nexus/dir": h_nexus_dir,
    "/api/model/start": h_model_start,
    "/api/model/stop": h_model_stop,
    "/api/scan": h_scan,
    "/api/oversized_decision": h_oversized_decision,
    "/api/build_chunks": h_build_chunks,
    "/api/item": h_item,
    "/api/run/start": h_run_start,
    "/api/run/stop": h_run_stop,
    "/api/final/build": h_build_final,
    "/api/final/run/start": h_final_run_start,
    "/api/saves/create": h_save_create,
    "/api/saves/delete": h_save_delete,
    "/api/defaults/apply": h_defaults_apply,
    "/api/items/rerun": h_items_rerun,
    "/api/followup/send": h_followup_send,
    "/api/followup/stop": h_followup_stop,
    "/api/followup/save": h_followup_save,
    "/api/followup/resume": h_followup_resume,
    "/api/prompt_sets/save": h_prompt_set_save,
    "/api/prompt_sets/delete": h_prompt_set_delete,
}

def main():
    port = 8787
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
    ALLOWED_HOSTS.update({f"127.0.0.1:{port}", f"localhost:{port}"})
    if os.name != "nt":
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))   # so atexit stops any model we started
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print(f"[harden] serving http://127.0.0.1:{port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_all_managed()

if __name__ == "__main__":
    main()
