#include "engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <netinet/in.h>
#include <regex>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <arpa/inet.h>
#include <fcntl.h>

namespace fs = std::filesystem;

struct Req {
    bool valid = false;
    std::string method, path, body;
    std::unordered_map<std::string,std::string> headers;
};

static std::string trim(const std::string& s){
    size_t a=0,b=s.size();
    while(a<b && std::isspace((unsigned char)s[a])) ++a;
    while(b>a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a,b-a);
}

static std::string now_iso(){
    std::time_t t=std::time(nullptr);
    char buf[64];
    std::tm tm{};
    localtime_r(&t,&tm);
    std::strftime(buf,sizeof(buf),"%Y-%m-%dT%H:%M:%S",&tm);
    return buf;
}

static std::string gen_id(){
    auto n = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream o; o << std::hex << (unsigned long long)n;
    return o.str();
}

static std::string jesc(const std::string& s){
    std::string e;
    e.reserve(s.size()+16);
    for(unsigned char c: s){
        if(c=='"') e += "\\\"";
        else if(c=='\\') e += "\\\\";
        else if(c=='\n') e += "\\n";
        else if(c=='\r') e += "\\r";
        else if(c=='\t') e += "\\t";
        else if(c < 0x20){ char b[8]; std::snprintf(b,sizeof(b),"\\u%04x", c); e += b; }
        else e.push_back((char)c);
    }
    return e;
}

static std::string jstr(const std::string& s){ return std::string("\"") + jesc(s) + "\""; }

static void send_all(int fd, const std::string& s){
    const char* p = s.c_str(); size_t n=s.size();
    while(n){ ssize_t k = ::send(fd,p,n,MSG_NOSIGNAL); if(k<=0) return; p+=k; n-=k; }
}

static void resp_json(int fd,const std::string& body,int code=200,const char* extra=""){
    std::string st = code==200?"200 OK":(code==404?"404 Not Found":(code==500?"500 Internal Server Error":"400 Bad Request"));
    send_all(fd,"HTTP/1.1 "+st+"\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\n"+std::string(extra)+"\r\n"+body);
}

static void resp_html(int fd,const std::string& body){
    send_all(fd,"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"+body);
}

static Req parse_req(int fd){
    Req r; std::string data; data.reserve(8192);
    char buf[4096];
    size_t header_end = std::string::npos;
    while((header_end = data.find("\r\n\r\n")) == std::string::npos){
        ssize_t n = recv(fd,buf,sizeof(buf),0);
        if(n<=0) return r;
        data.append(buf,n);
        if(data.size() > 8*1024*1024) return r;
    }
    std::string head = data.substr(0, header_end);
    std::istringstream hs(head);
    std::string line;
    if(!std::getline(hs,line)) return r;
    if(!line.empty() && line.back()=='\r') line.pop_back();
    std::istringstream ls(line);
    ls >> r.method >> r.path;
    while(std::getline(hs,line)){
        if(!line.empty() && line.back()=='\r') line.pop_back();
        size_t c = line.find(':');
        if(c==std::string::npos) continue;
        std::string k=line.substr(0,c), v=trim(line.substr(c+1));
        std::transform(k.begin(),k.end(),k.begin(),::tolower);
        r.headers[k]=v;
    }
    size_t cl = 0;
    auto it = r.headers.find("content-length");
    if(it!=r.headers.end()) cl = (size_t)std::strtoull(it->second.c_str(), nullptr, 10);
    r.body = data.substr(header_end+4);
    while(r.body.size() < cl){
        ssize_t n = recv(fd,buf,sizeof(buf),0);
        if(n<=0) break;
        r.body.append(buf,n);
    }
    r.valid = !r.method.empty();
    return r;
}

static std::string jp_str(const std::string& j, const std::string& key, const std::string& def=""){
    std::regex rx("\\\""+key+"\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"");
    std::smatch m; if(!std::regex_search(j,m,rx)) return def;
    std::string s = m[1].str(), out;
    out.reserve(s.size());
    for(size_t i=0;i<s.size();++i){
        if(s[i]=='\\' && i+1<s.size()){
            char c=s[++i];
            if(c=='n') out+='\n'; else if(c=='r') out+='\r'; else if(c=='t') out+='\t'; else out+=c;
        } else out+=s[i];
    }
    return out;
}

static double jp_num(const std::string& j, const std::string& key, double def=0){
    std::regex rx("\\\""+key+"\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    std::smatch m; if(!std::regex_search(j,m,rx)) return def;
    return std::strtod(m[1].str().c_str(), nullptr);
}

static bool jp_bool(const std::string& j, const std::string& key, bool def=false){
    std::regex rx("\\\""+key+"\\\"\\s*:\\s*(true|false)");
    std::smatch m; if(!std::regex_search(j,m,rx)) return def;
    return m[1].str()=="true";
}

static std::vector<std::string> jp_str_array(const std::string& j, const std::string& key){
    std::vector<std::string> out;
    std::regex rx("\\\""+key+"\\\"\\s*:\\s*\\[(.*?)\\]", std::regex::icase);
    std::smatch m; if(!std::regex_search(j,m,rx)) return out;
    std::string inner = m[1].str();
    std::regex item("\\\"((?:\\\\.|[^\\\"])*)\\\"");
    auto beg = std::sregex_iterator(inner.begin(), inner.end(), item);
    auto end = std::sregex_iterator();
    for(auto it=beg; it!=end; ++it) out.push_back(jp_str(std::string("{\"x\":\"")+(*it)[1].str()+"\"}", "x", ""));
    return out;
}

static std::vector<nexus::ChatMessage> jp_messages(const std::string& j){
    std::vector<nexus::ChatMessage> out;
    size_t p = j.find("\"messages\"");
    if(p==std::string::npos) return out;
    p = j.find('[', p); if(p==std::string::npos) return out;
    int depth = 1; size_t i=p+1;
    size_t start = std::string::npos;
    bool in_str=false, esc=false;
    for(; i<j.size(); ++i){
        char c=j[i];
        if(in_str){ if(esc) esc=false; else if(c=='\\') esc=true; else if(c=='\"') in_str=false; continue; }
        if(c=='\"'){ in_str=true; continue; }
        if(c=='{'){ if(depth==1) start=i; ++depth; }
        else if(c=='}'){ --depth; if(depth==1 && start!=std::string::npos){
            std::string obj=j.substr(start, i-start+1);
            nexus::ChatMessage m; m.role=jp_str(obj,"role"); m.content=jp_str(obj,"content");
            if(!m.role.empty()) out.push_back(std::move(m));
            start=std::string::npos;
        }}
        else if(c==']' && depth==1) break;
    }
    return out;
}

static std::string shell_escape(const std::string& s){
    std::string out="'";
    for(char c: s){ if(c=='\'') out += "'\\''"; else out.push_back(c); }
    out += "'";
    return out;
}

struct URLParts{ std::string host, path; int port=80; bool ok=false; };
static URLParts parse_url(const std::string& u){
    URLParts p;
    if(u.rfind("http://",0)!=0) return p;
    std::string rest = u.substr(7);
    size_t slash = rest.find('/');
    std::string hp = slash==std::string::npos ? rest : rest.substr(0,slash);
    p.path = slash==std::string::npos ? "/" : rest.substr(slash);
    size_t colon = hp.find(':');
    p.host = colon==std::string::npos ? hp : hp.substr(0,colon);
    p.port = colon==std::string::npos ? 80 : std::atoi(hp.substr(colon+1).c_str());
    p.ok = !p.host.empty();
    return p;
}

static std::string http_post_raw(const std::string& url, const std::string& content_type, const std::string& body){
    URLParts u = parse_url(url);
    if(!u.ok) return "";
    int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if(s < 0) return "";
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_port=htons(u.port);
    if(::inet_pton(AF_INET, u.host.c_str(), &addr.sin_addr) != 1){ ::close(s); return ""; }
    if(::connect(s, (sockaddr*)&addr, sizeof(addr)) != 0){ ::close(s); return ""; }
    std::ostringstream req;
    req << "POST " << u.path << " HTTP/1.1\r\n"
        << "Host: " << u.host << ":" << u.port << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    std::string rs = req.str();
    send_all(s, rs);
    std::string resp; char buf[4096]; ssize_t n;
    while((n=::recv(s,buf,sizeof(buf),0))>0) resp.append(buf,n);
    ::close(s);
    size_t sep = resp.find("\r\n\r\n");
    return sep==std::string::npos ? std::string() : resp.substr(sep+4);
}

static std::string collect_helper_sse(const std::string& body){
    std::string out; std::istringstream ss(body); std::string line;
    while(std::getline(ss,line)){
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line.rfind("data:",0)!=0) continue;
        std::string d = trim(line.substr(5));
        if(d=="[DONE]") break;
        if(d.find("\"token\"") != std::string::npos) out += jp_str(d,"token","");
    }
    return out;
}

struct HelperEndpoint {
    std::string name;
    std::string url;
    bool enabled = true;
};

struct ConversationState {
    std::string conv_id;
    std::string last_prompt;
    std::string session_path;
    std::string updated_at;
};

struct ServerState {
    nexus::Engine engine;
    std::mutex gen_mu;
    std::atomic<bool> stop_flag{false};

    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_path;
    std::string chat_format;
    int threads = 1;
    int gpu_layers = 0;
    int gpu2_layers = 0;
    int gpu2_device = -1;

    float temperature = 0.2f;
    float top_p = 0.9f;
    float repeat_penalty = 1.1f;
    int top_k = 40;
    int max_tokens = 4096;

    std::string workspace_root = "./workspace";
    std::string sessions_root  = "./live_sessions";
    std::vector<HelperEndpoint> helpers;
    std::vector<std::string> enabled_tools{"web_search","list_files","read_file","write_file","make_dir","delete_path","run_python","run_shell"};

    std::mutex conv_mu;
    std::unordered_map<std::string,ConversationState> conversations;
};

static std::string path_join_under(const std::string& root, const std::string& rel){
    fs::path base = fs::weakly_canonical(fs::path(root));
    fs::path want = fs::weakly_canonical(base / fs::path(rel).relative_path());
    auto bs = base.string();
    auto ws = want.string();
    if(ws.rfind(bs,0) != 0) throw std::runtime_error("path escapes workspace root");
    return ws;
}

static std::string slurp_file(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("open failed");
    std::ostringstream o; o << f.rdbuf(); return o.str();
}

static void write_file_full(const std::string& path, const std::string& data){
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("write failed");
    f << data;
}

static std::string run_cmd_capture(const std::string& cmd){
    FILE* p = popen(cmd.c_str(), "r");
    if(!p) return "[command failed to start]";
    char buf[4096]; std::string out;
    while(fgets(buf,sizeof(buf),p)) out += buf;
    int rc = pclose(p);
    out += "\n[exit=" + std::to_string(rc) + "]";
    return out;
}

static std::string get_tag_attr(const std::string& tag_content, const std::string& attr){
    std::regex rx(attr + R"(\s*=\s*\"([^\"]*)\")");
    std::smatch m; return std::regex_search(tag_content,m,rx) ? m[1].str() : "";
}

static bool tool_allowed(const ServerState& S, const std::string& name){
    return std::find(S.enabled_tools.begin(), S.enabled_tools.end(), name) != S.enabled_tools.end();
}

static std::string render_tree(const fs::path& p, int depth=0, int max_depth=3){
    if(depth > max_depth) return "";
    std::string out;
    std::vector<fs::directory_entry> ents;
    for(auto& e : fs::directory_iterator(p)) ents.push_back(e);
    std::sort(ents.begin(), ents.end(), [](auto&a, auto&b){ return a.path().filename().string() < b.path().filename().string(); });
    for(auto& e : ents){
        out += std::string(depth*2, ' ');
        out += e.is_directory() ? "[D] " : "[F] ";
        out += e.path().filename().string() + "\n";
        if(e.is_directory()) out += render_tree(e.path(), depth+1, max_depth);
    }
    return out;
}

static std::string do_web_search(const std::string& q){
    std::string cmd = "python3 - <<'PY'\n"
                      "import urllib.parse, urllib.request, re, html\n"
                      "q=" + std::string("\"") + jesc(q) + "\"\n"
                      "u='https://duckduckgo.com/html/?q='+urllib.parse.quote(q)\n"
                      "req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})\n"
                      "txt=urllib.request.urlopen(req,timeout=15).read().decode('utf-8','replace')\n"
                      "items=[]\n"
                      "for m in re.finditer(r'<a[^>]*class=\"result__a\"[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>', txt, re.S):\n"
                      "    href=html.unescape(m.group(1))\n"
                      "    title=re.sub('<[^>]+>','',m.group(2))\n"
                      "    items.append((title.strip(),href))\n"
                      "    if len(items)>=6: break\n"
                      "print('WEB SEARCH RESULTS')\n"
                      "for i,(t,h) in enumerate(items,1): print(f'{i}. {t}\\n   {h}')\n"
                      "PY";
    return run_cmd_capture(cmd);
}

static std::string execute_tool(ServerState& S, const std::string& tool_name, const std::string& input){
    if(!tool_allowed(S, tool_name)) return "[tool disabled: " + tool_name + "]";
    try {
        if(tool_name == "web_search") return do_web_search(input);
        if(tool_name == "list_files") {
            fs::create_directories(S.workspace_root);
            return render_tree(fs::path(S.workspace_root));
        }
        if(tool_name == "read_file") {
            std::string p = path_join_under(S.workspace_root, trim(input));
            return slurp_file(p);
        }
        if(tool_name == "write_file") {
            size_t nl = input.find('\n');
            if(nl == std::string::npos) return "[write_file expects first line = relative path, remaining lines = file content]";
            std::string rel = trim(input.substr(0,nl));
            std::string content = input.substr(nl+1);
            std::string p = path_join_under(S.workspace_root, rel);
            write_file_full(p, content);
            return "[wrote " + rel + "]";
        }
        if(tool_name == "make_dir") {
            std::string p = path_join_under(S.workspace_root, trim(input));
            fs::create_directories(p);
            return "[created directory " + trim(input) + "]";
        }
        if(tool_name == "delete_path") {
            std::string p = path_join_under(S.workspace_root, trim(input));
            std::uintmax_t n = fs::remove_all(p);
            return "[deleted " + trim(input) + " items=" + std::to_string((unsigned long long)n) + "]";
        }
        if(tool_name == "run_python") {
            fs::create_directories(S.workspace_root);
            std::string tmp = (fs::path(S.workspace_root) / "__tool_run__.py").string();
            write_file_full(tmp, input);
            std::string cmd = "cd " + shell_escape(S.workspace_root) + " && python3 " + shell_escape(tmp) + " 2>&1";
            return run_cmd_capture(cmd);
        }
        if(tool_name == "run_shell") {
            std::string cmd = "cd " + shell_escape(S.workspace_root) + " && bash -lc " + shell_escape(input) + " 2>&1";
            return run_cmd_capture(cmd);
        }
    } catch(const std::exception& e){
        return std::string("[tool error: ") + e.what() + "]";
    }
    return "[unknown tool: " + tool_name + "]";
}

static std::string helper_call(const HelperEndpoint& h, const std::string& prompt){
    std::string body = std::string("{\"messages\":[{\"role\":\"user\",\"content\":") + jstr(prompt) + "}],\"model\":\"1\"}";
    std::string raw = http_post_raw(h.url + "/api/chat", "application/json", body);
    return collect_helper_sse(raw);
}

static std::string live_system_banner(const ServerState& S){
    std::ostringstream o;
    o << "\nTOOLS AVAILABLE: use exactly these XML-style tags when you want a live tool.\n"
      << "<nexus:tool name=\"TOOL_NAME\">input</nexus:tool>\n"
      << "Available tools:\n";
    for(const auto& t : S.enabled_tools) o << "- " << t << "\n";
    if(!S.helpers.empty()){
        o << "HELPER MODELS:\n";
        for(const auto& h : S.helpers) if(h.enabled)
            o << "- " << h.name << " via <nexus:call model=\"" << h.name << "\">full prompt</nexus:call>\n";
    }
    o << "After a tool or helper result is injected, continue your answer naturally.\n";
    return o.str();
}

static void save_conv_checkpoint(ServerState& S, const std::string& conv_id, const std::string& prompt){
    fs::create_directories(S.sessions_root);
    std::string path = (fs::path(S.sessions_root) / conv_id).string();
    S.engine.save_session(path);
    std::lock_guard<std::mutex> lk(S.conv_mu);
    S.conversations[conv_id] = ConversationState{conv_id, prompt, path, now_iso()};
}

static bool load_conv_checkpoint(ServerState& S, const std::string& conv_id, ConversationState* out=nullptr){
    std::lock_guard<std::mutex> lk(S.conv_mu);
    auto it = S.conversations.find(conv_id);
    if(it == S.conversations.end()) return false;
    if(out) *out = it->second;
    return S.engine.load_session(it->second.session_path);
}

static const char* HTML = R"HTML(
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nexus Live</title>
<style>
:root{--bg:#12101a;--bg2:#171525;--bg3:#211d33;--tx:#e6e0ff;--mut:#9f93d3;--acc:#8b6df0;--acc2:#6d4fe0;--bd:#2c2741;--good:#5ed39a;--warn:#ffcc66}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#0f0d17,#151224 40%,#120f1d);color:var(--tx);font:14px/1.45 system-ui,Segoe UI,Roboto,Ubuntu,sans-serif;height:100vh;display:grid;grid-template-columns:320px 1fr}
#side{border-right:1px solid var(--bd);background:rgba(14,12,22,.95);padding:14px;overflow:auto}#main{display:grid;grid-template-rows:auto 1fr auto;min-width:0}
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:12px;padding:12px;margin-bottom:12px}.title{font-weight:700;margin-bottom:8px}input,textarea,button{font:inherit}input,textarea{width:100%;background:var(--bg3);border:1px solid var(--bd);color:var(--tx);border-radius:8px;padding:9px 10px}textarea{min-height:88px;resize:vertical}button{background:var(--acc);color:#fff;border:none;border-radius:8px;padding:9px 12px;cursor:pointer}button:hover{background:var(--acc2)}button.alt{background:transparent;border:1px solid var(--bd);color:var(--tx)}button.row{width:100%;margin-top:8px}
#top{padding:12px 16px;border-bottom:1px solid var(--bd);display:flex;gap:8px;align-items:center;background:rgba(20,17,32,.92)}#status{color:var(--mut);font-size:12px;margin-left:auto}
#conv{padding:18px;overflow:auto}.msg{max-width:900px;margin:0 auto 14px auto}.bubble{padding:12px 14px;border-radius:12px;border:1px solid var(--bd);white-space:pre-wrap}.user .bubble{background:#1d1830}.assistant .bubble{background:#171525}.event .bubble{background:#131d18;border-color:#224132;color:#bff0d5}.err .bubble{background:#25161a;border-color:#5d2b36;color:#ffb9c6}
#bar{display:flex;gap:10px;padding:12px 16px;border-top:1px solid var(--bd);background:rgba(20,17,32,.95)}#prompt{flex:1;min-height:72px;max-height:220px}.small{font-size:12px;color:var(--mut)}.tool-list label{display:flex;gap:8px;align-items:center;margin:6px 0}.helper{padding:8px;border:1px solid var(--bd);border-radius:10px;margin:8px 0}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
</style></head><body>
<div id="side">
  <div class="card"><div class="title">Connection</div>
    <div class="small">This server hosts the UI and the live tool loop itself.</div>
    <button class="row alt" onclick="refreshInfo()">Refresh info</button>
    <div id="info" class="small" style="margin-top:8px"></div>
  </div>
  <div class="card"><div class="title">Helpers</div>
    <div class="small">One per line: <span class="mono">name|http://host:port</span></div>
    <textarea id="helpers"></textarea>
    <button class="row" onclick="saveHelpers()">Save helpers</button>
  </div>
  <div class="card"><div class="title">Workspace</div>
    <input id="workspace" placeholder="./workspace">
    <div class="tool-list" id="tools"></div>
    <button class="row" onclick="saveConfig()">Save config</button>
  </div>
</div>
<div id="main">
  <div id="top"><strong>Nexus Live Server</strong><button class="alt" onclick="newChat()">New Chat</button><button class="alt" onclick="saveSession()">Save Session</button><button class="alt" onclick="loadSession()">Load Session</button><div id="status">idle</div></div>
  <div id="conv"></div>
  <div id="bar"><textarea id="prompt" placeholder="Send a message. Live tools and helper models will pause/resume in the same answer."></textarea><div style="display:flex;flex-direction:column;gap:8px"><button onclick="sendMsg()">Send</button><button class="alt" onclick="stopGen()">Stop</button></div></div>
</div>
<script>
let convId=''; let sending=false;
function add(role,text){const wrap=document.createElement('div');wrap.className='msg '+role;const b=document.createElement('div');b.className='bubble';b.textContent=text;wrap.appendChild(b);document.getElementById('conv').appendChild(wrap);document.getElementById('conv').scrollTop=1e9;return b;}
function setStatus(t){document.getElementById('status').textContent=t;}
async function refreshInfo(){const i=await fetch('/api/info').then(r=>r.json());document.getElementById('info').textContent=JSON.stringify(i,null,2);const c=await fetch('/api/config').then(r=>r.json());document.getElementById('workspace').value=c.workspace_root||'./workspace';document.getElementById('helpers').value=(c.helpers||[]).map(h=>`${h.name}|${h.url}|${h.enabled}`).join('\n');const tl=document.getElementById('tools');tl.innerHTML='';(c.all_tools||[]).forEach(t=>{const on=(c.enabled_tools||[]).indexOf(t)>=0;const lab=document.createElement('label');lab.innerHTML=`<input type="checkbox" value="${t}" ${on?'checked':''}> <span>${t}</span>`;tl.appendChild(lab);});}
async function saveHelpers(){const rows=document.getElementById('helpers').value.split(/\n+/).map(x=>x.trim()).filter(Boolean);const helpers=rows.map(r=>{const p=r.split('|');return {name:(p[0]||'').trim(),url:(p[1]||'').trim(),enabled:(p[2]||'true').trim()!=='false'};}).filter(x=>x.name&&x.url);await fetch('/api/helpers/set',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({helpers})});await refreshInfo();}
async function saveConfig(){const enabled_tools=[...document.querySelectorAll('#tools input:checked')].map(x=>x.value);await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({workspace_root:document.getElementById('workspace').value,enabled_tools})});await saveHelpers();}
async function newChat(){convId=''; await fetch('/api/reset',{method:'POST'}); document.getElementById('conv').innerHTML=''; setStatus('new chat');}
async function saveSession(){const p=prompt('Save path', 'manual_session'); if(!p) return; const r=await fetch('/api/session/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})}).then(r=>r.json()); setStatus(JSON.stringify(r));}
async function loadSession(){const p=prompt('Load path', 'manual_session'); if(!p) return; const r=await fetch('/api/session/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})}).then(r=>r.json()); setStatus(JSON.stringify(r));}
async function stopGen(){await fetch('/api/stop',{method:'POST'});}
async function sendMsg(){ if(sending) return; const ta=document.getElementById('prompt'); const text=ta.value.trim(); if(!text) return; ta.value=''; add('user', text); const out=add('assistant',''); sending=true; setStatus('streaming');
  const body={messages:[{role:'user',content:text}],conv_id:convId};
  const resp=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
  while(true){ const {value,done}=await reader.read(); if(done) break; buf += dec.decode(value,{stream:true}); let idx; while((idx=buf.indexOf('\n\n'))>=0){ const chunk=buf.slice(0,idx); buf=buf.slice(idx+2); const lines=chunk.split(/\n/).filter(x=>x.startsWith('data:')).map(x=>x.slice(5).trim()); for(const d of lines){ if(d==='[DONE]'){ sending=false; setStatus('done'); continue; } let obj={}; try{obj=JSON.parse(d);}catch{} if(obj.conv_id) convId=obj.conv_id; if(obj.token) out.textContent += obj.token; if(obj.event){ add(obj.event==='error'?'err':'event', (obj.name?obj.name+': ':'') + (obj.message||obj.input||obj.result||'')); } document.getElementById('conv').scrollTop=1e9; }} }
  sending=false; setStatus('idle'); }
refreshInfo();
</script></body></html>
)HTML";

static void sse_begin(int fd){
    send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nX-Accel-Buffering: no\r\nConnection: keep-alive\r\n\r\n");
}
static void sse_json(int fd, const std::string& payload){ send_all(fd, "data: " + payload + "\n\n"); }

static void handle_chat_sse(int fd, ServerState& S, const Req& r){
    auto messages = jp_messages(r.body);
    if(messages.empty()){ resp_json(fd,"{\"ok\":false,\"error\":\"no messages\"}"); return; }

    // Inject a live-tool banner into the first system slot or create one.
    bool has_system = false;
    for(auto& m : messages){ if(m.role == "system"){ m.content += "\n" + live_system_banner(S); has_system = true; break; } }
    if(!has_system){ nexus::ChatMessage sm; sm.role="system"; sm.content = live_system_banner(S); messages.insert(messages.begin(), sm); }

    std::string prompt = S.engine.format_chat(messages);
    std::string conv_id = jp_str(r.body, "conv_id", "");
    if(conv_id.empty()) conv_id = gen_id();

    nexus::GenerateConfig cfg;
    cfg.temperature = (float)jp_num(r.body, "temperature", S.temperature);
    cfg.top_p = (float)jp_num(r.body, "top_p", S.top_p);
    cfg.top_k = (int)jp_num(r.body, "top_k", S.top_k);
    cfg.repeat_penalty = (float)jp_num(r.body, "repeat_penalty", S.repeat_penalty);
    cfg.max_tokens = (int)jp_num(r.body, "max_tokens", S.max_tokens);
    cfg.n_gpu_layers = S.gpu_layers;
    cfg.stop_strings = S.engine.stop_strings();
    cfg.stop_strings.push_back("\nHuman:");
    cfg.stop_strings.push_back("\nUser:");

    std::unique_lock<std::mutex> lg(S.gen_mu);
    S.stop_flag = false;
    sse_begin(fd);
    sse_json(fd, std::string("{\"conv_id\":") + jstr(conv_id) + "}");

    try {
        bool continuation = false;
        std::string new_tail;
        ConversationState old{};
        if(load_conv_checkpoint(S, conv_id, &old)){
            if(!old.last_prompt.empty() && prompt.size() > old.last_prompt.size() && prompt.compare(0, old.last_prompt.size(), old.last_prompt) == 0){
                new_tail = prompt.substr(old.last_prompt.size());
                continuation = true;
            } else if(messages.size() == 2 && messages.back().role == "user"){
                std::vector<nexus::ChatMessage> only; only.push_back(messages.back());
                new_tail = S.engine.format_chat(only);
                continuation = true;
            }
        }

        nexus::GenerateSession sess = continuation
            ? [&](){ S.engine.extend_context(new_tail, false); return S.engine.begin_session_warm(cfg, {"nexus:tool","nexus:call"}); }()
            : S.engine.begin_session(prompt, cfg, {"nexus:tool","nexus:call"});

        while(true){
            auto chunk = sess.run([&](const std::string& tok, int32_t)->bool{
                if(S.stop_flag) return false;
                sse_json(fd, std::string("{\"token\":") + jstr(tok) + "}");
                return true;
            });

            if(chunk.paused){
                std::string result;
                if(chunk.tag_name == "nexus:tool"){
                    std::string name = get_tag_attr(chunk.tag_content, "name");
                    sse_json(fd, std::string("{\"event\":\"tool\",\"name\":") + jstr(name) + ",\"input\":" + jstr(chunk.tag_input) + "}");
                    result = execute_tool(S, name, chunk.tag_input);
                    sse_json(fd, std::string("{\"event\":\"tool_result\",\"name\":") + jstr(name) + ",\"result\":" + jstr(result) + "}");
                    sess.inject("\n<tool_result name=\"" + name + "\">\n" + result + "\n</tool_result>\n");
                } else if(chunk.tag_name == "nexus:call"){
                    std::string model = get_tag_attr(chunk.tag_content, "model");
                    auto it = std::find_if(S.helpers.begin(), S.helpers.end(), [&](const HelperEndpoint& h){ return h.name == model && h.enabled; });
                    if(it == S.helpers.end()) result = "[no helper model configured for '" + model + "']";
                    else result = helper_call(*it, chunk.tag_input);
                    sse_json(fd, std::string("{\"event\":\"helper_result\",\"name\":") + jstr(model) + ",\"result\":" + jstr(result) + "}");
                    sess.inject("\n<helper_result model=\"" + model + "\">\n" + result + "\n</helper_result>\n");
                } else {
                    result = "[unknown pause tag]";
                    sess.inject(result);
                }
                continue;
            }

            S.engine.sync_session_tokens(sess.tokens(), sess.pos());
            save_conv_checkpoint(S, conv_id, prompt);
            break;
        }
    } catch(const std::exception& e){
        sse_json(fd, std::string("{\"event\":\"error\",\"message\":") + jstr(e.what()) + "}");
    }

    sse_json(fd, "[DONE]");
}

static void handle(int fd, ServerState& S){
    Req r = parse_req(fd);
    if(!r.valid){ ::close(fd); return; }

    if(r.method=="OPTIONS"){
        send_all(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type\r\nAccess-Control-Allow-Methods: GET,POST,DELETE,OPTIONS\r\nConnection: close\r\n\r\n");
        ::close(fd); return;
    }

    try {
        if(r.method=="GET" && r.path=="/") { resp_html(fd, HTML); ::close(fd); return; }
        if(r.method=="GET" && r.path=="/api/info") {
            auto mi = S.engine.model_info();
            std::ostringstream o;
            o << "{"
              << "\"model_path\":" << jstr(S.model_path) << ","
              << "\"chat_format\":" << jstr(S.chat_format) << ","
              << "\"context_size\":" << S.engine.context_size() << ","
              << "\"max_context\":" << S.engine.max_context() << ","
              << "\"gpu_layers\":" << S.gpu_layers << ","
              << "\"threads\":" << S.threads
              << "}";
            resp_json(fd, o.str()); ::close(fd); return;
        }
        if(r.method=="GET" && r.path=="/api/config") {
            std::ostringstream o;
            o << "{"
              << "\"workspace_root\":" << jstr(S.workspace_root) << ","
              << "\"temperature\":" << S.temperature << ","
              << "\"top_p\":" << S.top_p << ","
              << "\"top_k\":" << S.top_k << ","
              << "\"repeat_penalty\":" << S.repeat_penalty << ","
              << "\"max_tokens\":" << S.max_tokens << ","
              << "\"enabled_tools\":[";
            for(size_t i=0;i<S.enabled_tools.size();++i){ if(i) o << ','; o << jstr(S.enabled_tools[i]); }
            o << "],\"all_tools\":[\"web_search\",\"list_files\",\"read_file\",\"write_file\",\"make_dir\",\"delete_path\",\"run_python\",\"run_shell\"],\"helpers\":[";
            for(size_t i=0;i<S.helpers.size();++i){ if(i) o << ','; o << "{\"name\":" << jstr(S.helpers[i].name) << ",\"url\":" << jstr(S.helpers[i].url) << ",\"enabled\":" << (S.helpers[i].enabled?"true":"false") << "}"; }
            o << "]}";
            resp_json(fd, o.str()); ::close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/config") {
            if(r.body.find("workspace_root") != std::string::npos) S.workspace_root = jp_str(r.body, "workspace_root", S.workspace_root);
            if(r.body.find("temperature") != std::string::npos) S.temperature = (float)jp_num(r.body, "temperature", S.temperature);
            if(r.body.find("top_p") != std::string::npos) S.top_p = (float)jp_num(r.body, "top_p", S.top_p);
            if(r.body.find("top_k") != std::string::npos) S.top_k = (int)jp_num(r.body, "top_k", S.top_k);
            if(r.body.find("repeat_penalty") != std::string::npos) S.repeat_penalty = (float)jp_num(r.body, "repeat_penalty", S.repeat_penalty);
            if(r.body.find("max_tokens") != std::string::npos) S.max_tokens = (int)jp_num(r.body, "max_tokens", S.max_tokens);
            auto arr = jp_str_array(r.body, "enabled_tools");
            if(!arr.empty()) S.enabled_tools = arr;
            fs::create_directories(S.workspace_root);
            resp_json(fd, "{\"ok\":true}"); ::close(fd); return;
        }
        if(r.method=="GET" && r.path=="/api/helpers"){
            std::ostringstream o; o << "{" << "\"helpers\":[";
            for(size_t i=0;i<S.helpers.size();++i){ if(i) o << ','; o << "{\"name\":" << jstr(S.helpers[i].name) << ",\"url\":" << jstr(S.helpers[i].url) << ",\"enabled\":" << (S.helpers[i].enabled?"true":"false") << "}"; }
            o << "]}"; resp_json(fd,o.str()); ::close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/helpers/set") {
            std::vector<HelperEndpoint> hs;
            std::regex item(R"(\{[^\}]*\"name\"\s*:\s*\"((?:\\.|[^\"])*)\"[^\}]*\"url\"\s*:\s*\"((?:\\.|[^\"])*)\"([^\}]*)\})");
            auto beg = std::sregex_iterator(r.body.begin(), r.body.end(), item), end=std::sregex_iterator();
            for(auto it=beg; it!=end; ++it){
                HelperEndpoint h; h.name=jp_str(std::string("{\"x\":\"")+(*it)[1].str()+"\"}","x",""); h.url=jp_str(std::string("{\"x\":\"")+(*it)[2].str()+"\"}","x",""); h.enabled = (*it)[3].str().find("false")==std::string::npos; if(!h.name.empty()&&!h.url.empty()) hs.push_back(h);
            }
            S.helpers = hs;
            resp_json(fd, "{\"ok\":true}"); ::close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/reset") {
            std::lock_guard<std::mutex> lk(S.gen_mu);
            S.engine.reset_context();
            resp_json(fd, "{\"ok\":true}"); ::close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/stop") {
            S.stop_flag = true;
            resp_json(fd, "{\"ok\":true}"); ::close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/session/save") {
            std::string path = jp_str(r.body, "path", "nexus_session");
            std::lock_guard<std::mutex> lk(S.gen_mu);
            bool ok = S.engine.save_session(path);
            resp_json(fd, ok ? std::string("{\"ok\":true,\"path\":") + jstr(path) + "}" : "{\"ok\":false,\"error\":\"save failed\"}");
            ::close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/session/load") {
            std::string path = jp_str(r.body, "path", "nexus_session");
            std::lock_guard<std::mutex> lk(S.gen_mu);
            bool ok = S.engine.load_session(path);
            resp_json(fd, ok ? std::string("{\"ok\":true,\"context_tokens\":") + std::to_string(S.engine.context_size()) + "}" : "{\"ok\":false,\"error\":\"load failed\"}");
            ::close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/chat") { handle_chat_sse(fd, S, r); ::close(fd); return; }

        resp_json(fd, "{\"error\":\"not found\"}", 404); ::close(fd); return;
    } catch(const std::exception& e){
        resp_json(fd, std::string("{\"ok\":false,\"error\":") + jstr(e.what()) + "}", 500);
        ::close(fd);
    }
}

int main(int argc, char** argv){
    ServerState S;
    if(argc < 2){
        std::fprintf(stderr, "usage: %s MODEL.gguf [port] [host]\n", argv[0]);
        return 1;
    }
    S.model_path = argv[1];
    if(argc > 2) S.port = std::atoi(argv[2]);
    if(argc > 3) S.host = argv[3];

    fs::create_directories(S.workspace_root);
    fs::create_directories(S.sessions_root);

    if(!S.engine.load(S.model_path, S.threads)){
        std::fprintf(stderr, "failed to load model: %s\n", S.model_path.c_str());
        return 1;
    }
    if(!S.chat_format.empty()) S.engine.override_chat_format(S.chat_format);
    if(S.gpu_layers != 0) S.engine.set_gpu_layers(S.gpu_layers);
    if(S.gpu2_layers > 0) S.engine.set_gpu2_layers(S.gpu2_layers, S.gpu2_device, S.gpu_layers > 0 ? S.gpu_layers : 0);

    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    int yes=1; setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_port=htons(S.port); addr.sin_addr.s_addr=inet_addr(S.host.c_str());
    if(::bind(srv, (sockaddr*)&addr, sizeof(addr)) != 0){ perror("bind"); return 1; }
    if(::listen(srv, 64) != 0){ perror("listen"); return 1; }

    std::fprintf(stderr, "nexus_live_server listening on http://%s:%d\n", S.host.c_str(), S.port);
    while(true){
        int fd = ::accept(srv, nullptr, nullptr);
        if(fd < 0) continue;
        std::thread([fd,&S]{ handle(fd,S); }).detach();
    }
}
