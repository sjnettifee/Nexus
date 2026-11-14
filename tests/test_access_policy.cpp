#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include <algorithm>
struct Req{std::string method,path,body;std::unordered_map<std::string,std::string>headers;bool valid=true;};
struct ServerState{ int cors_mode=0; std::vector<std::string> allow_origins; std::string api_key; };
static thread_local std::string tl_acao;
#include "policy.inc"
int fails=0;
void T(const char* name, ServerState S, std::string method, std::string path, std::unordered_map<std::string,std::string> h, std::string peer, int want){
  Req r; r.method=method; r.path=path; r.headers=h; std::string why; int got=nx_access_check(S,r,peer,why);
  printf("%s %-62s -> %d%s%s\n", got==want?"ok  ":"FAIL", name, got, why.empty()?"":"  ("+why.substr(0,60)+")" , got==0&&!tl_acao.empty()?(" ["+tl_acao.substr(0,tl_acao.find('\r'))+"]").c_str():"");
  if(got!=want) ++fails;
}
int main(){
  ServerState D; ServerState K; K.api_key="s3cret"; ServerState L; L.cors_mode=1; ServerState A; A.cors_mode=2;
  ServerState W; W.allow_origins={"https://myapp.example.com"};
  T("launcher on A (program, loopback)", D, "GET","/api/info",{{"host","127.0.0.1:8080"}},"127.0.0.1",0);
  T("hardener on B -> A (program, LAN, no key set)", D, "POST","/api/chat",{{"host","192.168.1.10:8080"}},"192.168.1.20",0);
  T("Nexus page opened from B (same origin)", D, "POST","/api/chat",{{"host","192.168.1.10:8080"},{"origin","http://192.168.1.10:8080"},{"sec-fetch-site","same-origin"}},"192.168.1.20",0);
  T("advanced UI page on B -> A and C (LAN origin)", D, "POST","/api/chat",{{"host","192.168.1.10:8080"},{"origin","http://192.168.1.20:3000"},{"sec-fetch-site","cross-site"}},"192.168.1.20",0);
  T("dev UI on localhost:5173 -> local Nexus", D, "POST","/api/chat",{{"host","127.0.0.1:8080"},{"origin","http://localhost:5173"}},"127.0.0.1",0);
  T("page on computer name 'desktop'", D, "GET","/api/info",{{"host","desktop:8080"},{"origin","http://laptop:8000"}},"192.168.1.30",0);
  T("Tailscale 100.x origin", D, "GET","/api/info",{{"host","100.101.1.2:8080"},{"origin","http://100.88.3.4:8000"}},"100.88.3.4",0);
  T("internet website fetch -> local Nexus", D, "POST","/api/kv/remove",{{"host","127.0.0.1:8080"},{"origin","https://evil.example"},{"sec-fetch-site","cross-site"}},"127.0.0.1",403);
  T("internet website <img> GET (no Origin)", D, "GET","/api/search_detail_by_url?url=x",{{"host","127.0.0.1:8080"},{"sec-fetch-site","cross-site"},{"referer","https://evil.example/page"}},"127.0.0.1",403);
  T("DNS rebinding (evil name -> 127.0.0.1, looks same-origin)", D, "POST","/api/chat",{{"host","rebind.evil.example:8080"},{"origin","http://rebind.evil.example:8080"}},"127.0.0.1",403);
  T("public-IP page (internet host) -> Nexus", D, "POST","/api/chat",{{"host","192.168.1.10:8080"},{"origin","http://203.0.113.9"}},"192.168.1.20",403);
  T("Nexus reached via public IP, its own page", D, "POST","/api/chat",{{"host","203.0.113.5:8080"},{"origin","http://203.0.113.5:8080"}},"198.51.100.7",0);
  T("allow-origin listed site", W, "POST","/api/chat",{{"host","192.168.1.10:8080"},{"origin","https://myapp.example.com"}},"192.168.1.20",0);
  T("local mode: LAN page refused", L, "POST","/api/chat",{{"host","192.168.1.10:8080"},{"origin","http://192.168.1.20:3000"}},"192.168.1.20",403);
  T("any mode: internet page allowed", A, "POST","/api/chat",{{"host","127.0.0.1:8080"},{"origin","https://evil.example"}},"127.0.0.1",0);
  T("key set: program on B without key", K, "POST","/api/chat",{{"host","192.168.1.10:8080"}},"192.168.1.20",401);
  T("key set: program on B with Bearer", K, "POST","/api/chat",{{"host","192.168.1.10:8080"},{"authorization","Bearer s3cret"}},"192.168.1.20",0);
  T("key set: X-API-Key header", K, "GET","/api/info",{{"host","192.168.1.10:8080"},{"x-api-key","s3cret"}},"192.168.1.20",0);
  T("key set: ?key= in URL", K, "GET","/api/export?id=3&key=s3cret",{{"host","192.168.1.10:8080"}},"192.168.1.20",0);
  T("key set: wrong key", K, "GET","/api/info",{{"host","192.168.1.10:8080"},{"x-api-key","nope"}},"192.168.1.20",401);
  T("key set: this computer needs no key", K, "POST","/api/chat",{{"host","127.0.0.1:8080"}},"127.0.0.1",0);
  T("key set: page itself (/) loads without key", K, "GET","/",{{"host","192.168.1.10:8080"},{"sec-fetch-site","none"}},"192.168.1.20",0);
  T("key set: preflight OPTIONS without key", K, "OPTIONS","/api/chat",{{"host","192.168.1.10:8080"},{"origin","http://192.168.1.20:3000"}},"192.168.1.20",0);
  printf(fails?"%d FAIL\n":"ALL OK\n",fails); return fails; }
