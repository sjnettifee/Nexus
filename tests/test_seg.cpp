#include <string>
#include <vector>
#include <map>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <algorithm>
struct ChatMessage { std::string role, content; };
struct ContextTurn { int start=0,end=0; std::string role; bool protected_part=false; std::string preview; };
struct Sec { int start_token, end_token; bool pinned; };
struct MockEngine {
    std::string fmt; std::vector<std::string> specials;
    std::string format_user_turn(const std::string& u, int) const {
        if (fmt=="chatml") return "<|im_start|>user\n"+u+"<|im_end|>\n<|im_start|>assistant\n";
        if (fmt=="llama3") return "<|start_header_id|>user<|end_header_id|>\n\n"+u+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        if (fmt=="gemma") return "<start_of_turn>user\n"+u+"<end_of_turn>\n<start_of_turn>model\n";
        if (fmt=="mistral") return "[INST] "+u+" [/INST]";
        if (fmt=="phi3") return "<|user|>\n"+u+"<|end|>\n<|assistant|>\n";
        return "User: "+u+"\n\nAssistant:";
    }
    std::string format_system_prefix(const std::vector<ChatMessage>& m) const {
        const std::string& x=m[0].content;
        if (fmt=="chatml") return "<|im_start|>system\n"+x+"<|im_end|>\n";
        if (fmt=="llama3") return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+x+"<|eot_id|>";
        if (fmt=="gemma") return "<bos><start_of_turn>user\n"+x+"\n\n";
        if (fmt=="mistral") return "<s>[INST] "+x+"\n\n";
        if (fmt=="phi3") return "<|system|>\n"+x+"<|end|>\n";
        return x+"\n\n";
    }
    std::string assistant_end() const {
        if (fmt=="chatml") return "<|im_end|>\n"; if (fmt=="llama3") return "<|eot_id|>"; if (fmt=="gemma") return "<end_of_turn>\n";
        if (fmt=="mistral") return "</s>"; if (fmt=="phi3") return "<|end|>\n"; return "\n\n"; }
    std::vector<int32_t> tokenize(const std::string& t, bool=false) const {
        std::vector<int32_t> o; size_t i=0;
        while (i<t.size()) { bool hit=false;
            for (size_t k=0;k<specials.size();++k) if (t.compare(i,specials[k].size(),specials[k])==0) { o.push_back((int32_t)k); i+=specials[k].size(); hit=true; break; }
            if (!hit) o.push_back(1000+(unsigned char)t[i++]); }
        return o; }
    std::string detokenize(int32_t id) const { if (id<1000) return specials[(size_t)id]; return std::string(1,(char)(id-1000)); }
    size_t vocab_size() const { return 2000; }
};
using Engine = MockEngine;
namespace nexus { using ::ChatMessage; }
#include "seg_gen.inc"
int main(){
    int fails=0;
    std::map<std::string,std::vector<std::string>> sp={
        {"chatml",{"<|im_start|>","<|im_end|>"}}, {"llama3",{"<|begin_of_text|>","<|start_header_id|>","<|end_header_id|>","<|eot_id|>"}},
        {"gemma",{"<bos>","<start_of_turn>","<end_of_turn>"}}, {"mistral",{"<s>","</s>","[INST]","[/INST]"}},
        {"phi3",{"<|system|>","<|user|>","<|assistant|>","<|end|>"}}, {"deepseek",{}}};
    for (auto& [f,s] : sp) {
        MockEngine e; e.fmt=f; e.specials=s;
        std::string conv = e.format_system_prefix({{"system","Be terse."}});
        int pin = (int)e.tokenize(conv).size();
        bool merged = (f=="gemma"||f=="mistral");
        for (int k=1;k<=3;++k) {
            std::string ut = e.format_user_turn("question "+std::to_string(k), -1);
            if (merged && k==1) { size_t m=ut.find("question"); ut = ut.substr(m); }   // the system prefix already opened the first user turn
            conv += ut + "answer " + std::to_string(k) + e.assistant_end();
        }
        auto toks=e.tokenize(conv);
        auto T=turns_of(e,toks,pin);
        printf("== %-8s (%zu tokens)\n", f.c_str(), toks.size());
        for (auto& t: T) printf("   [%4d,%4d) %-9s %s  \"%s\"\n", t.start,t.end,t.role.c_str(), t.protected_part?"PROT":"    ", t.preview.c_str());
        int nu=0,na=0,nx=0; for(auto&t:T){ nu+=t.role=="user"; na+=t.role=="assistant"; nx+=t.role=="exchange"; }
        bool ok = f=="deepseek" ? (T.size()==1) : f=="mistral" ? (nx==3) : (nu==3&&na==3) || (merged && nu+na>=5);
        if (T.size()>1 && !T[0].protected_part) ok=false;
        if (!ok){ ++fails; printf("   ^^^ UNEXPECTED\n"); }
    }
    printf(fails?"%d FAIL\n":"ALL OK\n",fails); return fails;
}
