#define NEXUS_VULKAN
#include "../src/gpu/vk_loader.h"
int main(){
    namespace v = nexus::gpu::vkl;
    bool ok = v::load();
    printf("load=%d lib=%s\n", ok, v::lib_path().c_str());
    if(!ok){ printf("err: %s\n", v::error().c_str()); return 0; }
    uint32_t n=0;
    VkResult r = vkEnumerateInstanceExtensionProperties(nullptr,&n,nullptr);  // via macro
    printf("ext r=%d n=%u\n",(int)r,n);
    VkInstanceCreateInfo ci{}; ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkInstance inst=VK_NULL_HANDLE;
    r=vkCreateInstance(&ci,nullptr,&inst);
    printf("createInstance r=%d\n",(int)r);
    if(r==VK_SUCCESS){ uint32_t d=0; vkEnumeratePhysicalDevices(inst,&d,nullptr); printf("devices=%u\n",d); vkDestroyInstance(inst,nullptr);}
}
