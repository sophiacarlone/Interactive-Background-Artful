#ifndef ENGINE_IMPL_H
#define ENGINE_IMPL_H

// @todo all this support query shit seems to significantly increase the startup time
// @todo eventually implement the debug log system starting at page 52 of the Vulkan Tutorial PDF

// glfw
#define GLFW_INCLUDE_VULKAN // causes GLFW to load the vulkan header
#include <GLFW/glfw3.h>
// #include <vulkan/vulkan.h> // ^^ GLFW already loads this

// glm
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>

// VulkanMemoryAllocator
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

// C++ standard library
#include <iostream>
#include <stdexcept>
#include <cstdlib> // EXIT_SUCCESS, EXIT_FAILURE
#include <vector>
#include <cstring> // strcmp
#include <optional>
#include <algorithm> // max_element
#include <set>
#include <limits>
#include<fstream>
#include <array>
#include <functional>

using std::vector, std::cout, std::runtime_error;

namespace engine_impl {

// vertex buffer gonna be AoS
struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    // @todo ... what's a binding?
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bd{};

        bd.binding = 0;
        bd.stride = sizeof(Vertex);
        bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // not using instanced rendering

        return bd;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> ad{};

        // positions
        ad[0].binding = 0;
        ad[0].location = 0;
        ad[0].format = VK_FORMAT_R32G32_SFLOAT; // vertex format (2 32-bit signed floats)
        ad[0].offset = offsetof(Vertex, pos);

        // colors
        ad[1].binding = 0;
        ad[1].location = 1;
        ad[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        ad[1].offset = offsetof(Vertex, color);

        return ad;
    }
};

// CONSTANTS -------------------------------------------------------------------------------------------------

// window dimensions
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// shaders
const char* VERT_SHADER_SPIRV_FILE = "vert.spv";
const char* FRAG_SHADER_SPIRV_FILE = "frag.spv";

// extensions
const vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME // probably equivalent to "VK_KHR_swapchain"
};

// validation layers
const vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation" // this is a big bundle of validation layers
};
#ifdef NDEBUG
    const bool ENABLE_VALIDATION_LAYERS = false;
#else
    const bool ENABLE_VALIDATION_LAYERS = true;
#endif

// vertices
const vector<Vertex> VERTICES = {
    { { 0.0f, -0.1f}, {1.0f, 0.0f, 0.0f} },
    { { 0.1f,  0.1f}, {0.0f, 1.0f, 0.0f} },
    { {-0.1f,  0.1f}, {0.0f, 0.0f, 1.0f} }
};

// -----------------------------------------------------------------------------------------------------------

struct AllocatedBuffer {
    VkBuffer buffer; // this doesn't allocate memory, but gets bound to memory allocated by VMA
    VmaAllocation allocation; // the VMA allocation information
};

// @todo understand and write here the purpose of this data structure
struct QueueFamilyIndices {
    // graphics and present queues usually end up being the same, but it's not guaranteed
    std::optional<uint32_t> graphicsFamily; // capable of executing a graphics pipeline
    std::optional<uint32_t> presentFamily;  // capable of presenting to a surface

    bool isComplete() {
        return graphicsFamily.has_value() && \
               presentFamily.has_value();
    }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    vector<VkSurfaceFormatKHR> formats; // supported color formats; see VkSurfaceFormatKHR definition
    vector<VkPresentModeKHR> presentModes;
};

class Engine {
public:
    void run(std::function<void()> mainLoopCallback);
    void update_position(glm::vec2 new_pos) { position_ = new_pos; }

private:
    GLFWwindow* window_;
    VkInstance instance_;
    VkSurfaceKHR surface_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    VkQueue graphicsQueue_;
    VkQueue presentQueue_;
    // swapchain stuff
    VkSwapchainKHR swapchain_;
    vector<VkImage> swapchainImages_;
    VkExtent2D swapchainExtent_;
    VkFormat swapchainImageFormat_;
    vector<VkImageView> swapchainImageViews_; // image views contain information on how to interpret an image
    // graphics pipeline stuff
    VkRenderPass renderPass_;
    VkPipelineLayout pipelineLayout_;
    VkPipeline graphicsPipeline_;
    vector<VkFramebuffer> swapchainFramebuffers_;
    // command stuff
    VkCommandPool commandPool_; // a command pool manages memory for command buffers
    VkCommandBuffer commandBuffer_; // a command buffer containing commands is submitted to a device queue
    // syncronization
    VkSemaphore imageAvailableSemaphore_;
    VkSemaphore renderFinishedSemaphore_;
    VkFence inFlightFence_; // @todo rename? The meaning "GPU operations are in flight" isn't obvious
    // buffers
    // The VMA library provides an allocator to manage memory allocation for us; we create it once,
    // then use it for all allocation calls. It does things like keeping one big VkBufferMemory and using
    // offsets in it for individual buffers.
    VmaAllocator bufferAllocator_;
    AllocatedBuffer vertexBuffer_;
    // world state (i.e. states of objects in the virtual world)
    glm::vec2 position_;

    // functions called by run()
    void initWindow();
    void initVulkan();
    void mainLoop(std::function<void()> callback);
    void cleanup();

    // physical device querying
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice);
    bool isDeviceSuitable(VkPhysicalDevice);
    int rateDevice(VkPhysicalDevice);
    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice);
    
    // helpers for initVulkan
    void createInstance();
    void createSurface();
    void selectPhysicalDevice();
    void createLogicalDeviceAndQueues();
    void createSwapchain();
    void createSwapchainImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createBufferAllocator();
    void createVertexBuffer();
    void allocateCommandBuffer();
    void createSyncObjects();
    void initWorldState();

    // helpers for cleanup
    void destroySwapchainImageViews();
    void destroyFramebuffers();
    void destroySyncObjects();

    // choosing swapchain settings
    // chooseSwapSurfaceFormat and chooseSwapPresentMode are standalone functions
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR&);

    // misc
    VkShaderModule createShaderModule(const vector<char>& spirv);
    void recordCommandBuffer(VkCommandBuffer, uint32_t imageIndex);
    void drawFrame();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

void Engine::run(std::function<void()> mainLoopCallback) {
    initWindow();
    initVulkan();
    mainLoop(mainLoopCallback);
    cleanup();
}

void Engine::drawFrame() {
    // Commmon steps to render a frame (from tutorial p136 "Outline of a frame"):
    //     wait for prev frame to finish
    //     acquire image from swapchain
    //     record a cmd buf to draw to image
    //     submit cmd buf to queue
    //     present swapchain image
    // Syncronization primitives:
    //     Semaphore (specifically, binary semaphore): blocks GPU but not CPU
    //         Semaphore S
    //         Operation A told to "signal" S
    //         Operation B told to "wait" for S
    //         A is called and runs
    //         B is called but waits for S to be signalled
    //         A finishes and signals S
    //         B runs
    //         B completes and resets ("unsignals") S
    //         S can now be reused
    //     Fence: makes CPU wait for a GPU operation to finish
    //         Doesn't automatically become unsignalled; if we want it reset, must do so manually
    
    // wait for GPU operations from the previous frame to complete
    vkWaitForFences(device_, 1, &inFlightFence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &inFlightFence_); // fences don't auto-reset

    // get an image from the swapchain
    uint32_t imageInd; // the output into this variable is the index in swapchainImages_
    vkAcquireNextImageKHR(
        device_, swapchain_, UINT64_MAX, imageAvailableSemaphore_, VK_NULL_HANDLE, &imageInd
    );

    // record command buffer
    vkResetCommandBuffer(commandBuffer_, 0);
    recordCommandBuffer(commandBuffer_, imageInd);

    // submit command buffer to queue
    VkSubmitInfo sInfo{};
    sInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // don't write the color attachment output until the swapchain image is available
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore_};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    sInfo.waitSemaphoreCount = 1;
    sInfo.pWaitSemaphores = waitSemaphores;
    sInfo.pWaitDstStageMask = waitStages;
    // cmd bufs to submit
    sInfo.commandBufferCount = 1;
    sInfo.pCommandBuffers = &commandBuffer_; // tutorial doesn't use a pointer, but I think that's a typo
    // semaphores to signal after devices finishes executing the commands in the buffer
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore_};
    sInfo.signalSemaphoreCount = 1;
    sInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue_, 1, &sInfo, inFlightFence_) != VK_SUCCESS) {
        throw runtime_error("failed to submit draw command buffer");
    }

    // configure presentation
    VkPresentInfoKHR pInfo{};
    pInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pInfo.waitSemaphoreCount = 1;
    pInfo.pWaitSemaphores = signalSemaphores; // wait for rendering to complete
    // swapchains to which to present, and the images to present
    VkSwapchainKHR swapchains[] = {swapchain_};
    pInfo.swapchainCount = 1;
    pInfo.pSwapchains = swapchains;
    pInfo.pImageIndices = &imageInd;
    pInfo.pResults = nullptr; // can use this to check every swapchain for successful presentation

    // present image to swapchain!
    vkQueuePresentKHR(presentQueue_, &pInfo);
}

void Engine::mainLoop(std::function<void()> callbackFunc) {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        //callbackFunc provides an interface (for whoever is using the engine) to update the world state
        callbackFunc();

        drawFrame();
    }

    // don't exit mainLoop until GPU operations complete; else we may begin cleanup prematurely
    vkDeviceWaitIdle(device_);
}

vector<char> readSpirvFile(const std::string& fname) {
    using std::ios, std::ifstream;

    // "ate" = start at end of file; it's a hack to get filesize
    ifstream file(fname, ios::ate | ios::binary);
    if (!file.is_open()) throw runtime_error("failed to open SPIRV file");
    size_t fileSize = (size_t) file.tellg();

    vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    return buffer;
}

void Engine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // don't use OpenGL (glfw does by default)
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // @todo enable later

    window_ = glfwCreateWindow(WIDTH, HEIGHT, "TITLE", nullptr, nullptr);
}

// @unused
void printAvailableInstanceExtensions() {
    uint32_t extensionCount = 0;
    // get count so we can allocate an array for the details
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    vector<VkExtensionProperties> extensions(extensionCount);
    // second param can't be nullptr /shrug
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
    cout << "available extensions:\n";
    for (const auto& extension : extensions) cout << "    " << extension.extensionName << '\n';

    // @note this pattern is the typical way to ask vulkan to enumerate something:
    // get count
    // allocate array
    // populate array
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    // get list of supported extensions
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    vector<VkExtensionProperties> supportedExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, supportedExtensions.data());

    // check that every required extension is in the list of supported extensions
    for (const char* requiredExtension : DEVICE_EXTENSIONS) {
        bool extensionFound = false;
        for (const auto& supportedExtension : supportedExtensions) {
            if (strcmp(requiredExtension, supportedExtension.extensionName) == 0) {
                extensionFound = true;
                break;
            }
        }
        if (!extensionFound) return false;
    }
    return true;
}

bool checkValidationLayerSupport() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : VALIDATION_LAYERS) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }

    return true;
}

void Engine::createInstance() {
    if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
        throw runtime_error("a requested validation layer is unavailable");
    }

    // application info isn't required but may enable optimizations by the graphics driver
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // no shit, but supposedly required by Vulkan
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.pEngineName = "No Engine";
    appInfo.apiVersion = VK_API_VERSION_1_0;
    // tutorial lists other initializations but I see no point

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    // glfwGetRequiredInstanceExtensions should include the required surface creation extensions
    createInfo.ppEnabledExtensionNames = glfwGetRequiredInstanceExtensions(&createInfo.enabledExtensionCount);
    if (ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount = VALIDATION_LAYERS.size();
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        throw runtime_error("failed to create vulkan instance");
    }
}

QueueFamilyIndices Engine::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
    
    for (int i = 0; i < queueFamilyCount; i++) {
        // graphics capability
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;
        // present capability
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
        if (presentSupport) indices.presentFamily = i;

        if (indices.isComplete()) break;
    }

    return indices;
}

// criteria:
// support graphics queues
// support present queues
// support all DEVICE_EXTENSIONS
// physical device swapchain supports window surface
bool Engine::isDeviceSuitable(VkPhysicalDevice physDevice) {
    // queues support
    QueueFamilyIndices indices = findQueueFamilies(physDevice);
    if (!indices.isComplete()) return false;

    // extensions support
    if (!checkDeviceExtensionSupport(physDevice)) return false;

    // window surface support
    SwapchainSupportDetails swapChainSupport = querySwapchainSupport(physDevice);
    // @todo why aren't we checking swapChainSupport.capabilities?
    if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty()) return false;

    return true;
}
int Engine::rateDevice(VkPhysicalDevice physDevice) {
    if (!isDeviceSuitable(physDevice)) return 0;
    int rating = 1; // "suitable"

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physDevice, &deviceProperties);
    // VkPhysicalDeviceFeatures deviceFeatures;
    // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    // this is the only rating criterion for now (apart from boolean suitability)
    // @todo hmm, Intel claims to be a discrete GPU so this doesn't help
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) rating += 1;

    return rating;
}

void Engine::createSurface() {
    // glfwCreateWindowSurface handles the platform-specific stuff for us
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
        throw runtime_error("failed to create window surface");
    }
}

// check if physical device is compatible with the window surface
// @todo need explanation of why we need to know the capabilities, formats, modes, etc.
SwapchainSupportDetails Engine::querySwapchainSupport(VkPhysicalDevice physDevice) {
    SwapchainSupportDetails supportDetails;

    // supported surface capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, surface_, &supportDetails.capabilities);
    // supported formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface_, &formatCount, nullptr);
    if (formatCount != 0) {
        supportDetails.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            physDevice, surface_, &formatCount, supportDetails.formats.data()
        );
    }
    // supported present modes
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, surface_, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        supportDetails.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            physDevice, surface_, &presentModeCount, supportDetails.presentModes.data()
        );
    }

    return supportDetails;
}

// choose surface format for swapchain
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const vector<VkSurfaceFormatKHR>& availableFormats) {
    // choose sRGB because it's "standard" for e.g. textures
    for (const VkSurfaceFormatKHR& format : availableFormats) {
        if (
            format.format     == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        ) return format;
    }
    // tutorial says that if the above fails, we can just pick whatever
    // but that seems dumb as hell to me, because it could result in hard-to-trace incorrect display colors
    throw runtime_error("failed to find appropriate surface format");
}

VkPresentModeKHR chooseSwapPresentMode(const vector<VkPresentModeKHR>& availablePresentModes) {
    for (const VkPresentModeKHR& presentMode : availablePresentModes) {
        // vsynced single-image queue; if a new image is ready but the queue is full, the image in the queue
        // is replaced.
        // @todo fast but may be energy-intensive; maybe allow user to select a less intensive mode
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) return presentMode;
    }

    // using this one as backup because Vulkan-compliant devices must support it
    return VK_PRESENT_MODE_FIFO_KHR; // vsynced FIFO queue (which blocks when full?)
}

// "swap extent" means "swapchain image resolution"
VkExtent2D Engine::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps) {
    // The max uint32_t is a special constant meaning "match the window resolution" (supposedly only works
    // with some window managers).

    // If not "match resolution", change nothing; else try to match resolution.
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return caps.currentExtent;
    }
    else {
        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);

        VkExtent2D newExt = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
        
        // make sure extent is within supported bounds
        newExt.width  = std::clamp(newExt.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        newExt.height = std::clamp(newExt.height, caps.minImageExtent.height, caps.maxImageExtent.height);

        return newExt;
    }
}

void Engine::createSwapchain() {
    SwapchainSupportDetails scSupport = querySwapchainSupport(physicalDevice_);
    
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(scSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(scSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(scSupport.capabilities);

    // Pick the number of images in the swapchain.
    // We select the minimum allowed + 1 (+1 supposedly recommended so we don't have to wait for the driver to
    // release an image back to use when we want to render to one).
    uint32_t requestedImageCount = scSupport.capabilities.minImageCount + 1;
    // don't exceed the max. A max of 0 indicates there's no max
    uint32_t maxImageCount = scSupport.capabilities.maxImageCount;
    if (maxImageCount > 0 && requestedImageCount > maxImageCount) requestedImageCount = maxImageCount;

    // fill the createInfo
    VkSwapchainCreateInfoKHR cInfo{};
    cInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    cInfo.surface = surface_;
    cInfo.minImageCount = requestedImageCount; // the vulkan implementation might create more
    cInfo.imageFormat = surfaceFormat.format;
    cInfo.imageColorSpace = surfaceFormat.colorSpace;
    cInfo.imageExtent = extent;
    cInfo.imageArrayLayers = 1; // a 2D image is one layer, regardless of how many color channels it has
    // Color attachment bit indicates we're gonna render directly to a swapchain image. In other scenarios
    // we could, e.g., render to another image for post-processing and then memory-transfer it over into a
    // swapchain image.
    cInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // if the graphics and present queue families are different, they need to share swapchain images
    QueueFamilyIndices qInds = findQueueFamilies(physicalDevice_); // @todo why do we keep doing this?
    uint32_t qFamInds[] = {qInds.graphicsFamily.value(), qInds.presentFamily.value()};
    if (qInds.graphicsFamily != qInds.presentFamily) {
        cInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        cInfo.queueFamilyIndexCount = 2;
        cInfo.pQueueFamilyIndices = qFamInds;
    }
    else {
        cInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        // we don't have to specify the following for exclusive mode, so we don't
        cInfo.queueFamilyIndexCount = 0;
        cInfo.pQueueFamilyIndices = nullptr;
    }

    cInfo.preTransform = scSupport.capabilities.currentTransform; // i.e. don't change anything
    cInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // don't allow window transparency
    cInfo.presentMode = presentMode;
    cInfo.clipped = VK_TRUE; // we don't care about pixels behind our window
    // @todo come back to oldSwapchain later. We'll probably need to recreate it if we allow window resizing
    cInfo.oldSwapchain = VK_NULL_HANDLE; // only used if the swapchain is being recreated

    // create the swapchain; fucking finally
    if (vkCreateSwapchainKHR(device_, &cInfo, nullptr, &swapchain_) != VK_SUCCESS) {
        throw runtime_error("failed to create swapchain");
    }

    // save handles to the swapchain images
    uint32_t trueImageCount;
    if (vkGetSwapchainImagesKHR(device_, swapchain_, &trueImageCount, nullptr) != VK_SUCCESS) {
        throw runtime_error("failed to get number of swapchain images");
    }
    swapchainImages_.resize(trueImageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &trueImageCount, swapchainImages_.data());

    // save extent and surface format; not used yet but might be later
    swapchainExtent_ = extent;
    swapchainImageFormat_ = surfaceFormat.format;
}

void Engine::createSwapchainImageViews() {
    swapchainImageViews_.resize(swapchainImages_.size());

    for (size_t i = 0; i < swapchainImages_.size(); ++i) {
        VkImageViewCreateInfo cInfo{};
        cInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        cInfo.image = swapchainImages_[i];
        cInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        cInfo.format = swapchainImageFormat_;
        // I think "swizzling" here refers to changing which component each component name actually refers to
        cInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        cInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        cInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        cInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        // subresourceRange: something something the purpose of the image and which part will be accessed
        // I don't understand most of this
        cInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        cInfo.subresourceRange.baseMipLevel = 0;
        cInfo.subresourceRange.levelCount = 1;
        cInfo.subresourceRange.baseArrayLayer = 0;
        cInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device_, &cInfo, nullptr, &swapchainImageViews_[i]) != VK_SUCCESS) {
            throw runtime_error("failed to create image views");
        };
    }
}
void Engine::destroySwapchainImageViews() {
    for (auto imageView : swapchainImageViews_) {
        vkDestroyImageView(device_, imageView, nullptr);
    }
}

VkShaderModule Engine::createShaderModule(const vector<char>& spirv) {
    VkShaderModuleCreateInfo cInfo{};
    cInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    cInfo.codeSize = spirv.size();
    // we don't have to worry about data alignment """in practice""" because, in most implementations, the
    // default allocator used by std::vector (and new and malloc) aligns to the largest primitve datatype.
    // C++ is fucking cursed.
    // ... @todo what effect does the potential uninitialized data at the of the array have?
    // @todo PLEASE do this in a way that doesn't rely on the implementation
    cInfo.pCode = reinterpret_cast<const uint32_t*>(spirv.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device_, &cInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw runtime_error("failed to create shader module");
    }
    return shaderModule;
}

void Engine::createRenderPass() {
    // @todo I don't really understand the attachment thing.
    // I think an "attachment" is just an image used in a framebuffer
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainImageFormat_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // because we're not multisampling (yet)
    // these apply to color and depth data
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear before the render pass
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // this applies to stenctil data; we're not using stencil for now
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // image layout (initial = what to expect, final = what to change it to after pass)
    // @todo ?? something something initial layout depends on various factors, so say it isn't defined. This
    // might not preserve the initial contents of the image, but we don't need them
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // "to be presented in swapchain"

    // A render pass consists of subpasses. Each subpass needs a reference to a color attachment (huh? I still
    // don't know what a color attachment is).
    // we're only using one subpass
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // @todo comment, losing patience
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // i.e. not compute
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef; // this is what `layout(location = 0)` refers to

    // the operations at the start and end of the render pass are implicit subpasses
    // @todo something something syncronization, revisit tutorial p144 to understand
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL; // the implicit subpass at the start of the render pass
    dep.dstSubpass = 0; // index of the explicit subpass we specified above
    // operations whose completion to wait for
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // wait for this before accessing image?
    dep.srcAccessMask = 0;
    // operations which should wait for completion of the subpasses?
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // @todo ????
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // @todo ?

    // create render pass
    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dep;

    if (vkCreateRenderPass(device_, &rpInfo, nullptr, &renderPass_) != VK_SUCCESS) {
        throw runtime_error("failed to create render pass");
    }
}

void Engine::createGraphicsPipeline() {
    // read spirv code
    vector<char> vertShaderCode = readSpirvFile(VERT_SHADER_SPIRV_FILE);
    vector<char> fragShaderCode = readSpirvFile(FRAG_SHADER_SPIRV_FILE);

    // create shader modules
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    // create programmable pipeline stages -------------------------------------------------------------------

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    // specify entrypoint; we could have multiple potential entrypoints in the shader, so we pick one here
    vertShaderStageInfo.pName = "main";
    // note: ...Info.pSPecializationInfo can be used to specify constants used in the shader at pipeline
    // creation time, which could be more efficient than pushing them during the render loop

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    // @todo why are we putting these in an array?
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // create fixed pipeline stages --------------------------------------------------------------------------

    // info on how vertices enter the pipeline
    VkVertexInputBindingDescription bindingDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();
    //
    VkPipelineVertexInputStateCreateInfo vertInputInfo{};
    vertInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertInputInfo.vertexBindingDescriptionCount = 1;
    vertInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
    vertInputInfo.pVertexAttributeDescriptions = attrDesc.data();

    // info on how to assemble primitives out of the vertices
    // i.e. specifies the assembly stage settings (I think), which is part of the rasterization stage
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
    inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // what the vertices represent
    // primitiveRestart enables fancy primitive specification using special indices. Look it up for details
    inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

    // viewport = the subset of the framebuffer to which to render
    // see picture examples in the tutorial, section "Viewports and Scissors"
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    // we're using the swapchain images as the framebuffers, so:
    viewport.width  = (float) swapchainExtent_.width;
    viewport.height = (float) swapchainExtent_.height;
    // @todo ?? what are minDepth and maxDepth?
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 0.0f;

    // everything outside the scissor rectangle is discarded (guessing it just shows up black)
    // We're not trying to only show a strict subregion of the framebuffer, so here scissor=viewport
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapchainExtent_;

    // create viewport state info (i.e. viewport and scissor info)
    VkPipelineViewportStateCreateInfo viewportStateInfo{};
    viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateInfo.viewportCount = 1;
    viewportStateInfo.pViewports = &viewport;
    viewportStateInfo.scissorCount = 1;
    viewportStateInfo.pScissors = &scissor;

    // @todo comment
    VkPipelineRasterizationStateCreateInfo rastInfo{};
    rastInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    // depthClampEnable=true would clamp primitives with out-of-bounds (defined by the near and far planes)
    // depth instead of discarding them
    rastInfo.depthClampEnable = VK_FALSE;
    // rasterizerDiscardEnable disables ("discards") the rasterization stage. By setting it to false, we
    // enable the rasterizer. Idk why someone would want to disable it
    rastInfo.rasterizerDiscardEnable = VK_FALSE;
    rastInfo.polygonMode = VK_POLYGON_MODE_FILL; // fill polygons / only draw edges / only draw vertices
    rastInfo.lineWidth = 1.0f; // width of lines (measured in number of fragments (i.e. "pixels"))
    // whether to cull "front-facing" or "back-facing" faces, or none at all
    rastInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    // defines direction a face "faces" by the order of vertices defining the polygon
    rastInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rastInfo.depthBiasEnable = VK_FALSE; // whether to modify depth values during rasterization
    // rastInfo.depthBiasConstantFactor = 
    // rastInfo.depthBiasClamp = 
    // rastInfo.depthBiasSlopeFactor = 

    // multisampling (an antialiasing method)
    // disabled for now
    VkPipelineMultisampleStateCreateInfo msInfo{};
    msInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msInfo.sampleShadingEnable = VK_FALSE;
    msInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    msInfo.minSampleShading = 1.0f;
    msInfo.pSampleMask = nullptr;
    msInfo.alphaToCoverageEnable = VK_FALSE;
    msInfo.alphaToOneEnable = VK_FALSE;

    // can also configure depth and stencil testing here (tutorial p113), but won't for now
    // VkPipelineDepthStencilStateCreateInfo dsInfo{};

    // color blending; i.e. how to blend new color from frag shader with color that was already in framebuf.
    // VkPipelineColorBlendAttachmentState configures blending per framebuffer;
    // VkPipelineCOlorBlendStateCreateInfo configures blending globally
    // @todo no, I don't actually understand the interplay between the two
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_FALSE;
    // not sure if this one is optional when blending is disabled
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | \
                         VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT; 
    // the rest of these are optional since blending is disabled above
    // color blending options
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    // alpha blending options
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;

    // @todo understand this. Skipped understanding due to lack of patience
    VkPipelineColorBlendStateCreateInfo cbInfo{};
    cbInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbInfo.logicOpEnable = VK_FALSE;
    cbInfo.logicOp = VK_LOGIC_OP_COPY;
    cbInfo.attachmentCount = 1;
    cbInfo.pAttachments = &cba;
    cbInfo.blendConstants[0] = 0.0f;
    cbInfo.blendConstants[1] = 0.0f;
    cbInfo.blendConstants[2] = 0.0f;
    cbInfo.blendConstants[3] = 0.0f;
    
    // @todo skipped Dynamic State (tutorial page 116); implementing it is probably necessary for e.g.
    // resizing window

    // push constants
    VkPushConstantRange pcRange;
    pcRange.offset = 0;
    pcRange.size = sizeof(glm::mat4);
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // only accessed by vertex shader

    // pipeline layout (something something descriptors, uniforms, push constants)
    VkPipelineLayoutCreateInfo plLayoutInfo{};
    plLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plLayoutInfo.setLayoutCount = 0;
    plLayoutInfo.pSetLayouts = nullptr;
    plLayoutInfo.pushConstantRangeCount = 1;
    plLayoutInfo.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(device_, &plLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        throw runtime_error("failed to create pipeline layout");
    }

    // finally, create the fucking pipeline
    VkGraphicsPipelineCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    plInfo.stageCount = 2;
    plInfo.pStages = shaderStages;
    plInfo.pVertexInputState = &vertInputInfo;
    plInfo.pInputAssemblyState = &inputAssemblyInfo;
    plInfo.pViewportState = &viewportStateInfo;
    plInfo.pRasterizationState = &rastInfo;
    plInfo.pMultisampleState = &msInfo;
    plInfo.pDepthStencilState = nullptr; // not using stencil (yet?)
    plInfo.pColorBlendState = &cbInfo;
    plInfo.pDynamicState = nullptr;
    plInfo.layout = pipelineLayout_;
    plInfo.renderPass = renderPass_;
    plInfo.subpass = 0;
    // it's less expensive to derive from an existing pipeline if functionality is similar
    // not doing that here since this is the first pipeline we're creating
    plInfo.basePipelineHandle = VK_NULL_HANDLE;
    plInfo.basePipelineIndex = -1;

    // second param is an optional pipeline cache, which can speed up multiple pipeline creation calls
    if (
        vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &plInfo, nullptr, &graphicsPipeline_)
        != VK_SUCCESS
    ) throw runtime_error("failed to create graphics pipeline");

    // we don't need the shader modules after linking; clean them up
    vkDestroyShaderModule(device_, vertShaderModule, nullptr);
    vkDestroyShaderModule(device_, fragShaderModule, nullptr);
}

void Engine::createFramebuffers() {
    swapchainFramebuffers_ = vector<VkFramebuffer>(swapchainImageViews_.size());

    // create a framebuffer for every image view
    for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
        VkImageView attachments[] = { swapchainImageViews_[i] }; // @todo what?

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = renderPass_; // framebuffer needs to be compatible with this render pass
        fbInfo.attachmentCount = 1;
        // must be in same order as layout descriptors; we use 1 attachment per framebuf so doesn't matter
        fbInfo.pAttachments = attachments;
        fbInfo.width  = swapchainExtent_.width;
        fbInfo.height = swapchainExtent_.height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device_, &fbInfo, nullptr, &swapchainFramebuffers_[i]) != VK_SUCCESS) {
            throw runtime_error("failed to create framebuffer");
        }
    }
}
void Engine::destroyFramebuffers() {
    for (VkFramebuffer fb : swapchainFramebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
}

void Engine::createCommandPool() {
    QueueFamilyIndices qfInds = findQueueFamilies(physicalDevice_); // @todo doing this yet again

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // allow rerecording of individual buffers (instead of all at once); we'll be doing so every frame
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // the queue family to which command buffers will be submitted
    poolInfo.queueFamilyIndex = qfInds.graphicsFamily.value(); // because we're going to submit draw commands

    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        throw runtime_error("failed to create command pool");
    }
}

// typeFilter is a bitmask specifying acceptable memory types
// this function returns a memory of some type in typeFilter, satisfying all `properties`
// @todo this ignores heap types, which can significantly affect performance
uint32_t Engine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProp);
    
    for (uint32_t i = 0; i < memProp.memoryTypeCount; ++i) {
        if (
            (typeFilter & (1 << i)) && // @todo ?? acceptable type?
            ((memProp.memoryTypes[i].propertyFlags & properties) == properties) // has all properties
        ) return i;
    }
    throw runtime_error("failed to find a suitable memory type");
}

void Engine::createBufferAllocator() {
    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.physicalDevice = physicalDevice_;
    allocInfo.device = device_;
    allocInfo.instance = instance_;
    if (vmaCreateAllocator(&allocInfo, &bufferAllocator_) != VK_SUCCESS) {
        throw runtime_error("failed to create buffer allocator");
    }
}

void Engine::createVertexBuffer() {
    // buffer info
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sizeof(VERTICES[0]) * VERTICES.size();
    bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // only used by graphics queue

    // allocation info
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    // we'll be using memcpy to write to the buffer, so SEQUENTIAL_WRITE_BIT is appropriate
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    // @todo not sure if I need to set these two; Vulkan Tutorial sets them, but isn't using VMA.
    // HOST_COHERENT_BIT makes it so that we don't need to explicitly flush writes to the mapped memory
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // create the VkBuffer, allocate the VkDeviceMemory, and bind them together
    // these are normally separate steps, but VMA handles all of them in one call
    if (
        vmaCreateBuffer(
            bufferAllocator_, &bufInfo, &allocInfo, &vertexBuffer_.buffer, &vertexBuffer_.allocation, nullptr
        ) != VK_SUCCESS
    ) throw runtime_error("allocator failed to allocate and bind vertex buffer");

    // copy vertices over to buffer
    void* data; // becomes a pointer to the mapped memory
    if (vmaMapMemory(bufferAllocator_, vertexBuffer_.allocation, &data) != VK_SUCCESS) {
        throw runtime_error("failed to map vertex buffer memory");
    }
    memcpy(data, VERTICES.data(), bufInfo.size);
    // don't need to explicitly flush writes to the memory here as long as we have set
    // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    vmaUnmapMemory(bufferAllocator_, vertexBuffer_.allocation);
    // at this point, the driver knows about the writes but the memory may not have been copied to the GPU
    // yet; but Vulkan guarantees it will have been completely copied before the next call to vkQueueSubmit
    // goes through
}

void Engine::allocateCommandBuffer() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // secondary buffers can be built from primary ones?
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer_) != VK_SUCCESS) {
        throw runtime_error("failed to allocate command buffer");
    }
}

void Engine::recordCommandBuffer(VkCommandBuffer cbuf, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // none of the flags needed yet
    beginInfo.pInheritanceInfo = nullptr; // only relevant for secondary command bufs

    if (vkBeginCommandBuffer(cbuf, &beginInfo) != VK_SUCCESS) {
        throw runtime_error("failed to begin recording command buffer");
    }

    // set up info needed for the "begin render pass" command
    VkRenderPassBeginInfo rpbInfo{};
    rpbInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbInfo.renderPass = renderPass_;
    rpbInfo.framebuffer = swapchainFramebuffers_[imageIndex];
    rpbInfo.renderArea.offset = {0, 0};
    rpbInfo.renderArea.extent = swapchainExtent_;
    //
    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}}; // @todo what's with all the braces?
    rpbInfo.clearValueCount = 1;
    rpbInfo.pClearValues = &clearColor;

    // record commands
    // VK_SUBPASS_CONTENTS_INLINE = no need to think about secondary cmd bufs
    vkCmdBeginRenderPass(cbuf, &rpbInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);
    // bind vertex buffer
    VkBuffer vertexBuffers[] = {vertexBuffer_.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cbuf, 0, 1, vertexBuffers, offsets);
    glm::mat4 posTransform = glm::translate(glm::identity<glm::mat4>(), glm::vec3(position_, 0.0));
    vkCmdPushConstants(
        cbuf, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(posTransform), &posTransform
    );
    // instanceCount = 1 because we're not doing instancing
    vkCmdDraw(cbuf, static_cast<uint32_t>(VERTICES.size()), 1, 0, 0);
    vkCmdEndRenderPass(cbuf);

    if (vkEndCommandBuffer(cbuf) != VK_SUCCESS) throw runtime_error("failed to end command buffer recording");
}

void Engine::createSyncObjects() {
    VkSemaphoreCreateInfo sInfo{};
    sInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    // no interesting information needed here

    VkFenceCreateInfo fInfo{};
    fInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // start the fence signaled so the CPU doesn't deadlock by waiting for the GPU to finish rendering before
    // the first frame
    fInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkResult r1 = vkCreateSemaphore(device_, &sInfo, nullptr, &imageAvailableSemaphore_);
    VkResult r2 = vkCreateSemaphore(device_, &sInfo, nullptr, &renderFinishedSemaphore_);
    VkResult r3 = vkCreateFence(    device_, &fInfo, nullptr, &inFlightFence_);
    if (r1 != VK_SUCCESS || r2 != VK_SUCCESS || r3 != VK_SUCCESS) {
        throw runtime_error("failed to create sync objects");
    }
}
void Engine::destroySyncObjects() {
    vkDestroySemaphore(device_, imageAvailableSemaphore_, nullptr);
    vkDestroySemaphore(device_, renderFinishedSemaphore_, nullptr);
    vkDestroyFence(    device_, inFlightFence_,           nullptr);
}

void Engine::initWorldState() {
    // initialize position transform as the identity transform of a 2D point
    position_ = glm::vec2(0.0);
}

void Engine::selectPhysicalDevice() {
    physicalDevice_ = VK_NULL_HANDLE;

    // get device list
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) throw runtime_error("found 0 physical devices");
    vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    // rate devices
    vector<int> deviceRatings(deviceCount);
    for (int i = 0; i < deviceCount; i++) deviceRatings[i] = rateDevice(devices[i]);

    // get best device
    auto bestRatingIterator = max_element(deviceRatings.begin(), deviceRatings.end());
    // guaranteed to not be end() because deviceCount > 0
    if (*bestRatingIterator == 0) throw runtime_error("found no suitable physical device");
    physicalDevice_ = devices[bestRatingIterator - deviceRatings.begin()];

    // print chosen device
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice_, &deviceProperties);
    #ifdef DEBUG
        cout << "chose " << deviceProperties.deviceName << '\n';
    #endif
}

void Engine::createLogicalDeviceAndQueues() {
    // set up queue creation information
    //
    // @todo we do this queue family search multiple times. No big deal but could save results the first time
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
    // the (e.g.) present and graphics queue families might end up being the same family, in which case we
    // don't want to give vkCreateDevice two queue creation infos when we only need one queue
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    // Vulkan requires priority specification regardless of how many queues there are
    float queuePriority = 1.0f;
    // for each unique queue family, make an info for creation of a queue from the family
    vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // create logical device
    //
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // queues
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    // features, extensions, validation layers
    VkPhysicalDeviceFeatures deviceFeatures{}; // not enabling any features for now
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
    deviceCreateInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    if (ENABLE_VALIDATION_LAYERS) {
        deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        deviceCreateInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }
    else deviceCreateInfo.enabledLayerCount = 0;
    // create device (this also creates the specified queues)
    if (vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_) != VK_SUCCESS) {
        throw runtime_error("failed to create logical device");
    }
    // get handles to the created queue(s). Note they could be handles to the same queue.
    // 3rd parameter is the index of queue to get from the family; since we only made 1 queue per family, the
    // index is 0.
    vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, indices.presentFamily.value(),  0, &presentQueue_ );
}

void Engine::initVulkan() {
    createInstance(); cout << "created instance\n";
    // should be after instance, before physical device (can affect phys dev selection)
    createSurface();                cout << "created surface\n";
    selectPhysicalDevice();         cout << "selected physical device\n";
    createLogicalDeviceAndQueues(); cout << "created logical device\n";
    createSwapchain();              cout << "created swapchain\n";
    createSwapchainImageViews();    cout << "created swapchain image views\n";
    createRenderPass();             cout << "created render pass\n";
    createGraphicsPipeline();       cout << "created graphics pipeline\n";
    createFramebuffers();           cout << "created framebuffers\n";
    createCommandPool();            cout << "created command pool\n";
    createBufferAllocator();        cout << "created buffer allocator\n";
    createVertexBuffer();           cout << "created vertex buffer\n";
    allocateCommandBuffer();        cout << "allocated command buffer\n";
    createSyncObjects();            cout << "created sync objects\n";
    initWorldState();               cout << "initialized push constants\n";
}

void Engine::cleanup() {
    // destroy things in reverse order of creation
    destroySyncObjects();
    vmaDestroyBuffer(bufferAllocator_, vertexBuffer_.buffer, vertexBuffer_.allocation);
    vmaDestroyAllocator(bufferAllocator_);
    vkDestroyCommandPool(device_, commandPool_, nullptr);
    destroyFramebuffers();
    vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
    vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    vkDestroyRenderPass(device_, renderPass_, nullptr);
    destroySwapchainImageViews();
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    vkDestroyDevice(device_, nullptr); // implicitly destroys associated queues
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);
    glfwDestroyWindow(window_); // do we need this line? glfwTerminate claims to destroy open windows
    glfwTerminate();
}

} // namespace engine_impl

#endif // ENGINE_IMPL_H