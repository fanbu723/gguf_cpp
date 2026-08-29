/*
gguf_messages:

头文件: 24byte
gguf magic number: 4byte
gguf version: 4byte
tensor_count: 8byte
metadata_kv_count: 8byte

元数据 KV:
metadata_kv_count:

张量信息表:
tensor_count

*/

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

// 强制 1 字节对齐，防止编译器在结构体内部插入填充字节
#pragma pack(push, 1)
struct GGUFHeader {
    uint32_t magic;             // 魔数，应该是 'GGUF' (0x46554747)
    uint32_t version;           // 版本号，目前是 3
    uint64_t tensor_count;      // 张量数量
    uint64_t metadata_kv_count; // 元数据键值对数量
    // 后面紧跟着 metadata_kv_count 个键值对
    // 然后是 tensor_count 个张量信息
};
#pragma pack(pop)

// 静态断言确保大小符合你的描述：4+4+8+8 = 24 字节
static_assert(sizeof(GGUFHeader) == 24, "GGUFHeader size must be 24 bytes");

class GGUFLoader {
  public:
    /**
     * @brief 读取 GGUF 文件头
     * @param filepath 文件路径
     * @param header 输出参数，存储读取到的头部信息
     * @return 成功返回 true，失败返回 false
     */
    static bool load_header(const fs::path &filepath, GGUFHeader &header);

    /**
     * @brief 检查文件是否为有效的 GGUF 文件
     * @param filepath 文件路径
     * @return 是 GGUF 文件返回 true，否则返回 false
     */
    static bool is_gguf_file(const fs::path &filepath);

  private:
    /**
     * @brief 验证魔数是否正确
     */
    static bool validate_magic(uint32_t magic);
};
