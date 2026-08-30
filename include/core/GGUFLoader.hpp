/*
 * ============================================================================
 * GGUF 文件格式规范 (v3)
 * ============================================================================
 *
 * 约定：所有多字节整数均为小端序（Little-Endian）。
 * 文件按逻辑顺序线性排列，解析器从头到尾依次读取四个区块。
 *
 * ──────────────────────────────────────────────────────────────────────────
 * ① 文件头（固定 24 字节，对应 GGUFHeader）
 * ──────────────────────────────────────────────────────────────────────────
 *   偏移  字段               类型        说明
 *   ────  ─────────────────  ──────────  ──────────────────────────────────
 *   0     magic              uint32_t    固定 0x46554747（ASCII "GGUF"）
 *   4     version            uint32_t    版本号，当前为 3
 *   8     tensor_count       uint64_t    张量总数
 *   16    metadata_kv_count  uint64_t    元数据键值对数量
 *
 * ──────────────────────────────────────────────────────────────────────────
 * ② 元数据键值对列表（重复 metadata_kv_count 次）
 * ──────────────────────────────────────────────────────────────────────────
 *   每项格式：key（字符串） + value_type（uint32_t） + value（变长）
 *
 *   字符串格式：uint64_t 长度（字节数） + UTF-8 内容（无结束符）
 *
 *   value_type 枚举（GGUFValueType）→ value 的编码：
 *     0  UINT8     → 1 字节              7  BOOL      → 1 字节 (0/1)
 *     1  INT8      → 1 字节              8  STRING    → 变长（同 key 格式）
 *     2  UINT16    → 2 字节              9  ARRAY     → uint32_t 元素类型
 *     3  INT16     → 2 字节                            + uint64_t 元素个数
 *     4  UINT32    → 4 字节                            + 元素序列（可递归）
 *     5  INT32     → 4 字节              10 UINT64    → 8 字节
 *     6  FLOAT32   → 4 字节              11 INT64     → 8 字节
 *                                        12 FLOAT64   → 8 字节
 *
 * ──────────────────────────────────────────────────────────────────────────
 * ③ 张量信息表（重复 tensor_count 次）
 * ──────────────────────────────────────────────────────────────────────────
 *   每项格式（依次）：
 *     name         字符串（长度前缀）
 *     n_dimensions uint32_t                  维度数量
 *     dimensions   uint64_t × n_dimensions   各维度大小
 *     data_type    uint32_t                  GGML 类型枚举（F32=0 / F16=1 / BF16=21 ...）
 *     offset       uint64_t                  张量数据在文件中的起始偏移（从文件头计）
 *
 * ──────────────────────────────────────────────────────────────────────────
 * ④ 张量数据区
 * ──────────────────────────────────────────────────────────────────────────
 *   所有张量的原始数据按 offset 定位依次存储；长度由 dimensions 与 data_type 共同决定。
 *   注意：张量数据可能未对齐，应按字节偏移直接读取。
 *
 * ──────────────────────────────────────────────────────────────────────────
 * 总结：完整文件布局
 * ──────────────────────────────────────────────────────────────────────────
 *   ┌─────────┬──────────────┬──────────────┬────────────────────┐
 *   │ ①Header │ ②Meta KV     │ ③Tensor 表   │ ④Tensor Data 区     │
 *   │ 24 字节  │ 列表（变长）   │ （变长）      │ （纯二进制裸数据）    │
 *   └─────────┴──────────────┴──────────────┴────────────────────┘
 *
 *   读取流程：Header → Meta KV 列表 → Tensor 信息表 → Tensor 数据区。
 *   数据区起点 = 24 + 元数据区大小 + 张量表大小；各张量数据再按 offset 定位。
 * ============================================================================
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

// 强制 1 字节对齐，防止编译器在结构体内部插入填充字节
#pragma pack(push, 1)
struct GGUFHeader {
    uint32_t magic;             // 魔数，应该是 'GGUF' (0x46554747)
    uint32_t version;           // 版本号，目前是 3
    uint64_t tensor_count;      // 张量数量
    uint64_t metadata_kv_count; // 元数据键值对数量
};
#pragma pack(pop)

// 静态断言确保大小符合你的描述：4+4+8+8 = 24 字节
static_assert(sizeof(GGUFHeader) == 24, "GGUFHeader size must be 24 bytes");

// ============================================================================
// 元数据键值对列表（Metadata KV Pairs）
// ============================================================================

// 值类型枚举（GGUFValueType），对应规范第 2.2 节
enum class GGUFValueType : std::uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// 递归 ARRAY 类型：前向声明以打破 std::variant 的循环依赖（决策 1）
struct ArrayValue;

// 元数据值：一个 std::variant 统一容纳所有基础类型 + ARRAY（通过 shared_ptr 间接递归）
using MetadataValue = std::variant<std::uint8_t,                // UINT8
                                   std::int8_t,                 // INT8
                                   std::uint16_t,               // UINT16
                                   std::int16_t,                // INT16
                                   std::uint32_t,               // UINT32
                                   std::int32_t,                // INT32
                                   float,                       // FLOAT32
                                   bool,                        // BOOL
                                   std::string,                 // STRING
                                   std::shared_ptr<ArrayValue>, // ARRAY
                                   std::uint64_t,               // UINT64
                                   std::int64_t,                // INT64
                                   double                       // FLOAT64
                                   >;

// ARRAY 值：元素类型 + 元素列表（每个元素仍是 MetadataValue，可继续嵌套 ARRAY）
struct ArrayValue {
    GGUFValueType element_type = GGUFValueType::UINT8; // 元素类型
    std::vector<MetadataValue> elements;               // 元素列表
};

// 元数据键值对（KV 列表中的一项）
struct GGUFMetadataKV {
    std::string key;                                 // 键名
    GGUFValueType value_type = GGUFValueType::UINT8; // 值类型
    MetadataValue value;                             // 值（variant）
};

// ============================================================================
// 张量信息表（Tensor Info Table）
// ============================================================================

struct GGUFTensorInfo {
    std::string name;                      // 张量名称
    std::uint32_t n_dimensions = 0;        // 维度数量
    std::vector<std::uint64_t> dimensions; // 各维度大小
    std::uint32_t data_type = 0;           // GGML 数据类型（F32=0 / F16=1 / BF16=21 ...）
    std::uint64_t offset = 0;              // 张量数据在文件中的起始偏移（从文件头计）

    // 元素总数 = 各维度乘积
    std::uint64_t element_count() const {
        std::uint64_t n = 1;
        for (auto d : dimensions)
            n *= d;
        return n;
    }
};

// ============================================================================
// 张量数据区（Tensor Data Region）
// ============================================================================

// 数据区视图：零拷贝，仅记录指针与大小，不持有数据（约束 C3 延迟加载）
// RAII：拥有 mmap 资源，析构自动释放；禁拷贝（防双释放）、可移动（转移所有权）
struct GGUFTensorData {
    const std::uint8_t *data_ptr = nullptr; // 数据区起始指针（mmap 后指向映射区）
    std::uint64_t data_offset = 0;          // 数据区在文件中的起始偏移
    std::uint64_t data_size = 0;            // 数据区总字节数
    void *map_base = nullptr;               // mmap 返回的原始基址（munmap 用）
    std::size_t map_len = 0;                // mmap 映射总长度（munmap 用）

    GGUFTensorData() = default;

    // 析构：自动释放映射（RAII）
    ~GGUFTensorData() {
        unmap();
    }

    // 禁拷贝：两个对象不能共享同一块映射（否则双释放）
    GGUFTensorData(const GGUFTensorData &) = delete;
    GGUFTensorData &operator=(const GGUFTensorData &) = delete;

    // 允许移动：转移映射所有权，源对象置空
    GGUFTensorData(GGUFTensorData &&other) noexcept {
        *this = std::move(other);
    }
    GGUFTensorData &operator=(GGUFTensorData &&other) noexcept;

    // 显式释放映射（幂等；析构时也会自动调用）
    void unmap() noexcept;
};

// 内存模型容器：Header + 元数据列表 + 张量表 + 数据区视图
struct GGUFModel {
    GGUFHeader header;                    // 文件头（24 字节）
    std::vector<GGUFMetadataKV> metadata; // 元数据 KV 列表
    std::vector<GGUFTensorInfo> tensors;  // 张量信息表
    GGUFTensorData data;                  // 张量数据区
};

// ============================================================================
// 值打印工具（供上层展示元数据）
// ============================================================================

// 值类型枚举 → 名称（"UINT8" / "ARRAY" ...）
// 微小查表函数，直接 inline 放头文件，调用处可内联为跳转表
inline const char *GGUFValueTypeName(GGUFValueType type) {
    switch (type) {
    case GGUFValueType::UINT8:
        return "UINT8";
    case GGUFValueType::INT8:
        return "INT8";
    case GGUFValueType::UINT16:
        return "UINT16";
    case GGUFValueType::INT16:
        return "INT16";
    case GGUFValueType::UINT32:
        return "UINT32";
    case GGUFValueType::INT32:
        return "INT32";
    case GGUFValueType::FLOAT32:
        return "FLOAT32";
    case GGUFValueType::BOOL:
        return "BOOL";
    case GGUFValueType::STRING:
        return "STRING";
    case GGUFValueType::ARRAY:
        return "ARRAY";
    case GGUFValueType::UINT64:
        return "UINT64";
    case GGUFValueType::INT64:
        return "INT64";
    case GGUFValueType::FLOAT64:
        return "FLOAT64";
    }
    return "?";
}

// 递归打印一个元数据值（ARRAY 只预览前几个元素，避免刷屏）
void printMetadataValue(std::ostream &os, const MetadataValue &value, int depth = 0,
                        std::size_t max_items = 3);

class GGUFLoader {
  public:
    /**
     * @brief 检查文件是否为有效的 GGUF 文件
     * @param filepath 文件路径
     * @return 是 GGUF 文件返回 true，否则返回 false
     */
    static bool is_gguf_file(const fs::path &filepath);

    /**
     * @brief 完整加载 GGUF 文件（头 + 元数据 KV 列表 + 张量信息表 + 数据区视图）
     * @param filepath 文件路径
     * @param model 输出参数，存储解析出的完整模型
     * @return 成功返回 true，失败返回 false
     */
    static bool load(const fs::path &filepath, GGUFModel &model);

    /**
     * @brief 读取一个长度前缀字符串（uint64 长度 + UTF-8 内容）
     */
    static bool read_string(std::istream &is, std::string &out);

    /**
     * @brief 按类型递归读取一个元数据值（ARRAY 会递归调用自身）
     */
    static bool read_value(std::istream &is, GGUFValueType type, MetadataValue &out);

    /**
     * @brief 把张量数据区映射进内存（零拷贝，按需调用）
     * @param filepath 文件路径
     * @param model 模型（load 之后调用）
     * @return 成功返回 true；已映射过也返回 true
     */
    static bool map_data(const fs::path &filepath, GGUFModel &model);

    /**
     * @brief 释放张量数据区映射（munmap），清空 data_ptr
     */
    static void unmap_data(GGUFModel &model);

  private:
    /**
     * @brief 验证魔数是否正确
     */
    static bool validate_magic(uint32_t magic);

    /**
     * @brief 从流中读取文件头并校验魔数（load / is_gguf_file 内部共用）
     */
    static bool read_header(std::istream &is, GGUFHeader &header);

    /**
     * @brief 解析元数据区（metadata_kv_count 个 KV 对）
     */
    static bool load_metadata(std::istream &is, std::uint64_t count,
                              std::vector<GGUFMetadataKV> &out);

    /**
     * @brief 解析张量信息表（tensor_count 个表项）
     */
    static bool load_tensor_info(std::istream &is, std::uint64_t count,
                                 std::vector<GGUFTensorInfo> &out);

    /**
     * @brief 定位张量数据区（延迟加载：只记录偏移与大小，不读取数据）
     */
    static bool load_data_region(std::istream &is, GGUFTensorData &data);
};
