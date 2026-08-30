#include "GGUFLoader.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <type_traits>

static constexpr uint32_t GGUF_MAGIC = 0x46554747;

// ----------------------------------------------------------------------------
// 内部工具
// ----------------------------------------------------------------------------
namespace {

// TODO: 目前只支持小端，未来需用 std::endian + byteswap 做跨平台适配
// 从流中读取一个 POD 基础类型（本机为小端，直接按位拷贝即可）
template <typename T> bool read_primitive(std::istream &is, T &out) {
    is.read(reinterpret_cast<char *>(&out), sizeof(T));
    return static_cast<bool>(is);
}

// 读取一个标量类型并存入 variant（供 read_value 的标量分支复用，消除重复）
template <typename T> bool read_scalar(std::istream &is, MetadataValue &out) {
    T v;
    if (!read_primitive(is, v))
        return false;
    out = v;
    return true;
}

} // namespace

// ----------------------------------------------------------------------------
// ① 文件头（固定 24 字节）
// ----------------------------------------------------------------------------

bool GGUFLoader::read_header(std::istream &is, GGUFHeader &header) {
    is.read(reinterpret_cast<char *>(&header), sizeof(GGUFHeader));
    if (is.gcount() != sizeof(GGUFHeader)) {
        std::cerr << "读取头部失败，期望 24 字节，实际读取 " << is.gcount() << " 字节" << std::endl;
        return false;
    }
    if (!validate_magic(header.magic)) {
        std::cerr << "无效的 GGUF 文件：魔数错误 (0x" << std::hex << header.magic << std::dec << ")"
                  << std::endl;
        return false;
    }
    return true;
}

bool GGUFLoader::validate_magic(uint32_t magic) {
    return magic == GGUF_MAGIC;
}

bool GGUFLoader::is_gguf_file(const fs::path &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        return false;
    }
    GGUFHeader header;
    return read_header(file, header);
}

// ----------------------------------------------------------------------------
// ② 元数据 KV 列表（含递归 ARRAY）
// ----------------------------------------------------------------------------

bool GGUFLoader::read_string(std::istream &is, std::string &out) {
    std::uint64_t len = 0;
    if (!read_primitive(is, len))
        return false;
    out.resize(static_cast<std::size_t>(len));
    if (len > 0) {
        is.read(out.data(), static_cast<std::streamsize>(len));
        if (is.fail())
            return false;
    }
    return true;
}

bool GGUFLoader::read_value(std::istream &is, GGUFValueType type, MetadataValue &out) {
    switch (type) {
    case GGUFValueType::UINT8:
        return read_scalar<std::uint8_t>(is, out);
    case GGUFValueType::INT8:
        return read_scalar<std::int8_t>(is, out);
    case GGUFValueType::UINT16:
        return read_scalar<std::uint16_t>(is, out);
    case GGUFValueType::INT16:
        return read_scalar<std::int16_t>(is, out);
    case GGUFValueType::UINT32:
        return read_scalar<std::uint32_t>(is, out);
    case GGUFValueType::INT32:
        return read_scalar<std::int32_t>(is, out);
    case GGUFValueType::FLOAT32:
        return read_scalar<float>(is, out);
    case GGUFValueType::BOOL: {
        std::uint8_t v;
        if (!read_primitive(is, v))
            return false;
        out = (v != 0);
        return true;
    }
    case GGUFValueType::STRING: {
        std::string v;
        if (!read_string(is, v))
            return false;
        out = std::move(v);
        return true;
    }
    case GGUFValueType::ARRAY: {
        std::uint32_t elem_type_raw = 0;
        std::uint64_t count = 0;
        if (!read_primitive(is, elem_type_raw))
            return false;
        if (!read_primitive(is, count))
            return false;
        auto arr = std::make_shared<ArrayValue>();
        arr->element_type = static_cast<GGUFValueType>(elem_type_raw);
        arr->elements.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            MetadataValue elem;
            if (!read_value(is, arr->element_type, elem))
                return false;
            arr->elements.push_back(std::move(elem));
        }
        out = std::move(arr);
        return true;
    }
    case GGUFValueType::UINT64:
        return read_scalar<std::uint64_t>(is, out);
    case GGUFValueType::INT64:
        return read_scalar<std::int64_t>(is, out);
    case GGUFValueType::FLOAT64:
        return read_scalar<double>(is, out);
    default:
        return false;
    }
}

bool GGUFLoader::load_metadata(std::istream &is, std::uint64_t count,
                               std::vector<GGUFMetadataKV> &out) {
    out.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        GGUFMetadataKV kv;
        if (!read_string(is, kv.key))
            return false;
        std::uint32_t vt = 0;
        if (!read_primitive(is, vt))
            return false;
        kv.value_type = static_cast<GGUFValueType>(vt);
        if (!read_value(is, kv.value_type, kv.value))
            return false;
        out.push_back(std::move(kv));
    }
    return true;
}

// ----------------------------------------------------------------------------
// ③ 张量信息表
// ----------------------------------------------------------------------------

bool GGUFLoader::load_tensor_info(std::istream &is, std::uint64_t count,
                                  std::vector<GGUFTensorInfo> &out) {
    out.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        GGUFTensorInfo t;
        if (!read_string(is, t.name))
            return false;
        if (!read_primitive(is, t.n_dimensions))
            return false;
        t.dimensions.resize(t.n_dimensions);
        for (auto &d : t.dimensions) {
            if (!read_primitive(is, d))
                return false;
        }
        if (!read_primitive(is, t.data_type))
            return false;
        if (!read_primitive(is, t.offset))
            return false;
        out.push_back(std::move(t));
    }
    return true;
}

// ----------------------------------------------------------------------------
// ④ 张量数据区（延迟加载视图：只记录偏移与大小，不读取数据）
// ----------------------------------------------------------------------------

bool GGUFLoader::load_data_region(std::istream &is, GGUFTensorData &data) {
    // GGUF 规范：张量数据区从 32 字节对齐处开始（GGUF_DEFAULT_ALIGNMENT = 32）。
    // 张量信息表结束位置（tellg）未必在 32 边界上，需向上补齐到下一个对齐点，
    // 否则所有张量数据会整体错位读取（产生伪 NaN / 乱码）。
    constexpr std::uint64_t GGUF_ALIGN = 32;
    data.data_offset = static_cast<std::uint64_t>(is.tellg());
    const std::uint64_t rem = data.data_offset % GGUF_ALIGN;
    if (rem != 0)
        data.data_offset += GGUF_ALIGN - rem;

    is.seekg(0, std::ios::end);
    const auto file_end = is.tellg();
    if (is.fail() || data.data_offset > static_cast<std::uint64_t>(file_end)) {
        std::cerr << "定位张量数据区失败" << std::endl;
        return false;
    }
    data.data_size = static_cast<std::uint64_t>(file_end) - data.data_offset;
    data.data_ptr = nullptr; // 待调用者 mmap 挂载（零拷贝）
    return true;
}

// ----------------------------------------------------------------------------
// 完整加载：按 ① → ② → ③ → ④ 顺序解析
// ----------------------------------------------------------------------------

bool GGUFLoader::load(const fs::path &filepath, GGUFModel &model) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        return false;
    }

    // ① 文件头（固定 24 字节）
    if (!read_header(file, model.header))
        return false;

    // ② 元数据 KV 列表
    if (!load_metadata(file, model.header.metadata_kv_count, model.metadata)) {
        std::cerr << "解析元数据 KV 列表失败" << std::endl;
        return false;
    }

    // ③ 张量信息表
    if (!load_tensor_info(file, model.header.tensor_count, model.tensors)) {
        std::cerr << "解析张量信息表失败" << std::endl;
        return false;
    }

    // ④ 张量数据区（延迟加载）
    if (!load_data_region(file, model.data)) {
        std::cerr << "定位张量数据区失败" << std::endl;
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// 值打印工具（展示用，与文件读取顺序无关）
// ----------------------------------------------------------------------------

void printMetadataValue(std::ostream &os, const MetadataValue &value, int depth,
                        std::size_t max_items) {
    std::visit(
        [&](const auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, bool>) {
                os << (x ? "true" : "false");
            } else if constexpr (std::is_same_v<T, float>) {
                os << std::fixed << std::setprecision(6) << x;
            } else if constexpr (std::is_same_v<T, double>) {
                os << std::fixed << std::setprecision(6) << x;
            } else if constexpr (std::is_same_v<T, std::string>) {
                os << '"' << x << '"';
            } else if constexpr (std::is_same_v<T, std::shared_ptr<ArrayValue>>) {
                const auto &arr = *x;
                os << "[ " << GGUFValueTypeName(arr.element_type) << " x" << arr.elements.size()
                   << " ]";
                if (!arr.elements.empty() && depth < 2) {
                    os << " (";
                    const std::size_t n = std::min<std::size_t>(arr.elements.size(), max_items);
                    for (std::size_t i = 0; i < n; ++i) {
                        if (i)
                            os << ", ";
                        printMetadataValue(os, arr.elements[i], depth + 1);
                    }
                    if (arr.elements.size() > n)
                        os << ", ...";
                    os << ")";
                }
            } else {
                os << +x; // 无符号窄类型提升为 int 打印
            }
        },
        value);
}