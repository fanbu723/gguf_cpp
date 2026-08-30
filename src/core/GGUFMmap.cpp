/*
 * GGUFMmap.cpp — 张量数据区内存映射（core::mmap 模块）
 *
 * 职责：把 GGUF 文件的数据区（[data_offset, data_offset+data_size)）
 *       零拷贝映射进进程地址空间，按需访问、按页加载。
 *       对应架构文档「模块 4：数据区延迟加载器」的落地实现。
 *
 * 设计要点：
 *  - mmap 偏移必须是页大小的整数倍，因此先把起点向下对齐到页边界，
 *    再记录偏差 delta，映射后 data_ptr = base + delta。
 *  - 映射建立后即可关闭 fd（映射仍有效）。
 *  - map_base / map_len 记录给 munmap 用。
 *
 * 说明：下面所有 `::` 前缀（如 ::open / ::mmap）表示调用"全局命名空间"的
 *       函数——即操作系统提供的 POSIX/C 库函数，而不是本项目自定义的函数。
 *       初学可理解为：带 `::` 的就是"调系统函数"。
 */

#include "GGUFLoader.hpp"

#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

bool GGUFLoader::map_data(const fs::path &filepath, GGUFModel &model) {
    // 已映射过则直接返回
    if (model.data.data_ptr != nullptr)
        return true;
    if (model.data.data_size == 0)
        return false;

    // ::open —— 全局命名空间的 open：打开文件，返回文件描述符 fd（失败为 -1）
    const int fd = ::open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "map_data: 无法打开文件 " << filepath << std::endl;
        return false;
    }

    // 页对齐：mmap 的偏移参数必须是页大小的整数倍
    // ::sysconf —— 全局的 sysconf：查询系统配置，这里取页大小（通常 4096）
    const long page = ::sysconf(_SC_PAGESIZE);
    const std::uint64_t aligned = (model.data.data_offset / static_cast<std::uint64_t>(page)) *
                                  static_cast<std::uint64_t>(page);
    const std::size_t delta = static_cast<std::size_t>(model.data.data_offset - aligned);
    const std::size_t total = static_cast<std::size_t>(model.data.data_size) + delta;

    // ::mmap —— 全局的 mmap：把文件的 [aligned, aligned+total) 区域映射进进程地址空间
    void *base = ::mmap(nullptr, total, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(aligned));
    ::close(fd); // ::close —— 全局的 close：关闭 fd（mmap 建立后即可关闭，映射仍有效）

    if (base == MAP_FAILED) {
        std::cerr << "map_data: mmap 失败" << std::endl;
        return false;
    }

    model.data.map_base = base;
    model.data.map_len = total;
    model.data.data_ptr = static_cast<const std::uint8_t *>(base) + delta;
    return true;
}

// ----------------------------------------------------------------------------
// GGUFTensorData：RAII 资源管理（实现）
// ----------------------------------------------------------------------------

// 释放映射（幂等：未映射则什么都不做）
void GGUFTensorData::unmap() noexcept {
    if (map_base != nullptr) {
        // ::munmap —— 全局的 munmap：解除之前 mmap 建立的映射
        ::munmap(map_base, map_len);
        map_base = nullptr;
        map_len = 0;
        data_ptr = nullptr;
    }
}

// 移动赋值：释放旧资源，转移新资源所有权，源对象置空
GGUFTensorData &GGUFTensorData::operator=(GGUFTensorData &&other) noexcept {
    if (this != &other) {
        unmap(); // 释放自己可能持有的旧映射

        data_ptr = other.data_ptr;
        data_offset = other.data_offset;
        data_size = other.data_size;
        map_base = other.map_base;
        map_len = other.map_len;

        other.data_ptr = nullptr;
        other.data_offset = 0;
        other.data_size = 0;
        other.map_base = nullptr;
        other.map_len = 0;
    }
    return *this;
}

void GGUFLoader::unmap_data(GGUFModel &model) {
    model.data.unmap();
}
