#!/bin/bash

SOURCE_DIR="adGraph"
TARGET_ROOT="adGraph"
# 设置最大并发进程数 (推荐根据 CPU 核心数调整)
MAX_THREADS=8 

# -------------------------- 函数定义 --------------------------

# 定义核心的转换/复制函数，将被 xargs 并行调用
process_file() {
    local source_file_path="$1"
    
    # 忽略目录，只处理文件
    if [ ! -f "$source_file_path" ]; then
        return 0
    fi

    # 1. 获取文件的原始相对路径
    relative_path="${source_file_path#$SOURCE_DIR/}"
    target_file_path="$TARGET_ROOT/$relative_path"
    
    # 2. 检查后缀名，确定操作类型
    extension="${source_file_path##*.}"
    # 将后缀转换为小写进行匹配
    ext_lower=$(echo "$extension" | tr '[:upper:]' '[:lower:]')
    
    # 需要转换的文件类型列表
    case "$ext_lower" in
        cu|cpp|c|cuh|h|hxx|hpp)
            operation="HIP"
            ;;
        *)
            operation="COPY"
            ;;
    esac
    
    # 3. 确保目标子目录存在 (在并行环境中是安全的)
    target_dir=$(dirname "$target_file_path")
    mkdir -p "$target_dir"
    
    # 4. 执行操作
    if [ "$operation" == "HIP" ]; then
        echo "HIPIFY: $relative_path (as .$extension file) (PID: $$)"
        
        # 执行 hipify-perl 命令，输出到目标文件，保持原始文件名
        hipify-perl "$source_file_path" > "$target_file_path"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: hipify-perl failed for $relative_path"
            return 1
        else
            echo "DONE: $relative_path (Hipify)"
        fi
    else
        echo "COPY: $relative_path (PID: $$)"
        # 直接复制文件，保留原名
        cp -f "$source_file_path" "$target_file_path"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to copy $relative_path"
            return 1
        else
            echo "DONE: $relative_path (Copy)"
        fi
    fi
    
    return 0
}

# 导出函数和变量
export -f process_file
export SOURCE_DIR TARGET_ROOT 

# -------------------------- 主逻辑 --------------------------

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found."
    exit 1
fi

echo "Starting HIPIFY/COPY process (Max Threads: $MAX_THREADS, Keeping all original extensions)..."
echo "--------------------------------------------------------"

# 查找 src_bak 中的所有文件和目录
find "$SOURCE_DIR" -print0 | 
xargs -0 -P "$MAX_THREADS" -I {} bash -c 'process_file "$1"' _ {}

echo "--------------------------------------------------------"
echo "Conversion and Copy process finished."
