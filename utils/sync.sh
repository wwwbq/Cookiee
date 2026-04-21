#!/bin/bash

RANK=$1
MASTER_IP=$2
MODEL_DIR=$3
SYNC_PORT=112233
REAL_PORT=46697 # 
MAX_CONCURRENT=10  # 最大并发下载数

# 并发下载函数
download() {
    local max_concurrent=$1
    local counter=0
    
    # 使用临时文件来避免数组传递问题
    local temp_file=$(mktemp)
    echo "$FILE_LIST" > "$temp_file"
    
    while IFS= read -r file; do
        if [ -n "$file" ]; then
            echo "Downloading: $file"
            wget -q "http://${MASTER_IP}:${REAL_PORT}/${MODEL_DIR}/${file}" -O "${file}" &
            
            counter=$((counter + 1))
            if [ $((counter % max_concurrent)) -eq 0 ]; then
                wait  # 等待当前批次完成
            fi
        fi
    done < "$temp_file"
    
    wait  # 等待所有剩余任务完成
    rm -f "$temp_file"
}

# 如果RANK=0，说明是主节点，启动HTTP服务
if [ "$RANK" -eq 0 ]; then
    python3 -m http.server $SYNC_PORT > /dev/null 2>&1 &
    echo "HTTP server started on port $SYNC_PORT in directory $MODEL_DIR"
    # 等待其他节点进行下载
    sleep 5

# 如果RANK不为0，则通过wget从主节点拉取文件
else
    mkdir -p $MODEL_DIR
    echo "Connecting to master: $MASTER_IP"
    # 等待主节点HTTP服务启动
    sleep 2
    
    #获取文件列表
    echo "Fetching file list from master..."
    FILE_LIST=$(curl -s "http://${MASTER_IP}:${REAL_PORT}/${MODEL_DIR}/" | grep -oE 'href="[^"]+"' | sed 's/href="//g' | sed 's/"//g' | grep -v '/$' | grep -v '^../$')
    
    if [ -z "$FILE_LIST" ]; then
        echo "Error: No files found on master node"
        exit 1
    fi
    echo "Found $(echo "$FILE_LIST" | wc -l) files to download"

    files_array=($FILE_LIST)

    cd "$MODEL_DIR"

    download "$MAX_CONCURRENT"
    echo "Model synchronization completed"

    cd "$ORIGINAL_DIR"

    # 将MODEL_DIR目录下的所有内容同步到当前目录
    # wget -r -np -nH http://${MASTER_IP}:${REAL_PORT}/${MODEL_DIR}/ > /dev/null 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "Model synchronization completed successfully"
    # else
    #     echo "Error: Failed to download model files"
    #     exit 1
    # fi
fi