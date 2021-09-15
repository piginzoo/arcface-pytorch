#!/bin/bash
Date=$(date +%Y%m%d%H%M)
if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep "name arcface"| grep -v grep|awk '{print $2}'|xargs -I {} kill -9 {}
    exit
fi

if [ "$1" = "debug" ]
then
    echo "调试模式"
    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --name arcface \
    --mode debug
elif [ "$1" = "prod" ]
then
    # 这个是用于docker的训练用的entry，CUDA_VISIBLE_DEVICES=0，因为显卡始终是第一块
    echo "Docker生产模式"
    CUDA_VISIBLE_DEVICES=0 \
      python train.py \
      --name arcface \
      --mode normal>logs/console.log
else
    echo "无法识别的模式：$1"
fi