Date=$(date +%Y%m%d%H%M)
if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep "name arcface"| grep -v grep|awk '{print $2}'|xargs -I {} kill -9 {}
    exit
fi

if [ "$1" = "console" ]
then
    echo "调试模式"
    python train.py \
    --name arcface \
    --mode debug
else
    echo "生产模式"
    CUDA_VISIBLE_DEVICES=1 \
      python train.py \
      --name captcha \
      --mode normal>> ./logs/console.${Date}.log 2>&1 &
fi