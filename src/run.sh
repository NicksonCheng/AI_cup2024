

mkdir -p "log/only_chinese[contest]"
for pid in {0..9}
do
    gpu=$((pid % 4))
    logfile="log/only_chinese[contest]/$pid.log"
    nohup python multichoice.py \
        --question_path ../contest \
        --source_path ../reference \
        --output_path ../output \
        --pid $pid \
        --partition 10 \
        --task "only_chinese[contest]" \
        --gpu $gpu \
        > "$logfile" 2>&1 &
        #--baai_path BAAI/bge-large-zh-v1.5 \
        #--reranker BAAI/bge-reranker-v2-m3
    echo "Started process with pid=$pid, log file: $logfile"
done