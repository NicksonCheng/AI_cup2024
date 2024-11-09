

mkdir -p "log/baai_1.5[contest]"
for pid in {0..9}
do
    gpu=$((pid % 4))
    logfile="log/baai_1.5[contest]/$pid.log"
    nohup python multichoice.py \
        --question_path ../contest \
        --source_path ../reference \
        --output_path ../output \
        --pid $pid \
        --partition 10 \
        --task "baai_1.5[contest]" \
        --gpu $gpu \
        > "$logfile" 2>&1 &
        #--baai_path BAAI/bge-large-zh-v1.5 \
        #--reranker BAAI/bge-reranker-v2-m3
    echo "Started process with pid=$pid, log file: $logfile"
done