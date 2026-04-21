export LMUData="XXX"
VLMEVALKIT_HOME=XXX/VLMEvalKit

eval_config=$1

# rm -rf /root/LMUData/AI2D_TEST.tsv
# rm -rf /root/LMUData/images/AI2D_TEST

helper kill gpu

torchrun --nproc-per-node=8 \
        ${VLMEVALKIT_HOME}/run.py \
        --config $eval_config \
        --work-dir ./saves/eval_results \
        --verbose \
        #--judge qwen-plus \
        #--reuse

helper kill thirdparty
helper gpu ./logs/gpu.log 8 70000
