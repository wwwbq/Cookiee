#python -m torch.distributed.run --nnodes 1 --node_rank 0 --nproc_per_node 8 --master_addr 127.0.0.1 --master_port 27000 sft.py configs/sft_think.yaml
RANK=$1
MASTER=XXX

kill_process() {
    # kill process
    helper kill python
    sleep 2
    helper kill python
}


#kill_process
helper kill gpu
sleep 1

echo "########## start pretrain ##########"
NNODES=2 NODE_RANK=${RANK} MASTER_ADDR=${MASTER} MASTER_PORT=27000 cookiee train examples/vlm/llava/pretrain.py examples/vlm/llava/config/pretrain.yaml

#kill_process
helper kill pretrain
sleep 2

# echo "########## Starting model synchronization... #########"
# sh sync.sh ${RANK} ${MASTER} saves/cookiee-vlm/preference_pretrain_exp23rerun

echo "########## start sft ##########"
NNODES=2 NODE_RANK=${RANK} MASTER_ADDR=${MASTER} MASTER_PORT=27000 cookiee train examples/vlm/llava/sft.py examples/vlm/llava/config/sft.yaml

#kill_process
helper kill sft
sleep 2

# kill $(ps -ef | grep 'python3 -m http.server' | awk '{print $2}')
# echo "HTTP server stopped"

# use gpu
helper gpu logs/gpu.log 8 70000