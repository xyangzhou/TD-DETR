ckpt_path=$1
eval_split_name=$2
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python td_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
