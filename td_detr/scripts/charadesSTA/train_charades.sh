dset_name=charades
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
exp_id=random
######## data paths
train_path=data/charades_sta/charades_sta_train_tvr_format.jsonl
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32
enc_layers=3
dec_layers=3


PYTHONPATH=$PYTHONPATH:. python td_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--n_epoch 100 \
--lw_saliency 4 \
--enc_layers ${enc_layers} \
--dec_layers ${dec_layers} \
--lr 0.0002 \
--contrastive_align_loss_coef 0.002 \
--clip_length 1 \
--max_v_l 100000 \
--n_epoch 100 \
--set_cost_span 10 \
--set_cost_giou 1 \
--span_loss_coef 10 \
--giou_loss_coef 1 \
${@:1}
