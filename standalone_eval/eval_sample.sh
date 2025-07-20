#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
submission_path=standalone_eval/results_final.jsonl
gt_path=data/highlight_val_release.jsonl
save_path=standalone_eval/results_final.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
