cd /home/michal/Documents/01-Projects/12-Visual-Occlusion-Reasoning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
T=scripts/perception_probe/clip_align_v2/train.py
OUT=outputs/perception_probe/clip_v2
mkdir -p $OUT
COMMON="--no-wandb --raw-target-loss --batch-size 6 --max-steps 0 --epochs 10 --patience 20 \
  --num-workers 2 --prefetch-factor 2 --seed 0 --unseen-task-seed 0 \
  --embed-dim 512 --embed-hidden 256"
gf () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
run () { name=$1; shift; rm -rf $OUT/$name
  while [ "$(gf)" -lt 4500 ]; do sleep 120; done
  echo "$(date '+%T') $name :: $*"
  $PY -u $T $COMMON "$@" --output-dir $OUT/$name > $OUT/$name.log 2>&1
  echo "$(date '+%T') $name exit=$?"
}
run all_w1        --decouple-z --align-center-anchors --align-cos-weight 1.0 --align-weight 1.0
run all_w0.3      --decouple-z --align-center-anchors --align-cos-weight 0.5 --align-weight 0.3
run intrunk_w0.1  --align-center-anchors --align-cos-weight 1.0 --align-weight 0.1
echo "$(date '+%T') clip_v2 done"
