cd /home/michal/Documents/01-Projects/12-Visual-Occlusion-Reasoning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
T=scripts/perception_probe/clip_align_v2/train.py
OUT=outputs/perception_probe/clip_v2
mkdir -p $OUT
COMMON="--no-wandb --raw-target-loss --batch-size 6 --max-steps 0 --epochs 10 --patience 20 \
  --num-workers 2 --prefetch-factor 2 --seed 0 --unseen-task-seed 0 \
  --embed-dim 512 --embed-hidden 256 --decouple-z --z-stop-grad --align-center-anchors"
gf () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
run () { name=$1; shift; rm -rf $OUT/$name
  while [ "$(gf)" -lt 4500 ]; do sleep 120; done
  echo "$(date '+%T') $name :: $*"
  $PY -u $T $COMMON "$@" --output-dir $OUT/$name > $OUT/$name.log 2>&1
  echo "$(date '+%T') $name exit=$?"
}
run sg_w1    --align-weight 1.0 --align-cos-weight 1.0
run sg_w3    --align-weight 3.0 --align-cos-weight 2.0
echo "$(date '+%T') stopgrad done"
