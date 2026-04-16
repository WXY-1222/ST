# Server Multi-GPU Training (SingularTrajectory / ST)

`trainval.py` now supports `torchrun` multi-GPU distributed training (DDP).

## 1) Single node, 8xA100

```bash
cd /path/to/SingularTrajectory-main

torchrun --standalone --nproc_per_node=8 trainval.py \
  --cfg ./config/stochastic/singulartrajectory-transformerdiffusion-zara1.json \
  --tag SingularTrajectory-stochastic-ddp \
  --dist_backend nccl \
  --num_workers 4 \
  --pin_memory
```

## 2) Using helper script

```bash
cd /path/to/SingularTrajectory-main

bash ./script/train_torchrun.sh \
  -n 8 \
  -c ./config/stochastic/singulartrajectory-transformerdiffusion-zara1.json \
  -t SingularTrajectory-stochastic-ddp \
  -b nccl \
  -w 4
```

## 3) Multi-node example

On node 0:

```bash
bash ./script/train_torchrun.sh -n 8 -N 2 -R 0 -A 10.0.0.1 -P 29500 -c <cfg> -t <tag>
```

On node 1:

```bash
bash ./script/train_torchrun.sh -n 8 -N 2 -R 1 -A 10.0.0.1 -P 29500 -c <cfg> -t <tag>
```

## Notes

- For single-process mode, the original command still works:
  - `python trainval.py --cfg <cfg> --tag <tag> --gpu_id 0`
- In DDP mode, `--gpu_id` is ignored and GPU selection should be handled by `torchrun`/environment.
- Checkpoint saving and major logs are rank-0 only.

## INTERACTION Baseline (for DIGIR comparison)

Use the DIGIR preprocessed pkl directly:

```bash
cd /path/to/SingularTrajectory-main

torchrun --standalone --nproc_per_node=8 trainval.py \
  --cfg ./config/interaction/singulartrajectory-transformerdiffusion-interaction.json \
  --tag ST-interaction-k3 \
  --dataset_dir /data/sdb/bitwxy/st_data \
  --checkpoint_dir /data/sdb/bitwxy/st_checkpoints \
  --dist_backend nccl \
  --num_workers 4 \
  --pin_memory \
  --eval_every 4 \
  --eval_k 3 \
  --best_metric minADE_k \
  --nan_fill nan \
  --miss_threshold 2.0
```

Notes:
- `interaction_data_path` is read from the cfg file. Update it to your actual DIGIR pkl path when needed.
- INTERACTION has no ETH/UCY-style homography/vectorfield in this pipeline, so ST uses initial anchors (no map-based anchor refinement).
- If your INTERACTION pkl has no `test` split, training still works (`train` + `val`), but `--test` mode now requires an explicit `test` split (no silent fallback to `val`).
