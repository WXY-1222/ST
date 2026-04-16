#!/bin/bash
set -e

echo "Start distributed SingularTrajectory training"

nproc_per_node=8
nnodes=1
node_rank=0
master_addr="127.0.0.1"
master_port=29500

cfg="./config/stochastic/singulartrajectory-transformerdiffusion-zara1.json"
tag="SingularTrajectory-ddp"
dist_backend="nccl"
num_workers=4
dataset_dir=""
checkpoint_dir=""
test_mode=0

while getopts n:N:R:A:P:c:t:b:w:d:k:T flag
do
  case "${flag}" in
    n) nproc_per_node=${OPTARG};;
    N) nnodes=${OPTARG};;
    R) node_rank=${OPTARG};;
    A) master_addr=${OPTARG};;
    P) master_port=${OPTARG};;
    c) cfg=${OPTARG};;
    t) tag=${OPTARG};;
    b) dist_backend=${OPTARG};;
    w) num_workers=${OPTARG};;
    d) dataset_dir=${OPTARG};;
    k) checkpoint_dir=${OPTARG};;
    T) test_mode=1;;
    *)
      echo "usage: $0 [-n NPROC_PER_NODE] [-N NNODES] [-R NODE_RANK] [-A MASTER_ADDR] [-P MASTER_PORT] [-c CFG] [-t TAG] [-b DIST_BACKEND] [-w NUM_WORKERS] [-d DATASET_DIR] [-k CHECKPOINT_DIR] [-T]" >&2
      exit 1
      ;;
  esac
done

extra_args=()
if [ -n "${dataset_dir}" ]; then
  extra_args+=("--dataset_dir" "${dataset_dir}")
fi
if [ -n "${checkpoint_dir}" ]; then
  extra_args+=("--checkpoint_dir" "${checkpoint_dir}")
fi
if [ ${test_mode} -eq 1 ]; then
  extra_args+=("--test")
fi

torchrun \
  --nnodes ${nnodes} \
  --nproc_per_node ${nproc_per_node} \
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
  trainval.py \
  --cfg "${cfg}" \
  --tag "${tag}" \
  --dist_backend "${dist_backend}" \
  --num_workers ${num_workers} \
  --pin_memory \
  "${extra_args[@]}"

echo "Done."
