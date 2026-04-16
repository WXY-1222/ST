import os
import argparse
import torch
import torch.distributed as dist
import baseline
from SingularTrajectory import SingularTrajectory
from utils import *
from utils import trainer as trainer_module


def setup_distributed(dist_backend: str = "nccl"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=dist_backend, init_method="env://")
    return is_distributed, rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/singulartrajectory-transformerdiffusion-zara1.json", type=str, help="config file path")
    parser.add_argument('--tag', default="SingularTrajectory-TEMP", type=str, help="personal tag for the model")
    parser.add_argument('--gpu_id', default="0", type=str, help="gpu id for single-process run")
    parser.add_argument('--test', default=False, action='store_true', help="evaluation mode")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--num_workers', default=4, type=int, help="dataloader workers per process")
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true', help="enable DataLoader pin_memory")
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help="disable DataLoader pin_memory")
    parser.set_defaults(pin_memory=True)
    parser.add_argument('--dist_backend', default="nccl", choices=["nccl", "gloo"], type=str, help="distributed backend for torchrun")
    parser.add_argument('--dataset_dir', default="", type=str, help="override dataset root directory in config")
    parser.add_argument('--checkpoint_dir', default="", type=str, help="override checkpoint root directory in config")
    args = parser.parse_args()

    # Respect single-process GPU selection while keeping torchrun behavior unchanged.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    is_distributed, rank, local_rank, world_size = setup_distributed(args.dist_backend)
    args.distributed = is_distributed
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.is_main = (rank == 0)

    def mprint(*values, **kwargs):
        if args.is_main:
            print(*values, **kwargs)

    try:
        mprint("===== Arguments =====")
        mprint("Distributed:", is_distributed, f"(world_size={world_size}, rank={rank}, local_rank={local_rank})")
        if args.is_main:
            print_arguments(vars(args))

        mprint("===== Configs =====")
        hyper_params = get_exp_config(args.cfg)
        if args.dataset_dir:
            hyper_params.dataset_dir = args.dataset_dir
        if args.checkpoint_dir:
            hyper_params.checkpoint_dir = args.checkpoint_dir
        if args.is_main:
            print_arguments(hyper_params)

        PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
        hook_func = DotDict({
            "model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
            "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
            "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook
        })

        trainer_name_list = [name for name in trainer_module.__dict__.keys() if hyper_params.baseline in name.lower()]
        if len(trainer_name_list) == 0:
            raise ValueError(f"Cannot find trainer for baseline `{hyper_params.baseline}`.")
        ModelTrainer = getattr(trainer_module, trainer_name_list[0])

        model_trainer = ModelTrainer(
            base_model=PredictorModel,
            model=SingularTrajectory,
            hook_func=hook_func,
            args=args,
            hyper_params=hyper_params,
        )

        if not args.test:
            model_trainer.init_descriptor()
            model_trainer.fit()
        else:
            model_trainer.load_model()
            results = model_trainer.test()
            if args.is_main:
                print("Testing...", end=' ')
                print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])
    finally:
        cleanup_distributed()
