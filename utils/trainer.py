import os
import pickle
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from . import *


class STTrainer:
    r"""Base class for all Trainers"""

    def __init__(self, args, hyper_params):
        self.args, self.hyper_params = args, hyper_params
        self.distributed = bool(getattr(args, "distributed", False))
        self.rank = int(getattr(args, "rank", 0))
        self.local_rank = int(getattr(args, "local_rank", 0))
        self.world_size = int(getattr(args, "world_size", 1))
        self.is_main = (self.rank == 0)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}" if self.distributed else "cuda")
        else:
            self.device = torch.device("cpu")

        if self.is_main:
            print("Trainer initiating...")

        # Reproducibility
        reproducibility_settings(seed=int(getattr(args, "seed", 0)))

        self.model, self.optimizer, self.scheduler = None, None, None
        self.loader_train, self.loader_val, self.loader_test = None, None, None
        self.dataset_dir = os.path.join(hyper_params.dataset_dir, hyper_params.dataset)
        self.checkpoint_dir = os.path.join(hyper_params.checkpoint_dir, args.tag, hyper_params.dataset)
        if self.is_main:
            print("Checkpoint dir:", self.checkpoint_dir)
        self.log = {'train_loss': [], 'val_loss': []}
        self.stats_func, self.stats_meter = None, None
        self.reset_metric()

        if not args.test:
            # Save arguments and configs
            if self.is_main and not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            if self.is_main:
                with open(os.path.join(self.checkpoint_dir, 'args.pkl'), 'wb') as fp:
                    pickle.dump(args, fp)

                with open(os.path.join(self.checkpoint_dir, 'config.pkl'), 'wb') as fp:
                    pickle.dump(hyper_params, fp)

        self.barrier()

    def unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def barrier(self):
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _reduce_sum_count(self, total, count):
        tensor = torch.tensor([float(total), float(count)], dtype=torch.float64, device=self.device)
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor[0].item()), float(tensor[1].item())

    def _progress(self, loader, desc):
        return tqdm(loader, desc=desc, mininterval=1, disable=not self.is_main)

    def _broadcast_model_state(self):
        if not (self.distributed and dist.is_available() and dist.is_initialized()):
            return
        backend = dist.get_backend()
        with torch.no_grad():
            for value in self.unwrap_model().state_dict().values():
                if not torch.is_tensor(value):
                    continue

                # torch.distributed.broadcast requires contiguous tensors.
                tensor_to_broadcast = value if value.is_contiguous() else value.contiguous()

                # NCCL only supports CUDA tensors.
                moved_to_cuda = False
                if backend == "nccl" and not tensor_to_broadcast.is_cuda:
                    tensor_to_broadcast = tensor_to_broadcast.to(self.device)
                    moved_to_cuda = True

                dist.broadcast(tensor_to_broadcast, src=0)

                needs_copy_back = (tensor_to_broadcast.data_ptr() != value.data_ptr()) or moved_to_cuda
                if needs_copy_back:
                    value.copy_(tensor_to_broadcast.to(device=value.device, dtype=value.dtype))

    def _sync_loader_epoch(self, loader, epoch):
        if loader is None:
            return
        batch_sampler = getattr(loader, "batch_sampler", None)
        sampler = getattr(loader, "sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _set_epoch(self, epoch):
        self._sync_loader_epoch(self.loader_train, epoch)
        self._sync_loader_epoch(self.loader_val, epoch)
        self._sync_loader_epoch(self.loader_test, epoch)

    def _collect_metric_means(self):
        results = {}
        for metric_name, metric_meter in self.stats_meter.items():
            if len(metric_meter.data) > 0:
                local_values = np.concatenate(metric_meter.data, axis=0)
            else:
                local_values = np.array([], dtype=np.float32)

            if self.distributed and dist.is_available() and dist.is_initialized():
                gathered = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered, local_values)
                merged = [x for x in gathered if x is not None and len(x) > 0]
                if len(merged) == 0:
                    results[metric_name] = float('nan')
                else:
                    results[metric_name] = float(np.concatenate(merged, axis=0).mean())
            else:
                results[metric_name] = float(local_values.mean()) if local_values.size > 0 else float('nan')
        return results

    def _compute_digir_batch_metrics(self, pred, gt, eval_k=5, miss_threshold=2.0):
        k = max(1, min(int(eval_k), int(pred.size(0))))
        pred_k = pred[:k]
        displacement = (pred_k - gt.unsqueeze(dim=0)).norm(p=2, dim=-1)  # K, N, T
        min_ade = displacement.mean(dim=2).min(dim=0)[0]
        min_fde = displacement[:, :, -1].min(dim=0)[0]
        miss = min_fde.gt(float(miss_threshold)).type_as(min_fde)
        return (
            min_ade.detach().cpu().numpy(),
            min_fde.detach().cpu().numpy(),
            miss.detach().cpu().numpy(),
        )

    @staticmethod
    def _compute_scene_batch_metrics(minade, minfde, miss, scene_id):
        scene_id = np.asarray(scene_id)
        if scene_id.ndim > 1:
            scene_id = scene_id.reshape(-1)

        if scene_id.shape[0] != minade.shape[0]:
            scene_id = np.array(["scene"] * int(minade.shape[0]), dtype=object)

        unique_scene = np.unique(scene_id)
        scene_minade, scene_minfde, scene_miss = [], [], []
        for sid in unique_scene:
            mask = (scene_id == sid)
            if not np.any(mask):
                continue
            scene_minade.append(float(minade[mask].mean()))
            scene_minfde.append(float(minfde[mask].mean()))
            scene_miss.append(float(miss[mask].mean()))

        return (
            np.asarray(scene_minade, dtype=np.float32),
            np.asarray(scene_minfde, dtype=np.float32),
            np.asarray(scene_miss, dtype=np.float32),
        )

    @staticmethod
    def _merge_numpy_arrays(local_arrays):
        if len(local_arrays) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(local_arrays, axis=0)

    def _gather_numpy_array(self, local_array):
        if self.distributed and dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, local_array)
            valid = [x for x in gathered if x is not None and len(x) > 0]
            if len(valid) == 0:
                return np.array([], dtype=np.float32)
            return np.concatenate(valid, axis=0)
        return local_array

    @staticmethod
    def _placeholder_metric_value(nan_fill):
        return 0.0 if str(nan_fill).lower() == "zero" else float('nan')

    def _finalize_digir_metrics(
        self,
        minade_list,
        minfde_list,
        miss_list,
        scene_minade_list=None,
        scene_minfde_list=None,
        scene_miss_list=None,
        eval_k=5,
        nan_fill="nan",
    ):
        minade_local = self._merge_numpy_arrays(minade_list)
        minfde_local = self._merge_numpy_arrays(minfde_list)
        miss_local = self._merge_numpy_arrays(miss_list)
        scene_minade_local = self._merge_numpy_arrays(scene_minade_list or [])
        scene_minfde_local = self._merge_numpy_arrays(scene_minfde_list or [])
        scene_miss_local = self._merge_numpy_arrays(scene_miss_list or [])

        minade_all = self._gather_numpy_array(minade_local)
        minfde_all = self._gather_numpy_array(minfde_local)
        miss_all = self._gather_numpy_array(miss_local)
        scene_minade_all = self._gather_numpy_array(scene_minade_local)
        scene_minfde_all = self._gather_numpy_array(scene_minfde_local)
        scene_miss_all = self._gather_numpy_array(scene_miss_local)

        placeholder = self._placeholder_metric_value(nan_fill)

        if minade_all.size == 0 or minfde_all.size == 0 or miss_all.size == 0:
            return {
                f"minADE_{int(eval_k)}": float('nan'),
                f"minFDE_{int(eval_k)}": float('nan'),
                "MissRate": float('nan'),
                f"scene_minADE_{int(eval_k)}": float('nan'),
                f"scene_minFDE_{int(eval_k)}": float('nan'),
                "scene_MissRate": float('nan'),
                "IntentAcc": placeholder,
                "ITC": placeholder,
                "CollisionRate": placeholder,
                "OffRoadRate": placeholder,
            }

        scene_minade_value = float(scene_minade_all.mean()) if scene_minade_all.size > 0 else float('nan')
        scene_minfde_value = float(scene_minfde_all.mean()) if scene_minfde_all.size > 0 else float('nan')
        scene_miss_value = float(scene_miss_all.mean()) if scene_miss_all.size > 0 else float('nan')

        return {
            f"minADE_{int(eval_k)}": float(minade_all.mean()),
            f"minFDE_{int(eval_k)}": float(minfde_all.mean()),
            "MissRate": float(miss_all.mean()),
            f"scene_minADE_{int(eval_k)}": scene_minade_value,
            f"scene_minFDE_{int(eval_k)}": scene_minfde_value,
            "scene_MissRate": scene_miss_value,
            "IntentAcc": placeholder,
            "ITC": placeholder,
            "CollisionRate": placeholder,
            "OffRoadRate": placeholder,
        }

    @staticmethod
    def _unpack_obs_pred_batch(batch):
        if isinstance(batch, dict):
            return batch["obs_traj"], batch["pred_traj"]
        return batch[:2]

    @staticmethod
    def _extract_scene_id(batch, num_agents):
        if isinstance(batch, dict) and "scene_id" in batch:
            return batch["scene_id"]
        return np.array(["scene"] * int(num_agents), dtype=object)

    def init_descriptor(self):
        # Singular space initialization
        if self.is_main:
            print("Singular space initialization...")
        obs_traj, pred_traj = self.loader_train.dataset.obs_traj, self.loader_train.dataset.pred_traj
        obs_traj, pred_traj = augment_trajectory(obs_traj, pred_traj)
        if self.distributed:
            if self.is_main:
                self.unwrap_model().calculate_parameters(obs_traj, pred_traj)
            self.barrier()
            self._broadcast_model_state()
        else:
            self.unwrap_model().calculate_parameters(obs_traj, pred_traj)
        if self.is_main:
            print("Anchor generation...")
        self.barrier()

    def init_adaptive_anchor(self, dataset):
        if self.is_main:
            print("Adaptive anchor initialization...")
        dataset.anchor = self.unwrap_model().calculate_adaptive_anchor(dataset)
        self.barrier()

    def train(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def valid(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, loader=None, split="Test", eval_k=5, miss_threshold=2.0, nan_fill="nan"):
        raise NotImplementedError

    def fit(self):
        if self.is_main:
            print("Training started...")
        best_metric = str(getattr(self.args, "best_metric", "val_loss"))
        if best_metric not in {"val_loss", "minADE_k"}:
            raise ValueError(f"Unsupported best metric `{best_metric}`")

        best_score = float('inf')
        best_epoch = -1
        eval_every = max(1, int(getattr(self.args, "eval_every", 1)))
        eval_k = int(getattr(self.args, "eval_k", 5))
        miss_threshold = float(getattr(self.args, "miss_threshold", 2.0))
        nan_fill = str(getattr(self.args, "nan_fill", "nan"))

        for epoch in range(self.hyper_params.num_epochs):
            self._set_epoch(epoch)
            self.train(epoch)
            self.valid(epoch)

            if self.hyper_params.lr_schd:
                self.scheduler.step()

            current_val = self.log['val_loss'][-1]
            if best_metric == "val_loss" and current_val < best_score:
                best_score = current_val
                best_epoch = epoch
                self.save_model()

            if self.is_main:
                print(" ")
                print("Dataset: {0}, Epoch: {1}".format(self.hyper_params.dataset, epoch))
                print("Train_loss: {0:.8f}, Val_los: {1:.8f}".format(self.log['train_loss'][-1], current_val))
                min_val_epoch = int(np.array(self.log['val_loss']).argmin())
                min_val_loss = float(np.array(self.log['val_loss']).min())
                print("Min_val_epoch: {0}, Min_val_loss: {1:.8f}".format(min_val_epoch, min_val_loss))
            should_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == self.hyper_params.num_epochs)
            if should_eval:
                eval_results = self.test(
                    loader=self.loader_val,
                    split="Eval",
                    eval_k=eval_k,
                    miss_threshold=miss_threshold,
                    nan_fill=nan_fill,
                )
                eval_metric_name = f"minADE_{eval_k}"
                eval_metric_value = float(eval_results.get(eval_metric_name, float('nan')))
                if best_metric == "minADE_k" and np.isfinite(eval_metric_value) and eval_metric_value < best_score:
                    best_score = eval_metric_value
                    best_epoch = epoch
                    self.save_model()
                if self.is_main:
                    print("Eval metrics:")
                    print(f"  minADE_{eval_k}: {eval_results[f'minADE_{eval_k}']:.6f}")
                    print(f"  minFDE_{eval_k}: {eval_results[f'minFDE_{eval_k}']:.6f}")
                    print(f"  MissRate: {eval_results['MissRate']:.2%}")
                    print(f"  scene_minADE_{eval_k}: {eval_results[f'scene_minADE_{eval_k}']:.6f}")
                    print(f"  scene_minFDE_{eval_k}: {eval_results[f'scene_minFDE_{eval_k}']:.6f}")
                    print(f"  scene_MissRate: {eval_results['scene_MissRate']:.2%}")
                    print("  IntentAcc: N/A (ST dataset has no intent labels)")
                    print("  ITC: N/A (ST dataset has no intent labels)")
                    print("  CollisionRate: N/A (not defined in ST evaluator)")
                    print("  OffRoadRate: N/A (no map/off-road labels in ST pipeline)")
                    if best_metric == "minADE_k":
                        print(f"  Best selection metric: {eval_metric_name} (lower is better)")

            if self.is_main and best_epoch >= 0:
                if best_metric == "val_loss":
                    print(f"Best checkpoint by val_loss -> epoch: {best_epoch}, score: {best_score:.8f}")
                else:
                    print(f"Best checkpoint by minADE_{eval_k} -> epoch: {best_epoch}, score: {best_score:.8f}")
            if self.is_main:
                print(" ")
            self.barrier()

        if best_epoch < 0:
            if self.is_main:
                print("Warning: no finite checkpoint selection metric observed; saving last model as model_best.pth.")
            self.save_model()

        if self.is_main:
            print("Done.")

    def reset_metric(self):
        self.stats_func = {'ADE': compute_batch_ade, 'FDE': compute_batch_fde}
        self.stats_meter = {x: AverageMeter() for x in self.stats_func.keys()}

    def get_metric(self):
        return self.stats_meter

    def load_model(self, filename='model_best.pth'):
        model_path = os.path.join(self.checkpoint_dir, filename)
        state_dict = torch.load(model_path, map_location=self.device)
        self.unwrap_model().load_state_dict(state_dict)
        self.barrier()

    def save_model(self, filename='model_best.pth'):
        if not self.is_main:
            return
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        model_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.unwrap_model().state_dict(), model_path)


class STSequencedMiniBatchTrainer(STTrainer):
    r"""Base class using sequenced mini-batch training strategy"""

    def __init__(self, args, hyper_params):
        super().__init__(args, hyper_params)

        # Dataset preprocessing
        obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
        num_workers = int(getattr(args, "num_workers", 0))
        pin_memory = bool(getattr(args, "pin_memory", True))
        seed = int(getattr(args, "seed", 0))
        interaction_data_path = str(getattr(hyper_params, "interaction_data_path", "") or "")
        self.loader_train = get_dataloader(
            self.dataset_dir, 'train', obs_len, pred_len, batch_size=1, skip=skip,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed,
            interaction_data_path=interaction_data_path)
        self.loader_val = get_dataloader(
            self.dataset_dir, 'val', obs_len, pred_len, batch_size=1,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed,
            interaction_data_path=interaction_data_path)
        self.loader_test = None
        try:
            self.loader_test = get_dataloader(
                self.dataset_dir, 'test', obs_len, pred_len, batch_size=1,
                num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
                rank=self.rank, world_size=self.world_size, seed=seed,
                interaction_data_path=interaction_data_path)
        except KeyError:
            if bool(getattr(args, "test", False)):
                raise
            if self.is_main:
                print("Warning: test split not found; training continues without loader_test.")

    def train(self, epoch):
        self.model.train()
        loss_batch = 0.0
        loss_cum = None
        accum_steps = 0
        num_batches = 0.0

        loader = self._progress(self.loader_train, desc=f'Train Epoch {epoch}')
        for cnt, batch in enumerate(loader, start=1):
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(obs_traj, pred_traj)

            loss = torch.nan_to_num(output["loss_euclidean_ade"], nan=0.0)
            loss_cum = loss if loss_cum is None else (loss_cum + loss)
            accum_steps += 1
            if (cnt % self.hyper_params.batch_size != 0) and (cnt != len(self.loader_train)):
                continue

            loss_cum = loss_cum / max(accum_steps, 1)
            loss_cum.backward()

            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)

            self.optimizer.step()
            loss_batch += float(loss_cum.item())
            num_batches += 1.0
            loss_cum = None
            accum_steps = 0

        global_loss, global_denom = self._reduce_sum_count(loss_batch, max(num_batches, 1.0))
        self.log['train_loss'].append(global_loss / max(global_denom, 1e-12))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_sum = 0.0
        num_ped = 0.0

        loader = self._progress(self.loader_val, desc=f'Valid Epoch {epoch}')
        for batch in loader:
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            output = self.model(obs_traj, pred_traj)
            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_sum += float(recon_loss.item())
            num_ped += float(obs_traj.size(0))

        global_loss_sum, global_num_ped = self._reduce_sum_count(loss_sum, num_ped)
        self.log['val_loss'].append(global_loss_sum / max(global_num_ped, 1e-12))

    @torch.no_grad()
    def test(self, loader=None, split="Test", eval_k=5, miss_threshold=2.0, nan_fill="nan"):
        self.model.eval()
        eval_loader = self.loader_test if loader is None else loader
        if eval_loader is None:
            raise RuntimeError("Test loader is unavailable. Provide `loader` explicitly or prepare test split.")
        desc = f"{split} {self.hyper_params.dataset.upper()} scene"
        eval_k = int(eval_k)
        miss_threshold = float(miss_threshold)
        minade_list, minfde_list, miss_list = [], [], []
        scene_minade_list, scene_minfde_list, scene_miss_list = [], [], []

        loader_iter = self._progress(eval_loader, desc=desc)
        for batch in loader_iter:
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            output = self.model(obs_traj)
            minade, minfde, miss = self._compute_digir_batch_metrics(
                output["recon_traj"], pred_traj, eval_k=eval_k, miss_threshold=miss_threshold
            )
            minade_list.append(minade)
            minfde_list.append(minfde)
            miss_list.append(miss)
            scene_id = self._extract_scene_id(batch, minade.shape[0])
            scene_minade, scene_minfde, scene_miss = self._compute_scene_batch_metrics(minade, minfde, miss, scene_id)
            scene_minade_list.append(scene_minade)
            scene_minfde_list.append(scene_minfde)
            scene_miss_list.append(scene_miss)

        return self._finalize_digir_metrics(
            minade_list,
            minfde_list,
            miss_list,
            scene_minade_list=scene_minade_list,
            scene_minfde_list=scene_minfde_list,
            scene_miss_list=scene_miss_list,
            eval_k=eval_k,
            nan_fill=nan_fill,
        )


class STCollatedMiniBatchTrainer(STTrainer):
    r"""Base class using collated mini-batch training strategy"""

    def __init__(self, args, hyper_params):
        super().__init__(args, hyper_params)

        # Dataset preprocessing
        batch_size = hyper_params.batch_size
        obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
        num_workers = int(getattr(args, "num_workers", 0))
        pin_memory = bool(getattr(args, "pin_memory", True))
        seed = int(getattr(args, "seed", 0))
        interaction_data_path = str(getattr(hyper_params, "interaction_data_path", "") or "")
        self.loader_train = get_dataloader(
            self.dataset_dir, 'train', obs_len, pred_len, batch_size=batch_size, skip=skip,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed,
            interaction_data_path=interaction_data_path)
        self.loader_val = get_dataloader(
            self.dataset_dir, 'val', obs_len, pred_len, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed,
            interaction_data_path=interaction_data_path)
        self.loader_test = None
        try:
            self.loader_test = get_dataloader(
                self.dataset_dir, 'test', obs_len, pred_len, batch_size=1,
                num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
                rank=self.rank, world_size=self.world_size, seed=seed,
                interaction_data_path=interaction_data_path)
        except KeyError:
            if bool(getattr(args, "test", False)):
                raise
            if self.is_main:
                print("Warning: test split not found; training continues without loader_test.")

    def train(self, epoch):
        self.model.train()
        loss_batch = 0.0
        num_batches = 0.0

        loader = self._progress(self.loader_train, desc=f'Train Epoch {epoch}')
        for batch in loader:
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(obs_traj, pred_traj)

            loss = torch.nan_to_num(output["loss_euclidean_ade"], nan=0.0)
            loss_batch += float(loss.item())
            num_batches += 1.0

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        global_loss, global_num_batches = self._reduce_sum_count(loss_batch, num_batches)
        self.log['train_loss'].append(global_loss / max(global_num_batches, 1e-12))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_sum = 0.0
        num_ped = 0.0

        loader = self._progress(self.loader_val, desc=f'Valid Epoch {epoch}')
        for batch in loader:
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            output = self.model(obs_traj, pred_traj)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_sum += float(recon_loss.item())
            num_ped += float(obs_traj.size(0))

        global_loss_sum, global_num_ped = self._reduce_sum_count(loss_sum, num_ped)
        self.log['val_loss'].append(global_loss_sum / max(global_num_ped, 1e-12))

    @torch.no_grad()
    def test(self, loader=None, split="Test", eval_k=5, miss_threshold=2.0, nan_fill="nan"):
        self.model.eval()
        eval_loader = self.loader_test if loader is None else loader
        if eval_loader is None:
            raise RuntimeError("Test loader is unavailable. Provide `loader` explicitly or prepare test split.")
        desc = f"{split} {self.hyper_params.dataset.upper()} scene"
        eval_k = int(eval_k)
        miss_threshold = float(miss_threshold)
        minade_list, minfde_list, miss_list = [], [], []
        scene_minade_list, scene_minfde_list, scene_miss_list = [], [], []

        loader_iter = self._progress(eval_loader, desc=desc)
        for batch in loader_iter:
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            output = self.model(obs_traj)
            minade, minfde, miss = self._compute_digir_batch_metrics(
                output["recon_traj"], pred_traj, eval_k=eval_k, miss_threshold=miss_threshold
            )
            minade_list.append(minade)
            minfde_list.append(minfde)
            miss_list.append(miss)
            scene_id = self._extract_scene_id(batch, minade.shape[0])
            scene_minade, scene_minfde, scene_miss = self._compute_scene_batch_metrics(minade, minfde, miss, scene_id)
            scene_minade_list.append(scene_minade)
            scene_minfde_list.append(scene_minfde)
            scene_miss_list.append(scene_miss)

        return self._finalize_digir_metrics(
            minade_list,
            minfde_list,
            miss_list,
            scene_minade_list=scene_minade_list,
            scene_minfde_list=scene_minfde_list,
            scene_miss_list=scene_miss_list,
            eval_k=eval_k,
            nan_fill=nan_fill,
        )


class STTransformerDiffusionTrainer(STCollatedMiniBatchTrainer):
    r"""SingularTrajectory model trainer"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)
        cfg = DotDict({
            'scheduler': 'ddim',
            'steps': 10,
            'beta_start': 1.e-4,
            'beta_end': 5.e-2,
            'beta_schedule': 'linear',
            'k': hyper_params.k,
            's': hyper_params.num_samples
        })
        predictor_model = base_model(cfg).to(self.device)
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).to(self.device)
        if self.distributed:
            self.model = DDP(
                eigentraj_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        else:
            self.model = eigentraj_model
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=hyper_params.lr,
                                           weight_decay=hyper_params.weight_decay)

        if hyper_params.lr_schd:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                             step_size=hyper_params.lr_schd_step,
                                                             gamma=hyper_params.lr_schd_gamma)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0.0
        num_batches = 0.0

        if self.loader_train.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_train.dataset)

        loader = self._progress(self.loader_train, desc=f'Train Epoch {epoch}')
        for batch in loader:
            obs_traj = batch["obs_traj"].to(self.device, non_blocking=True)
            pred_traj = batch["pred_traj"].to(self.device, non_blocking=True)
            adaptive_anchor = batch["anchor"].to(self.device, non_blocking=True)
            scene_mask = batch["scene_mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info=additional_information)

            loss = torch.nan_to_num(output["loss_euclidean_ade"], nan=0.0)
            loss_batch += float(loss.item())
            num_batches += 1.0

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        global_loss, global_num_batches = self._reduce_sum_count(loss_batch, num_batches)
        self.log['train_loss'].append(global_loss / max(global_num_batches, 1e-12))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_sum = 0.0
        num_ped = 0.0

        if self.loader_val.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_val.dataset)

        loader = self._progress(self.loader_val, desc=f'Valid Epoch {epoch}')
        for batch in loader:
            obs_traj = batch["obs_traj"].to(self.device, non_blocking=True)
            pred_traj = batch["pred_traj"].to(self.device, non_blocking=True)
            adaptive_anchor = batch["anchor"].to(self.device, non_blocking=True)
            scene_mask = batch["scene_mask"].to(self.device, non_blocking=True)

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info=additional_information)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_sum += float(recon_loss.item())
            num_ped += float(obs_traj.size(0))

        global_loss_sum, global_num_ped = self._reduce_sum_count(loss_sum, num_ped)
        self.log['val_loss'].append(global_loss_sum / max(global_num_ped, 1e-12))

    @torch.no_grad()
    def test(self, loader=None, split="Test", eval_k=5, miss_threshold=2.0, nan_fill="nan"):
        self.model.eval()
        eval_loader = self.loader_test if loader is None else loader
        if eval_loader is None:
            raise RuntimeError("Test loader is unavailable. Provide `loader` explicitly or prepare test split.")
        desc = f"{split} {self.hyper_params.dataset.upper()} scene"
        eval_k = int(eval_k)
        miss_threshold = float(miss_threshold)
        minade_list, minfde_list, miss_list = [], [], []
        scene_minade_list, scene_minfde_list, scene_miss_list = [], [], []

        if eval_loader.dataset.anchor is None:
            self.init_adaptive_anchor(eval_loader.dataset)

        loader_iter = self._progress(eval_loader, desc=desc)
        for batch in loader_iter:
            obs_traj = batch["obs_traj"].to(self.device, non_blocking=True)
            pred_traj = batch["pred_traj"].to(self.device, non_blocking=True)
            adaptive_anchor = batch["anchor"].to(self.device, non_blocking=True)
            scene_mask = batch["scene_mask"].to(self.device, non_blocking=True)

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, adaptive_anchor, addl_info=additional_information)
            minade, minfde, miss = self._compute_digir_batch_metrics(
                output["recon_traj"], pred_traj, eval_k=eval_k, miss_threshold=miss_threshold
            )
            minade_list.append(minade)
            minfde_list.append(minfde)
            miss_list.append(miss)
            scene_id = self._extract_scene_id(batch, minade.shape[0])
            scene_minade, scene_minfde, scene_miss = self._compute_scene_batch_metrics(minade, minfde, miss, scene_id)
            scene_minade_list.append(scene_minade)
            scene_minfde_list.append(scene_minfde)
            scene_miss_list.append(scene_miss)

        return self._finalize_digir_metrics(
            minade_list,
            minfde_list,
            miss_list,
            scene_minade_list=scene_minade_list,
            scene_minfde_list=scene_minfde_list,
            scene_miss_list=scene_miss_list,
            eval_k=eval_k,
            nan_fill=nan_fill,
        )
