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
        self.dataset_dir = hyper_params.dataset_dir + hyper_params.dataset + '/'
        self.checkpoint_dir = hyper_params.checkpoint_dir + '/' + args.tag + '/' + hyper_params.dataset + '/'
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
                with open(self.checkpoint_dir + 'args.pkl', 'wb') as fp:
                    pickle.dump(args, fp)

                with open(self.checkpoint_dir + 'config.pkl', 'wb') as fp:
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
        for value in self.unwrap_model().state_dict().values():
            if torch.is_tensor(value):
                dist.broadcast(value, src=0)

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

    @staticmethod
    def _unpack_obs_pred_batch(batch):
        if isinstance(batch, dict):
            return batch["obs_traj"], batch["pred_traj"]
        return batch[:2]

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
    def test(self):
        raise NotImplementedError

    def fit(self):
        if self.is_main:
            print("Training started...")
        best_val = float('inf')

        for epoch in range(self.hyper_params.num_epochs):
            self._set_epoch(epoch)
            self.train(epoch)
            self.valid(epoch)

            if self.hyper_params.lr_schd:
                self.scheduler.step()

            current_val = self.log['val_loss'][-1]
            if current_val < best_val:
                best_val = current_val
                self.save_model()

            if self.is_main:
                print(" ")
                print("Dataset: {0}, Epoch: {1}".format(self.hyper_params.dataset, epoch))
                print("Train_loss: {0:.8f}, Val_los: {1:.8f}".format(self.log['train_loss'][-1], current_val))
                print("Min_val_epoch: {0}, Min_val_loss: {1:.8f}".format(np.array(self.log['val_loss']).argmin(),
                                                                         np.array(self.log['val_loss']).min()))
                print(" ")
            self.barrier()

        if self.is_main:
            print("Done.")

    def reset_metric(self):
        self.stats_func = {'ADE': compute_batch_ade, 'FDE': compute_batch_fde}
        self.stats_meter = {x: AverageMeter() for x in self.stats_func.keys()}

    def get_metric(self):
        return self.stats_meter

    def load_model(self, filename='model_best.pth'):
        model_path = self.checkpoint_dir + filename
        state_dict = torch.load(model_path, map_location=self.device)
        self.unwrap_model().load_state_dict(state_dict)
        self.barrier()

    def save_model(self, filename='model_best.pth'):
        if not self.is_main:
            return
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        model_path = self.checkpoint_dir + filename
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
        self.loader_train = get_dataloader(
            self.dataset_dir, 'train', obs_len, pred_len, batch_size=1, skip=skip,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed)
        self.loader_val = get_dataloader(
            self.dataset_dir, 'val', obs_len, pred_len, batch_size=1,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed)
        self.loader_test = get_dataloader(
            self.dataset_dir, 'test', obs_len, pred_len, batch_size=1,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0.0
        loss_cum = None
        num_batches = 0.0

        loader = self._progress(self.loader_train, desc=f'Train Epoch {epoch}')
        for cnt, batch in enumerate(loader, start=1):
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(obs_traj, pred_traj)

            loss = torch.nan_to_num(output["loss_euclidean_ade"], nan=0.0)
            if (cnt % self.hyper_params.batch_size != 0) and (cnt != len(self.loader_train)):
                loss_cum = loss if loss_cum is None else (loss_cum + loss)
                continue

            loss_cum = loss if loss_cum is None else (loss_cum + loss)
            loss_cum = loss_cum / self.hyper_params.batch_size
            loss_cum.backward()

            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)

            self.optimizer.step()
            loss_batch += float(loss_cum.item())
            num_batches += 1.0
            loss_cum = None

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
    def test(self):
        self.model.eval()
        self.reset_metric()

        loader = self._progress(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene")
        for batch in loader:
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return self._collect_metric_means()


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
        self.loader_train = get_dataloader(
            self.dataset_dir, 'train', obs_len, pred_len, batch_size=batch_size, skip=skip,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed)
        self.loader_val = get_dataloader(
            self.dataset_dir, 'val', obs_len, pred_len, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed)
        self.loader_test = get_dataloader(
            self.dataset_dir, 'test', obs_len, pred_len, batch_size=1,
            num_workers=num_workers, pin_memory=pin_memory, distributed=self.distributed,
            rank=self.rank, world_size=self.world_size, seed=seed)

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
    def test(self):
        self.model.eval()
        self.reset_metric()

        loader = self._progress(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene")
        for batch in loader:
            obs_traj, pred_traj = self._unpack_obs_pred_batch(batch)
            obs_traj = obs_traj.to(self.device, non_blocking=True)
            pred_traj = pred_traj.to(self.device, non_blocking=True)

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return self._collect_metric_means()


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
    def test(self):
        self.model.eval()
        self.reset_metric()

        if self.loader_test.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_test.dataset)

        loader = self._progress(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene")
        for batch in loader:
            obs_traj = batch["obs_traj"].to(self.device, non_blocking=True)
            pred_traj = batch["pred_traj"].to(self.device, non_blocking=True)
            adaptive_anchor = batch["anchor"].to(self.device, non_blocking=True)
            scene_mask = batch["scene_mask"].to(self.device, non_blocking=True)

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, adaptive_anchor, addl_info=additional_information)

            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return self._collect_metric_means()
