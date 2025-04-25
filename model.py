# import os
import math
import pickle
# from itertools import chain

from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
import os.path as osp

import numpy as np
# from torchvision import transforms, utils as tv_utils
from lightning.pytorch import callbacks, Trainer, LightningModule


from airimu_datasets.dataset import SeqeuncesDataset, SeqInfDataset, SeqDataset

import pypose as pp

# from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler
# from diffusers.pipelines.ddim.pipeline_ddim_override import DDIMOverridePipeline

# from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from scheduler.ddim import DDIMScheduler
from scheduler.ddpm import DDPMScheduler
from backbones.unet import UNet1DConditional
from backbones.rnn import RNNConditionalNetwork


from torch.optim.lr_scheduler import LambdaLR


import torch.utils.data as Data

from integrate import batch_integrate

from collections import defaultdict
# some global stuff necessary for the program
# torch.set_float32_matmul_precision('medium')
# to_tensor = transforms.ToTensor()
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
# import wandb
from data import AirIMUData
import argparse
# from diffusers.optimization import get_cosine_schedule_with_warmup
# import prettytable
from backbones.rnn import RNNConditionalNetwork


# TODO: handle this global variable better
data_cfg_root = "/home/shzhou/project/inertia_only/ioo_diffusion_merged/ioo_diffusion/config/airimu_data/exp/"


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """
    Taken from diffusers/optimization.py
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# class PipelineCheckpoint(callbacks.ModelCheckpoint):

#     def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint) -> None:
#         # only ema parameters (if any) saved in pipeline
#         # with pl_module.maybe_ema():
#         pipe_path = osp.join(
#             osp.dirname(self.best_model_path),
#             'pipeline'
#         )
#         pl_module.save_pretrained(pipe_path)

    # different from original implementation, we just save the pipeline, not the model
    # return super().on_save_checkpoint(trainer, pl_module, checkpoint)


def infer_window(dataset_conf, model, batch_size, collate_fn, seqlen=200):
    dataset_conf.data_list[0]["window_size"] = seqlen
    dataset_conf.data_list[0]["step_size"] = seqlen
    net_out_result = {}
    wandb_logger = WandbLogger(log_model="all")
    trainer = Trainer(logger=wandb_logger)
    for data_conf in dataset_conf.data_list:
        for path in data_conf.data_drive:
            dataset_conf["mode"] = "infevaluate"
            eval_dataset = SeqeuncesDataset(
                data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
            eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=collate_fn, drop_last=False)
            trainer.test(model, dataloaders=eval_loader)
            # print(model.evaluate_states.keys())
            net_out_result[path] = model.evaluate_states
    # with open("pl_infer_my_collate.pkl", 'wb') as handle:
    #     pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_model_window(dataset_conf, inference_state, seqlen, device):
    """
    This should replace evaluate.py, for evaluate performance in the window, in a batch fashion
    It uses batch_integrate as in evaluate_test.py, to efficiently evaluate all windows in a trajectory at once
    """
    # TODO: need to verify the result to match original implementation
    start_idx = 0
    res = {}

    for data_conf in dataset_conf.data_list:
        print(data_conf)
        for data_name in data_conf.data_drive:
            dummy_dataset = SeqDataset(data_conf.data_root, data_name, device, name=data_conf.name,
                                       duration=seqlen, step_size=seqlen, drop_last=False, conf=dataset_conf)
            offset = len(dummy_dataset)
            inf_state_seq = {}
            for k, v in inference_state.items():
                inf_state_seq[k] = v[start_idx:start_idx+offset]
            start_idx += offset
            # TODO: do we really need SeqInDataset, which merely adds the correction to the raw reading.
            # we might want to make it explicit here
            test_len = 200
            dataset = SeqInfDataset(data_conf.data_root, data_name, inf_state_seq, device=device,
                                    name=data_conf.name, duration=test_len, step_size=test_len, drop_last=False,
                                    conf=dataset_conf)
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()

            integrator = pp.module.IMUPreintegrator(init['pos'], init['rot'], init['vel'],
                                                    gravity=gravity,
                                                    reset=True
                                                    ).double()
            relative_infstate = batch_integrate(integrator=integrator,
                                                dataset=dataset, init=init, gtinit=True, use_gt_rot=False,
                                                device=device)

            # print(relative_infstate.keys())

            # FIXME: apply the mask to the infered state
            # index_id = dataset.index_map[:, -1]
            # mask = dataset.get_mask()
            # select_mask = mask[index_id]

            # for k, v in relative_infstate.items():
            #     if "dist" in k:
            #         relative_infstate[k] = v[0, select_mask]

            print(data_name)
            print("pos dist: ", relative_infstate['pos_dist'].mean().item())
            res[data_name] = relative_infstate
    return res


def compute_metric(res):
    metric_res = {"rpe": {}, "rp_rmse": {}, "roe": {}}
    for data_name, data in res.items():
        rpe = data['pos_dist'].mean().item()
        rp_rmse = torch.sqrt((data['pos_dist']**2).mean()).item()
        roe = 180./np.pi * data['rot_dist'].mean().item()
        metric_res['rpe'][data_name] = rpe
        metric_res['rp_rmse'][data_name] = rp_rmse
        metric_res['roe'][data_name] = roe
    return metric_res


# def make_table_from_metric(metric_res):
#     tables = []
#     for metric_name, res in metric_res.items():
#         table = wandb.Table(columns=["name", metric_name])
#         for data_name, value in res.items():
#             table.add_data(data_name, value)
#         # print(table)
#         tables.append((table, metric_name))
#     return tables


class IMUDiffusion(LightningModule):
    def __init__(self,
                 cfg
                 #  models_cfg: DictConfig,
                 #  training_cfg: DictConfig,
                 #  inference_cfg: DictConfig
                 ):
        super().__init__()

        # self.training_cfg = training_cfg
        # self.inference_cfg = inference_cfg
        if cfg.model_type == "unet":
            self.model = UNet1DConditional(input_dim=6, global_cond_dim=256)
        elif cfg.model_type == "rnn":
            self.model = RNNConditionalNetwork()
        else:
            raise ValueError(f"Unsupported model type: {cfg.model_type}")
        self.train_noise_scheduler = DDPMScheduler()
        self.infer_noise_scheduler = DDIMScheduler()
        self.n_inference_timesteps = 25

        # FIXME: temporarily, hard code cfg in dict
        self.cfg = cfg
        self.cfg = DictConfig(self.cfg)
        self.save_hyperparameters(self.cfg)

    # def temp_load_from_ckpt(self, ckpt_path: str):
    #     # saved ckpt_path is the pipeline path, we can recover the full HF pipeline
    #     pipeline = DDPMPipeline.from_pretrained(
    #         ckpt_path, use_safetensors=False)
    #     self.model = pipeline.unet

    def fetch_data_from_batch_train(self, batch):
        acc = batch['acc']
        gyro = batch['gyro']
        imu_reading = torch.cat([gyro, acc], dim=-1).float()
        bias_gyro = batch['bias_gyro']
        bias_acc = batch['bias_acc']
        bias = torch.cat([bias_gyro, bias_acc], dim=-1).float()
        return imu_reading, bias

    def fetch_data_from_batch_infer(self, batch):
        # because of custom collate_fn, we need to extract the data from the tuple first
        if isinstance(batch, tuple):
            batch = batch[0]
        acc = batch['acc']
        gyro = batch['gyro']
        imu_reading = torch.cat([gyro, acc], dim=-1)
        return imu_reading

    def training_step(self, batch, batch_idx):
        imu_reading, bias = self.fetch_data_from_batch_train(batch)
        if self.cfg.mode == "frame_rate":
            # repeat 4 times to make the network happy, since 4 is the minimum size it accepts
            bias = bias[:, 0:1, :].repeat(1, 4, 1)

        # if self.cfg.out_len < 4:
        #     # repeat 4 times to make the network happy, since 4 is the minimum size it accepts
        #     # it makes more sense to mean the last bias in the window
        #     bias = bias[:, -1:, :].repeat(1, 4, 1)
        # else:
        #     bias = bias[:, -self.cfg.out_len:, :]

        noise = torch.randn_like(bias)

        timesteps = torch.randint(
            low=0,
            high=self.train_noise_scheduler.num_train_timesteps,
            size=(bias.size(0), ), device=self.device
        ).long()
        noisy_bias = self.train_noise_scheduler.add_noise(
            bias, noise, timesteps)

        # Predict the noise residual
        global_cond_input = imu_reading
        model_output = self.model(
            noisy_bias, timesteps, global_cond_input=global_cond_input)[0]

        # FIXME: should we take the loss on only the first dimension, for frame rate case
        # if self.cfg.mode == "frame_rate":
        #     model_output = model_output[:, 0:1, :]
        #     noise = noise[:, 0:1, :]

        # if self.cfg.out_len < 4:
        #     model_output = model_output[:, :self.cfg.out_len, :]
        #     noise = noise[:, :self.cfg.out_len:, :]

        loss = F.mse_loss(model_output, noise)

        log_key = f'{"train" if self.training else "val"}/loss'
        self.log_dict({log_key: loss},
                      prog_bar=True, sync_dist=True,
                      on_step=self.training,
                      on_epoch=not self.training)

        return loss

    def on_validation_start(self):
        self.evaluate_val_states = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        imu_reading = self.fetch_data_from_batch_infer(batch)
        bias_denoised = self.sample(imu_reading)
        assert bias_denoised.shape == imu_reading.shape
        self.evaluate_val_states["correction_gyro"].append(
            -bias_denoised[:, :, :3])
        self.evaluate_val_states["correction_acc"].append(
            -bias_denoised[:, :, 3:])

    def on_validation_epoch_end(self):
        for k, v in self.evaluate_val_states.items():
            self.evaluate_val_states[k] = torch.cat(v, dim=0)
        res = evaluate_model_window(dataset_conf=self.trainer.datamodule.airimu_conf.dataset.inference,
                                    inference_state=self.evaluate_val_states, seqlen=self.cfg.seqlen, device=self.device)
        res = compute_metric(res)
        # self.log_dict(res)
        # tables = make_table_from_metric(res)
        # for table, name in tables:
        #     # print(name)
        #     # print(table.data)
        #     pt = prettytable.PrettyTable()
        #     pt.field_names = table.columns
        #     pt.add_rows(table.data)
        #     print(pt)
        for metric_name, record in res.items():
            self.logger.log_metrics(
                {f"{metric_name}": np.mean(list(record.values())[:4])},
                step=self.global_step
            )
        pos_metric = np.mean(list(res['rp_rmse'].values())[:4])
        rot_metric = np.mean(list(res['roe'].values())[:4])
        metric = pos_metric * 3 + rot_metric
        # self.logger.log_metrics(
        #     {"total_metric": metric}, step=self.global_step)
        self.log("total_metric", metric)

    def on_test_start(self):
        self.evaluate_states = defaultdict(list)

    def test_step(self, batch, batch_idx):
        # TODO: this should replace infer.py::inference
        # (batch_size, window_size, 6)

        # TODO: currently, the evaluation step requires batch size to be 1, so that the padding of the last batch will work.
        # This is a bad design... and it serverly impacts the performance. Let's fix it if we have time.
        imu_reading = self.fetch_data_from_batch_infer(batch)
        bias_denoised = self.sample(imu_reading)
        # inte_state = {
        #     "correction_gyro": -bias_denoised[:, :, :3],
        #     "correction_acc": -bias_denoised[:, :, 3:],
        # }
        # # TODO: get the logic straight here
        # self.update_state(self.evaluate_states, inte_state)
        # assert bias_denoised.shape == imu_reading.shape
        self.evaluate_states["correction_gyro"].append(
            -bias_denoised[:, :, :3])
        self.evaluate_states["correction_acc"].append(-bias_denoised[:, :, 3:])

    def on_test_end(self):
        """
        When the testing finishes, the evaluated network output is saved.
        Then we should start the evaluation to get the score
        """
        # TODO: save evaluate_states to a file
        for k, v in self.evaluate_states.items():
            self.evaluate_states[k] = torch.cat(v, dim=0)
        # self.save_state(self.evaluate_states)
        # evaluate right away if requested
        if self.cfg.eval_on_test:
            # seqlen is hard-coded to 1 second window (200)
            res = evaluate_model_window(dataset_conf=self.trainer.datamodule.airimu_conf.dataset.inference,
                                        inference_state=self.evaluate_states, seqlen=self.cfg.seqlen, device=self.device)
            res = compute_metric(res)
            # self.log_dict(res)
            # tables = make_table_from_metric(res)
            # for table, name in tables:
            #     # print(name)
            #     # print(table.data)
            #     pt = prettytable.PrettyTable()
            #     pt.field_names = table.columns
            #     pt.add_rows(table.data)
            #     print(pt)
            #     self.logger.log_table(
            #         key=name, columns=table.columns, data=table.data)

        # TODO: log the evaluation result (e.g. with w&b)
        # return self.evaluate_states

    def save_state(self, states):
        with open("evaluate_states.pkl", 'wb') as f:
            pickle.dump(states, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def sample(self, imu_reading, **kwargs: dict):
        bias_denoised = self.infer_noise_scheduler.generate(model=self.model,
                                                            cond=imu_reading.float(),
                                                            seq_len=4 if self.cfg.mode == "frame_rate" else self.cfg.seqlen,
                                                            num_inference_steps=self.n_inference_timesteps,
                                                            batch_size=imu_reading.shape[0])
        if self.cfg.mode == "frame_rate":
            # in this case, only the first dimension is meaningful
            bias_denoised = bias_denoised[:, 0:1, :].repeat(
                1, imu_reading.shape[1], 1)
        return bias_denoised

    def save_pretrained(self, path: str, push_to_hub: bool = False):
        # self._fix_hydra_config_serialization()

        pipe = self.get_train_pipeline()
        pipe.save_pretrained(path, safe_serialization=False,
                             push_to_hub=push_to_hub)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.learning_rate)
        # sched = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.99)
        # sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
        # sched = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=self.cfg.learning_rate * 0.01,
        #                                        max_lr=self.cfg.learning_rate, mode='triangular2')
        sched = get_cosine_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=100,
            num_training_steps=(self.trainer.estimated_stepping_batches),
        )
        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': sched, 'interval': 'step'}
        }


# def main(cfg: DictConfig):
#     OmegaConf.resolve(cfg)  # resolve all string interpolation
#     system = IMUDiffusion(cfg)
#     # system = IMUDiffusion(cfg.models, cfg.training, cfg.inference)
#     # datamodule = ImageDatasets(cfg.data)\

#     trainer = Trainer(
#         gradient_clip_val=1.0,  # clip_grad_norm_ for diffusion model
#         callbacks=[
#             callbacks.LearningRateMonitor(
#                 'epoch', log_momentum=True, log_weight_decay=True),
#             # PipelineCheckpoint(mode='min', monitor='FID'),
#             # PipelineCheckpoint(),
#             callbacks.RichProgressBar()
#         ],
#         # logger=hy.utils.instantiate(cfg.logger, _recursive_=True),
#         **cfg.pl_trainer
#     )

#     # FIXME: we need to move dataset related config to the new config system
#     train_loader = SeqeuncesDataset(
#         data_set_config=cfg.dataset.train)

#     trainer.fit(system, train_dataloaders=train_loader,
#                 ckpt_path=cfg.resume_from_checkpoint
#                 )


def train(name, dataset_name, rate_mode, ckpt_path=None):
    # TODO: prepare configs: cfg.models, cfg.training, cfg.inference

    # seqlen is the window size for IMU reading input (context)
    # out_len is the window size for predicted length, which may be different from seqlen
    # for frame rate, out_len = 1, for imu rate, out_len = seqlen
    # additionally, for the network, the minimal input size is 4
    seqlen = 200
    batch_size = 1024
    overlap_factor = 100
    model_cfg = {"eval_on_test": True, "seqlen": seqlen,
                 #  "out_len": 1,
                 "mode": rate_mode,
                 "learning_rate": 8e-5,
                 "model_type": "unet",
                 }
    model_cfg = DictConfig(model_cfg)
    system = IMUDiffusion(model_cfg)
    # datamodule = ImageDatasets(cfg.data)\
    # tb_logger = TensorBoardLogger("lightning_logs", name=name)
    # training log with tensorboard by default
    trainer = Trainer(
        max_epochs=20000,
        gradient_clip_val=1.0,  # clip_grad_norm_ for diffusion model
        # logger=tb_logger,
        devices=1, num_nodes=1,
        check_val_every_n_epoch=5,
        callbacks=[
            callbacks.LearningRateMonitor(
                'epoch', log_momentum=True, log_weight_decay=True),
            # PipelineCheckpoint(
            #     save_last=True, monitor='total_metric', mode='min', save_top_k=5),
        ],
        # logger=hy.utils.instantiate(cfg.logger, _recursive_=True),
        # **cfg.pl_trainer
    )

    # FIXME: we need to move dataset related config to the new config system

    data_cfg = {"seqlen": seqlen, "batch_size": batch_size,
                "data_name": dataset_name, "overlap_factor": overlap_factor}
    data_cfg = DictConfig(data_cfg)
    data_cfg_path = osp.join(data_cfg_root, f"{dataset_name}/codenet.conf")

    data_module = AirIMUData(
        data_cfg, data_cfg_path)

    trainer.fit(system, datamodule=data_module,
                ckpt_path=ckpt_path
                )


def infer(model_path, dataset_name, rate_mode):
    # model_path = "ddpm-sim-20040-test"
    # data_cfg_path = "/home/shzhou/project/inertia_only/AirIMU/configs/exp/EuRoC/codenet.conf"

    # model_path = "ddim-tumvi"
    # data_cfg_path = "/home/shzhou/project/inertia_only/AirIMU/configs/exp/TUMVI/codenet.conf"

    seq_len = 200

    data_cfg_path = osp.join(data_cfg_root, f"{dataset_name}/codenet.conf")

    model_cfg = {"eval_on_test": True, "seqlen": seq_len,
                 "model_type": "unet",
                 "mode": rate_mode, "learning_rate": 3e-5,
                 }
    model_cfg = DictConfig(model_cfg)
    model = IMUDiffusion(model_cfg)
    # test may require training with pytorchligntnign in the first place, let's have a temporary workaround
    # model.temp_load_from_ckpt(model_path)
    model.load_from_checkpoint(model_path)


    cfg = {"seqlen": seq_len, "batch_size": 2048, "overlap_factor": 1}
    cfg = DictConfig(cfg)
    data_module = AirIMUData(
        cfg, data_cfg_path)

    # data_module = AirIMUData(
    #     cfg, "/home/shzhou/project/inertia_only/AirIMU/configs/exp/TUMVI/codenet.conf")

    # this result should produce the exactly same table we see before
    # this is equivalent to infer followed by
    # trainer = Trainer()  # dummy trainer for testing only
    # wandb_logger = WandbLogger(log_model="all")
    # trainer = Trainer(logger=wandb_logger, devices=1, num_nodes=1)
    trainer = Trainer(devices=1, num_nodes=1)
    trainer.test(model, datamodule=data_module)

    # infer_window(data_module.airimu_conf.dataset.inference,
    #              model, 2048, data_module.collate_fn)

    # infer_window(data_module.airimu_conf.dataset.inference,
    #              model, 1, collate_fcs[data_module.airimu_conf.dataset.collate])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None,
                        help="train or infer")
    parser.add_argument(
        "--name", type=str, help="for training, it is run name; for infer, it is model path")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--rate_mode", type=str,
                        help="imu_rate or frame_rate")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--use_wandb", action='store_true',
                        help="Use wandb for logging")
    args = parser.parse_args()
    # train(name="euroc_imu_rate_overlapping_100")

    dataset_name = {"euroc": "EuRoC"}[args.dataset.lower()]

    print(f"begin {args.mode}")
    if args.mode == "train":
        train(name=args.name, dataset_name=dataset_name,
              rate_mode=args.rate_mode, ckpt_path=args.ckpt_path)
    elif args.mode == "infer":
        infer(model_path=args.name,
              dataset_name=dataset_name, rate_mode=args.rate_mode)
