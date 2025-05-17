import math
import pickle
import os.path as osp
from collections import defaultdict
import argparse

from omegaconf import DictConfig
import prettytable
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from lightning.pytorch import callbacks, Trainer, LightningModule
import pypose as pp


from airimu_datasets.dataset import SeqInfDataset, SeqDataset
from scheduler.ddim import DDIMScheduler
from scheduler.ddpm import DDPMScheduler
from backbones.unet import UNet1DConditional
from backbones.rnn import RNNConditionalNetwork
from integrate import batch_integrate
from data import AirIMUData
from lightning.pytorch.loggers import TensorBoardLogger




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


def evaluate_model_window(dataset_conf, inference_state, seqlen, device):
    """
    This should replace evaluate.py, for evaluate performance in the window, in a batch fashion
    It uses batch_integrate as in evaluate_test.py, to efficiently evaluate all windows in a trajectory at once
    """
    start_idx = 0
    res = {}

    for data_conf in dataset_conf.data_list:
        for data_name in data_conf.data_drive:
            dummy_dataset = SeqDataset(data_conf.data_root, data_name, device, name=data_conf.name,
                                       duration=seqlen, step_size=seqlen, drop_last=False, conf=dataset_conf)
            offset = len(dummy_dataset)
            inf_state_seq = {}
            for k, v in inference_state.items():
                inf_state_seq[k] = v[start_idx:start_idx+offset]
            start_idx += offset
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





class IMUDiffusion(LightningModule):
    def __init__(self,
                 cfg
                 ):
        super().__init__()

      
        if cfg.model_type == "unet":
            self.model = UNet1DConditional(input_dim=6, global_cond_dim=256)
        elif cfg.model_type == "rnn":
            self.model = RNNConditionalNetwork()
        else:
            raise ValueError(f"Unsupported model type: {cfg.model_type}")
        self.train_noise_scheduler = DDPMScheduler()
        self.infer_noise_scheduler = DDIMScheduler()
        self.n_inference_timesteps = 25

        self.cfg = cfg
        self.cfg = DictConfig(self.cfg)
        self.save_hyperparameters(self.cfg)



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
        for metric_name, record in res.items():
            if metric_name == "roe" or metric_name == "rp_rmse":
                pt = prettytable.PrettyTable()
                pt.field_names = ["name", metric_name]
                
                for data_name, value in record.items():
                    pt.add_rows([[data_name, value]])
                print(pt)
                        
        for metric_name, record in res.items():
            self.logger.log_metrics(
                {f"{metric_name}": np.mean(list(record.values())[:4])},
                step=self.global_step
            )
        pos_metric = np.mean(list(res['rp_rmse'].values())[:4])
        rot_metric = np.mean(list(res['roe'].values())[:4])
        metric = pos_metric * 3 + rot_metric
        self.log("total_metric", metric)

    def on_test_start(self):
        self.evaluate_states = defaultdict(list)

    def test_step(self, batch, batch_idx):
        imu_reading = self.fetch_data_from_batch_infer(batch)
        bias_denoised = self.sample(imu_reading)
       
        self.evaluate_states["correction_gyro"].append(
            -bias_denoised[:, :, :3])
        self.evaluate_states["correction_acc"].append(-bias_denoised[:, :, 3:])

    def on_test_end(self):
        """
        When the testing finishes, the evaluated network output is saved.
        Then we should start the evaluation to get the score
        """
        for k, v in self.evaluate_states.items():
            self.evaluate_states[k] = torch.cat(v, dim=0)
        # evaluate right away if requested
        if self.cfg.eval_on_test:
            # seqlen is hard-coded to 1 second window (200)
            res = evaluate_model_window(dataset_conf=self.trainer.datamodule.airimu_conf.dataset.inference,
                                        inference_state=self.evaluate_states, seqlen=self.cfg.seqlen, device=self.device)
            res = compute_metric(res)

            if args.use_wandb:
                import wandb, prettytable
                def make_table_from_metric(metric_res):
                    tables = []
                    for metric_name, res in metric_res.items():
                        table = wandb.Table(columns=["name", metric_name])
                        for data_name, value in res.items():
                            table.add_data(data_name, value)
                        tables.append((table, metric_name))
                    return tables
                tables = make_table_from_metric(res)
                for table, name in tables:
                    pt = prettytable.PrettyTable()
                    pt.field_names = table.columns
                    pt.add_rows(table.data)
                    print(pt)
                    self.logger.log_table(
                        key=name, columns=table.columns, data=table.data)

       

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

        pipe = self.get_train_pipeline()
        pipe.save_pretrained(path, safe_serialization=False,
                             push_to_hub=push_to_hub)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.learning_rate)
        sched = get_cosine_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=100,
            num_training_steps=(self.trainer.estimated_stepping_batches),
        )
        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': sched, 'interval': 'step'}
        }


def train(name, dataset_name, rate_mode, data_cfg_root, ckpt_path=None):
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
                 "learning_rate": args.lr,
                 "model_type": args.model_type,
                 }
    model_cfg = DictConfig(model_cfg)
    system = IMUDiffusion(model_cfg)
    tb_logger = TensorBoardLogger("lightning_logs", name=name)
    trainer = Trainer(
        max_epochs=20000,
        gradient_clip_val=1.0,  # clip_grad_norm_ for diffusion model
        logger=tb_logger,
        devices=1, num_nodes=1,
        check_val_every_n_epoch=5,
        callbacks=[
            callbacks.LearningRateMonitor(
                'epoch', log_momentum=True, log_weight_decay=True),
            callbacks.ModelCheckpoint(
                 monitor='total_metric', mode='min', save_top_k=3, save_last=True
            )
        ],
    )


    data_cfg = {"seqlen": seqlen, "batch_size": batch_size,
                "data_name": dataset_name, "overlap_factor": overlap_factor}
    data_cfg = DictConfig(data_cfg)
    data_cfg_path = osp.join(data_cfg_root, f"{dataset_name}/codenet.conf")

    data_module = AirIMUData(
        data_cfg, data_cfg_path)

    trainer.fit(system, datamodule=data_module,
                ckpt_path=ckpt_path
                )


def infer(model_path, dataset_name, rate_mode, data_cfg_root):
    seq_len = 200

    data_cfg_path = osp.join(data_cfg_root, f"{dataset_name}/codenet.conf")

    model_cfg = {"eval_on_test": True, "seqlen": seq_len,
                 "model_type": args.model_type,
                 "mode": rate_mode, "learning_rate": args.lr,
                 }
    model_cfg = DictConfig(model_cfg)
    model = IMUDiffusion(model_cfg)
    model.load_from_checkpoint(model_path)


    cfg = {"seqlen": seq_len, "batch_size": 2048, "overlap_factor": 1}
    cfg = DictConfig(cfg)
    data_module = AirIMUData(
        cfg, data_cfg_path)

    if args.use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        wandb_logger = WandbLogger(log_model="all")
        trainer = Trainer(logger=wandb_logger, devices=1, num_nodes=1)
    else:
        trainer = Trainer(devices=1, num_nodes=1)
    trainer.test(model, datamodule=data_module)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--model_type", type=str, default="rnn",
                        help="unet or rnn")
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

    dataset_name = {"euroc": "EuRoC"}[args.dataset.lower()]
    data_cfg_root = "/home/shzhou/project/inertia_only/ioo_diffusion_merged/ioo_diffusion/config/airimu_data/exp/"

    print(f"begin {args.mode}")
    if args.mode == "train":
        train(name=args.name, dataset_name=dataset_name,
              rate_mode=args.rate_mode, data_cfg_root=data_cfg_root, ckpt_path=args.ckpt_path)
    elif args.mode == "infer":
        infer(model_path=args.name,
              dataset_name=dataset_name, rate_mode=args.rate_mode, data_cfg_root=data_cfg_root)
