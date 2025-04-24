# Disclaimer: This code was influenced by
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import math
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional, Union
from .utils import randn_tensor
#from imu_diffusion.refactor.utils import unnormalize_to_zero_to_one


def cosine_beta_schedule(timesteps, beta_start=0.0, beta_end=0.999, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample=True,
        set_alpha_to_one=True,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.num_infer_timesteps = num_train_timesteps
        # self.clip_sample = clip_sample
        # self.betas = (
        #     cosine_beta_schedule(num_train_timesteps, beta_start, beta_end)
        #     if beta_schedule == "cosine"
        #     else torch.linspace(
        #         beta_start, beta_end, num_train_timesteps, dtype=torch.float32
        #     )
        # )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    # def _set_timesteps(self, num_inference_steps, offset=0):
    #     self.timesteps = (
    #         np.arange(
    #             0,
    #             self.num_train_timesteps,
    #             self.num_train_timesteps // num_inference_steps,
    #         )[::-1] + offset
    #     )

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev/ beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        
        dynamic_thresholding_ratio = 0.995
        
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=1.0
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        timestep_spacing = "leading"
        
        if num_inference_steps > self.num_inference_steps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.train_timesteps`:"
                f" {self.num_inference_steps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_inference_steps} timesteps."
            )

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_inference_steps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif timestep_spacing == "leading":
            step_ratio = self.num_inference_steps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += 0
        elif timestep_spacing == "trailing":
            step_ratio = self.num_inference_steps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.num_inference_steps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _step(self, model_output, timestep, sample, eta=0.0, generator=None, variance_noise=None):
        prediction_type = "epsilon"
        thresholding = False
        clip_sample = True
        clip_sample_range = 1.0
        
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_infer_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
    
        # 4. Clip or threshold "predicted x_0"
        if thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -clip_sample_range, clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
        # self.alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        # sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        # sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        # return (
        #     sqrt_alpha_prod[:, None, None, None] * original_samples
        #     + sqrt_one_minus_alpha_prod[:, None, None, None] * noise
        # )

    @torch.no_grad()
    def generate(
        self,
        model,
        cond,
        batch_size=1,
        seq_len=200,  # The length of the sequence to generate
        generator=None,
        eta=0.0,
        num_inference_steps=50,
        device=None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        #image = torch.randn((batch_size, seq_len, 6), generator=generator).to(device)  # Adjust the shape based on your model
        image = randn_tensor((batch_size, seq_len, 6))
        image = image.to(device)

        self.set_timesteps(num_inference_steps)

        #iterate over the timesteps as the starting point for the DDIM sampling
        for t in tqdm(self.timesteps):
            # 1. Predict the noise at the current timestep
            model_output = model(image, t.to(device), global_cond_input=cond)[0]

            # 2. Perform the denoising step: x_t -> x_t-1
            image = self._step(model_output, t, image, eta=eta, generator=generator)
        
        return image

    def __len__(self):
        return self.num_infer_timesteps

    # @torch.no_grad()
    # def generate(
    #     self,
    #     model,
    #     batch_size=1,
    #     generator=None,
    #     eta=1.0,
    #     num_inference_steps=50,
    #     device=None,
    # ):
    #     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    #     image = torch.randn(
    #         (batch_size, model.in_channels, model.sample_size, model.sample_size),
    #         generator=generator,
    #     ).to(device)
    #     self._set_timesteps(num_inference_steps)

    #     for t in tqdm(self.timesteps):
    #         model_output = model(image, t)["sample"]
    #         # predict previous mean of image x_t-1 and add variance depending on eta
    #         # do x_t -> x_t-1
    #         image = self._step(model_output, t, image, eta, generator=generator)
        
    #     image = unnormalize_to_zero_to_one(image)
    #     return {"sample": image.cpu().permute(0, 2, 3, 1).numpy(), "sample_pt": image}

    # def __len__(self):
    #     return self.num_train_timesteps
