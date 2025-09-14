import math
from functools import partial
from typing import List, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import DenoiserConfig, SamplerConfig

Conv2D1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv2D3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)


class FourierFeatures(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.weight = nn.Parameter(torch.randn(1, dim // 2), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        f = 2.0 * math.pi * x[:, None] @ self.weight
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class GroupNorm(nn.Module):
    """group数を自動調整するGroupNorm"""

    def __init__(self, in_channels: int, group_size: int = 32, eps: float = 1e-5):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

        groups = self.group_size if in_channels % self.group_size == 0 else 1
        self.norm = nn.GroupNorm(groups, in_channels, eps=self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AdaGroupNorm2D(nn.Module):
    """条件付きスケーリング付きGroupNorm（Adaptive）"""

    def __init__(self, in_channels: int, cond_channels: int, group_size: int = 32, eps: float = 1e-5):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.norm = GroupNorm(in_channels, group_size, eps)
        self.gamma = nn.Linear(cond_channels, in_channels, bias=False)
        self.beta = nn.Linear(cond_channels, in_channels, bias=False)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        gamma = gamma[:, :, None, None]  # (B,C) -> (B,C,1,1)
        beta = beta[:, :, None, None]  # (B,C) -> (B,C,1,1)
        return x * (1 + gamma) + beta


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = 8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.head_dim = head_dim
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0, f"dim must be divisible by head_dim ({head_dim})"

        self.norm = nn.LayerNorm(in_channels, eps=1e-5)
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3)
        self.out_proj = nn.Linear(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_norm = self.norm(x)
        qkv = self.qkv_proj(x_norm)  # (B, T, 3C)
        qkv = qkv.view(x.shape[0], x.shape[1], 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 (B, H, T, D)

        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        attn = attn / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.in_channels)  # (B, T, C)
        return x + self.out_proj(out)


class SelfAttention2D(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = 8) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0, f"in_channels must be divisible by head_dim ({head_dim})"

        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv2D1x1(in_channels, in_channels * 3)
        self.out_proj = Conv2D1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_norm = self.norm(x)  # (B, C, H, W)
        qkv = self.qkv_proj(x_norm)  # (B, 3C, H, W)
        qkv = qkv.view(b, 3, self.n_head, c // self.n_head, h * w)  # (B, 3, H, D, HW)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 各 (B, H, D, HW)

        attn = torch.matmul(q.transpose(-2, -1), k)  # (B, H, HW, HW)
        attn = attn / math.sqrt(k.shape[-2])
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))  # (B, H, HW, D)

        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)  # (B, C, H, W)
        return x + self.out_proj(out)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        use_attention: bool,
        mode: str = "",
        output_scale_factor: float = 2.0,
    ):
        super().__init__()
        self.mode = mode
        self.output_scale_factor = output_scale_factor

        self.proj = Conv2D1x1(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.norm1 = AdaGroupNorm2D(in_channels, cond_channels)
        self.act1 = nn.SiLU()
        self.conv1 = Conv2D3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm2D(out_channels, cond_channels)
        self.act2 = nn.SiLU()
        self.conv2 = Conv2D3x3(out_channels, out_channels)
        self.attn = SelfAttention2D(out_channels) if use_attention else nn.Identity()

        if mode == "down":
            self.down = nn.MaxPool2d(2)
        elif mode == "up":
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.mode == "down":
            x = self.down(x)
        elif self.mode == "up":
            x = self.up(x)
        residual = self.proj(x)
        x = self.norm1(x, cond)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x, cond)
        x = self.act2(x)
        x = self.conv2(x)
        x = (x + residual) / self.output_scale_factor
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        channels_list: list[int],
        res_block_num_list: list[int],
        use_attention_list: list[bool],
    ):
        super().__init__()

        block_num = len(channels_list)

        # --- dowmsamples
        downs = []
        for i in range(block_num):
            ch = channels_list[i]
            downs.append(
                ResBlock(
                    in_channels=in_channels,
                    out_channels=ch,
                    cond_channels=cond_channels,
                    use_attention=use_attention_list[i],
                    mode="" if i == 0 else "down",
                )
            )
            for j in range(res_block_num_list[i] - 1):
                downs.append(
                    ResBlock(
                        in_channels=ch,
                        out_channels=ch,
                        cond_channels=cond_channels,
                        use_attention=use_attention_list[i],
                        mode="",
                    )
                )
        self.down_layers = nn.ModuleList(downs)

        # --- middle
        self.middle_layers = nn.ModuleList(
            [
                ResBlock(
                    in_channels=channels_list[-1],
                    out_channels=channels_list[-1],
                    cond_channels=cond_channels,
                    use_attention=True,
                    mode="",
                )
                for _ in range(2)
            ]
        )

        # --- upsamples
        ups = []
        for i in reversed(range(block_num)):
            ch = channels_list[i]
            for j in range(res_block_num_list[i] - 1):
                ups.append(
                    ResBlock(
                        in_channels=ch + ch,
                        out_channels=ch,
                        cond_channels=cond_channels,
                        use_attention=use_attention_list[i],
                        mode="",
                    )
                )
            ups.append(
                ResBlock(
                    in_channels=ch + ch,
                    out_channels=ch,
                    cond_channels=cond_channels,
                    use_attention=use_attention_list[i],
                    mode="" if i == 0 else "up",
                )
            )
        self.up_layers = nn.ModuleList(ups)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        down_outpus = []
        for down in self.down_layers:
            x = down(x, cond)
            down_outpus.append(x)
        for m in self.middle_layers:
            x = m(x, cond)
        for up, skip in zip(self.up_layers, reversed(down_outpus)):
            x = torch.cat([x, skip], dim=1)
            x = up(x, cond)
        return x


class Denoiser(nn.Module):
    def __init__(self, img_shape, action_num: int, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.condition_channels % 2 == 0
        c_ch = cfg.condition_channels

        # --- condition
        self.noise_emb = FourierFeatures(c_ch)
        self.act_emb = nn.Embedding(action_num, c_ch)
        self.cond_block = nn.Sequential(
            nn.LazyLinear(c_ch),
            nn.SiLU(),
            nn.Linear(c_ch, c_ch),
        )

        # --- denoiser
        self.conv_in = Conv2D3x3(img_shape[-1] * 2, cfg.channels_list[0])
        self.unet = UNet(cfg.channels_list[0], c_ch, cfg.channels_list, cfg.res_block_num_list, cfg.use_attention_list)
        self.out_norm = GroupNorm(cfg.channels_list[-1])
        self.out_act = nn.SiLU()
        self.out_conv = Conv2D3x3(cfg.channels_list[-1], img_shape[-1])
        nn.init.zeros_(self.out_conv.weight)

        self.loss_fn = nn.HuberLoss(reduction="none")
        self.optimizer = optim.AdamW(self.parameters(), lr=cfg.lr)

    def forward(self, noisy_obs: torch.Tensor, c_noise: torch.Tensor, prev_img: torch.Tensor, action_index: torch.Tensor):
        # --- condition
        noise_emb = self.noise_emb(c_noise)
        act_emb = self.act_emb(action_index)
        cond = torch.cat([noise_emb, act_emb], dim=-1)
        cond = self.cond_block(cond)

        x = torch.cat([noisy_obs, prev_img], dim=1)
        x = self.conv_in(x)
        x = self.unet(x, cond)
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x

    def denoise(self, noisy_img, sigma, prev_img, action_index):
        # (batch, h, w, ch) -> (batch, ch, h, w)
        noisy_img = noisy_img.permute((0, 3, 1, 2))
        prev_img = prev_img.permute((0, 3, 1, 2))

        # sigma = tf.sqrt(sigma**2 + self.cfg.sigma_offset_noise**2)

        # 正規化
        c_in = 1 / torch.sqrt(sigma**2 + self.cfg.sigma_data**2)
        scaled_noisy_img = c_in * noisy_img
        scaled_prev_img = prev_img / self.cfg.sigma_data
        c_noise = (torch.log(sigma) / 4).squeeze([1, 2, 3])

        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * torch.sqrt(c_skip)
        network_output = self(scaled_noisy_img, c_noise, scaled_prev_img, action_index)
        denoised_img = c_skip * noisy_img + c_out * network_output
        return denoised_img.permute((0, 2, 3, 1))


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: SamplerConfig, torch_dtype, device):
        self.denoiser = denoiser
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.device = device
        # 生成回数は固定なのでスケジュールを事前に作成
        self.sigmas = self.create_timesteps()

    def create_timesteps(self) -> List[float]:
        if self.cfg.num_steps_denoising == 1:
            return [self.cfg.sigma_max, 0]
        min_inv_rho = self.cfg.sigma_min ** (1 / self.cfg.rho)
        max_inv_rho = self.cfg.sigma_max ** (1 / self.cfg.rho)
        N = self.cfg.num_steps_denoising
        t = [
            (max_inv_rho + i / (N - 1) * (min_inv_rho - max_inv_rho)) ** self.cfg.rho
            for i in range(N)  #
        ]
        t += [0]
        return t

    def sample(self, prev_obs: torch.Tensor, action_index: torch.Tensor):
        b, h, w, ch = prev_obs.shape

        gamma_ = min(self.cfg.s_churn / self.cfg.num_steps_denoising, 2**0.5 - 1)

        x = torch.randn((b, h, w, ch), dtype=self.torch_dtype, device=self.device) * self.sigmas[0]
        trajectory: list[torch.Tensor] = [x]
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_min <= sigma <= self.cfg.s_max else 0
            sigma_hat = sigma * (gamma + 1)

            eps = torch.randn_like(x) * self.cfg.s_noise
            x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5

            torch_sigma_hat = torch.full((b, 1, 1, 1), sigma_hat, dtype=self.torch_dtype, device=self.device)
            denoised_img = self.denoiser.denoise(x, torch_sigma_hat, prev_obs, action_index)

            d = (x - denoised_img) / sigma_hat
            dt = next_sigma - sigma_hat

            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                torch_next_sigma = torch.full((b, 1, 1, 1), next_sigma, dtype=self.torch_dtype, device=self.device)
                denoised_img_2 = self.denoiser.denoise(x_2, torch_next_sigma, prev_obs, action_index)
                d_dash = (x_2 - denoised_img_2) / next_sigma
                x = x + dt * (d + d_dash) / 2

            trajectory.append(x)

        return x, trajectory

    def render_rgb_array(self, screen, worker) -> Optional[np.ndarray]:
        from srl.base.rl.worker_run import WorkerRun
        from srl.utils.pygame_wrapper import PygameScreen

        screen = cast(PygameScreen, screen)
        worker = cast(WorkerRun, worker)
        obs_space = worker.config.obs_render_img_space
        action_num = worker.config.action_space.n
        dtype = worker.config.get_dtype("np")

        # --- draw
        IMG_W = 128
        IMG_H = 128
        STR_H = 20
        PADDING = 4
        ACT_NUM = 5
        DIFF_NUM = 5

        img = torch.from_numpy(worker.render_image_state[np.newaxis, ...].astype(dtype)).to(self.device)

        # 横にアクション後の結果を表示
        for a in range(min(ACT_NUM, action_num)):
            x = (IMG_W + PADDING) * a
            y = STR_H + PADDING

            screen.draw_text(x, y, f"{a}", color=(255, 255, 255))
            y += STR_H + PADDING

            # --- diffision draw
            act = torch.from_numpy(np.array([a]).astype(np.int64)).to(self.device)
            n_obs, trajectory = self.sample(img, act)
            n_img = obs_space.to_image(n_obs.detach().cpu().numpy()[0])
            screen.draw_image_rgb_array(x, y, n_img, (IMG_W, IMG_W))

            y += IMG_H + PADDING
            trajectory = list(reversed(trajectory))
            for i in range(min(len(trajectory), DIFF_NUM)):
                t_obs = trajectory[i]
                t_img = obs_space.to_image(t_obs.detach().cpu().numpy()[0])
                screen.draw_image_rgb_array(x, y, t_img, (IMG_W, IMG_W))
                y += IMG_H + PADDING
