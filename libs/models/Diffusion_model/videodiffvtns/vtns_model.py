import warnings
import torch.nn.functional as F
import torch.nn as nn
import einops
import math
import torch
import os
import pywt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_linear_log_snr(t):
    return -torch.log(torch.expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class LearnedSinusoidalPosEmb(nn.Module):
     def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))
    def forward(self, x):
        x = einops.rearrange(x, 'b -> b 1')
        freqs = x * einops.rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class SegDiff(nn.Module):
    def __init__(self,
                 bit_scale=0.01,
                 timesteps=3,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion="ddim",
                 accumulation=True,
                 num_classes=1) -> None:
        super().__init__()

        self.encoder = SegformerForImageClassification.from_pretrained(seg_model_path)

        self.head_config = AutoConfig.from_pretrained(seg_model_path)
        self.head_config.num_labels = num_classes

        self.bit_scale = bit_scale
        self.diffusion_type = diffusion
        self.num_classes = num_classes
        self.randsteps = randsteps
        self.accumulation = accumulation
        self.sample_range = sample_range
        self.timesteps = timesteps
        self.decode_head = SegformerDecodeHeadwTime._from_config(self.head_config)
        self.time_difference = time_difference
        self.x_inp_dim = 64
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.x_inp_dim)

        self.transform = nn.Sequential(
            nn.BatchNorm2d(self.x_inp_dim * 4),
            nn.Conv2d(self.x_inp_dim * 4, self.x_inp_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.x_inp_dim * 8, self.x_inp_dim, kernel_size=3, stride=1, padding=1)
        )

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")
        time_dim = 1024  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )
        self.conv_frames = nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv_masks = nn.Conv2d(1, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.encoder_Memory = SegformerForImageClassification.from_pretrained(seg_model_path)
        self.KV_memoory = KeyValue()
        self.KV_query = KeyValue()
        self.Temporal_Memory = Temporal_Memorize()
        self.Temporal_DWT = Temporal_Memorize_DWT()
        self.conv_DWT = nn.Conv2d(96, 32, kernel_size=1)
        self.diffuser_temporal = Diffuser_SegformerDecodeHeadwTime._from_config(self.head_config)
        self.diffuser_spatial = Diffuser_SegformerDecodeHeadwTime._from_config(self.head_config)
        self.tempolrtKAN = KANBlock(dim=64, num_heads=1, mlp_ratio=1, qkv_bias=False, qk_scale=False,
                                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                                sr_ratio=8)
        self.tempo_norm = nn.LayerNorm(64)
        self.wavelrtKAN = KANBlock(dim=64, num_heads=1, mlp_ratio=1, qkv_bias=False, qk_scale=False,
                                  drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                                  sr_ratio=8)
        self.wave_norm = nn.LayerNorm(64)

    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():
            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def To_DWT(self, tensor):

        cA0, (cH0, cV0, cD0) = pywt.dwt2(tensor.cpu().detach(), 'haar')
        LL0 = torch.from_numpy(cA0).to(tensor.device)
        HH0 = torch.from_numpy(cH0 + cV0 + cD0).to(tensor.device)
        LL0 = F.interpolate(LL0, size=tensor.size()[2:], mode='bilinear', align_corners=False)
        HH0 = F.interpolate(HH0, size=tensor.size()[2:], mode='bilinear', align_corners=False)

        cA1, (cH1, cV1, cD1) = pywt.dwt2(tensor.cpu().detach(), 'db3')
        LL1 = torch.from_numpy(cA1).to(tensor.device)
        HH1 = torch.from_numpy(cH1 + cV1 + cD1).to(tensor.device)
        LL1 = F.interpolate(LL1, size=tensor.size()[2:], mode='bilinear', align_corners=False)
        HH1 = F.interpolate(HH1, size=tensor.size()[2:], mode='bilinear', align_corners=False)

        cA2, (cH2, cV2, cD2) = pywt.dwt2(tensor.cpu().detach(), 'dmey')
        LL2 = torch.from_numpy(cA2).to(tensor.device)
        HH2 = torch.from_numpy(cH2 + cV2 + cD2).to(tensor.device)

        LL = torch.cat((LL0, LL1, LL2), dim=1)
        HH = torch.cat((HH0, HH1, HH2), dim=1)

        return LL, HH

    def memorize_frames(self, imgs, masks):
        inp_features = self.conv_frames(imgs) + self.conv_masks(masks)
        temporal_feat = list(self.encoder_Memory(inp_features, output_hidden_states=True).hidden_states)[
            0]  # B, C, H, W
        memo_K, memo_V = self.KV_memoory(temporal_feat)  # B, C, H, W

        memo_K_LL, memo_K_HH = self.To_DWT(memo_K)
        memo_V_LL, memo_V_HH = self.To_DWT(memo_V)

        memo_K_LL, memo_K_HH = self.conv_DWT(memo_K_LL), self.conv_DWT(memo_K_HH)
        memo_V_LL, memo_V_HH = self.conv_DWT(memo_V_LL), self.conv_DWT(memo_V_HH)

        memo_K = memo_K.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)  # B, HW, C
        memo_V = memo_V.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)  # B, HW, C
        memo_K_LL = memo_K_LL.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)  # B, HW, C
        memo_K_HH = memo_K_HH.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)  # B, HW, C
        memo_V_LL = memo_V_LL.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)  # B, HW, C
        memo_V_HH = memo_V_HH.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)  # B, HW, C

        return temporal_feat, memo_K, memo_V, memo_K_LL, memo_V_LL, memo_K_HH, memo_V_HH

    def extract_feat(self, img):
        _, _, H, W = img.size()
        output_feats = self.encoder(img, output_hidden_states=True)
        return list(output_feats.hidden_states)

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = einops.repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_sample(self, img, temporal_feat_lst, keys_lst, vals_lst, keys_dict, vals_dict, opt, keys_dict_LL,
                    vals_dict_LL, keys_dict_HH, vals_dict_HH):
        output_hidden_states = self.extract_feat(img)
        x = output_hidden_states[0]
        b, c, h, w = x.size()
        device = x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = einops.repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)
            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)
            input_times = self.time_mlp(log_snr)
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next

            if self.accumulation:
                outs.append(mask_logit)

        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)

        logit = mask_logit.mean(dim=0, keepdim=True)

        return logit

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        if self.diffusion == "ddim":
            out = self.ddim_sample(x, img_metas)
        else:
            raise NotImplementedError
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward(self, img, gt_semantic_seg, temporal_feat_lst, keys_lst, vals_lst, keys_dict, vals_dict, opt,
                keys_dict_LL, vals_dict_LL,
                keys_dict_HH, vals_dict_HH):

        output_hidden_states = self.extract_feat(img)  # B, 256, h/4, w/4
        x = output_hidden_states[0]
        b, c, h, w = x.size()
        device = x.device
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes
        gt_down = self.embedding_table(gt_down).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale
        times = torch.zeros((b,), device=device).float().uniform_(self.sample_range[0], self.sample_range[1])  # [bs]
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise
        input_times = self.time_mlp(noise_level)
        return mask_logit

    def origin_memo(self, keys_lst, vals_lst):
        assert len(keys_lst) == len(vals_lst)
        if len(keys_lst) >= 3:
            KM_temporal, VM_temporal = torch.cat(keys_lst[-3:], dim=1), torch.cat(vals_lst[-3:], dim=1)  # B, HW, C
        elif len(keys_lst) == 2:
            KM_temporal, VM_temporal = torch.cat(keys_lst[-2:], dim=1), torch.cat(vals_lst[-2:], dim=1)  # B, HW, C
        elif len(keys_lst) == 1:
            KM_temporal, VM_temporal = keys_lst[-1], vals_lst[-1]  # B, HW, C
        return KM_temporal, VM_temporal
