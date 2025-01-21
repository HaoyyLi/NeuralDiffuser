import inspect
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput

# processes and stores attention probabilities
class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class GuidedStableDiffusion(DiffusionPipeline):
    """ guided stable diffusion based on the amazing repo by @crowsonkb and @Jack000
    - https://github.com/Jack000/glid-3-xl
    - https://github.dev/crowsonkb/k-diffusion
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        # guided_model,
        unet: UNet2DConditionModel,
        scheduler: Union[PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler],
        # feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            # guided_model=guided_model,
            unet=unet,
            scheduler=scheduler,
            # feature_extractor=feature_extractor,
        )

        # self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        # self.cut_out_size = (
        #     feature_extractor.size
        #     if isinstance(feature_extractor.size, int)
        #     else feature_extractor.size["shortest_edge"]
        # )
        # self.make_cutouts = MakeCutouts(self.cut_out_size)

        # set_requires_grad(self.guided_model, False)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            h = self.unet.config.attention_head_dim
            if isinstance(h, list):
                h = h[-1]
            slice_size = h // 2
            
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        self.enable_attention_slicing(None)

    def freeze_vae(self):
        set_requires_grad(self.vae, False)

    def unfreeze_vae(self):
        set_requires_grad(self.vae, True)

    def freeze_unet(self):
        set_requires_grad(self.unet, False)

    def unfreeze_unet(self):
        set_requires_grad(self.unet, True)

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        condition,
        noise_pred_original,
        guided_condition,
        classifier_guidance_scale,
        num_cutouts,
        use_cutouts=True,
        cal_loss=None
    ):
        latents = latents.detach().requires_grad_()

        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=condition).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            # pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            pred_original_sample = (latents - beta_prod_t * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
            # sample = pred_original_sample
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        sample = 1 / self.vae.config.scaling_factor * sample
        image = self.vae.decode(sample).sample
        # image = (image / 2 + 0.5).clamp(0, 1)

        # if use_cutouts:
        #     image = self.make_cutouts(image, num_cutouts)
        # else:
        #     image = transforms.Resize(self.cut_out_size)(image)
        # image = self.normalize(image).to(latents.dtype)
        
        loss = cal_loss(
                        image,
                        guided_condition,
                        # num_cutouts,
                        # use_cutouts=True,
                    ) * classifier_guidance_scale
        
        grads = torch.autograd.grad(loss, latents)[0] 
        
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma**2)
            noise_pred = noise_pred_original
        else:
            noise_pred = noise_pred_original - torch.sqrt(1 - alpha_prod_t) * grads
        return noise_pred, latents.detach()
    

    @torch.enable_grad()
    def cond_fn2(
        self,
        latents,
        timestep,
        index,
        condition,
        noise_pred_original,
        guided_condition,
        classifier_guidance_scale,
        num_cutouts,
        use_cutouts=True,
        cal_loss=None
    ):

        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=condition).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            # sample = pred_original_sample * (fac) + latents * (1 - fac)
            sample = pred_original_sample
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        
        sample = sample.detach().requires_grad_()

        # image = (image / 2 + 0.5).clamp(0, 1)

        # if use_cutouts:
        #     image = self.make_cutouts(image, num_cutouts)
        # else:
        #     image = transforms.Resize(self.cut_out_size)(image)
        # image = self.normalize(image).to(latents.dtype)

        max_iters = 1
        
        lr = 1e-2
        tv_loss = None
        loss_cutoff = 0.00001
        optimizer = torch.optim.Adam([sample], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)

        loss = None
        _ = None
        weights = torch.ones_like(sample).cuda()
        ones = torch.ones_like(sample).cuda()
        zeros = torch.zeros_like(sample).cuda()

        for _ in range(max_iters):
            sample = 1 / self.vae.config.scaling_factor * sample
            image = self.vae.decode(sample).sample
            loss = -1*cal_loss(
                            image,
                            guided_condition,
                            # num_cutouts,
                            # use_cutouts=True,
                        ) * classifier_guidance_scale
            # for __ in range(loss.shape[0]):
            #     if loss[__] < loss_cutoff:
            #         weights[__] = zeros[__]
            #     else:
            #         weights[__] = ones[__]
            before_x = torch.clone(sample.data)
            m_loss = loss.sum()
            if tv_loss != None:
                diff1 = sample[:, :, :, :-1] - sample[:, :, :, 1:]
                diff2 = sample[:, :, :-1, :] - sample[:, :, 1:, :]
                diff3 = sample[:, :, 1:, :-1] - sample[:, :, :-1, 1:]
                diff4 = sample[:, :, :-1, :-1] - sample[:, :, 1:, 1:]
                loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
                m_loss += tv_loss * loss_var

            m_loss.backward()
            optimizer.step()

            if scheduler != None:
                scheduler.step()

            with torch.no_grad():
                sample.data = before_x * (1 - weights) + weights * sample.data

            if weights.sum() == 0:
                break
        
        sample.requires_grad = False
        torch.set_grad_enabled(False)

        x_prev = sample
        noise_pred = (latents - self.scheduler.alphas_cumprod[timestep].sqrt()*x_prev) / (1-self.scheduler.alphas_cumprod[timestep]).sqrt()
        return noise_pred, latents.detach()

    @torch.no_grad()
    def __call__(
        self,
        condition: Optional[torch.Tensor],
        uncond_embeddings: Optional[torch.Tensor],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        sag_scale: Optional[float] = 0.75,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        cal_loss = None,
        cal_loss_2 = None,
        classifier_guidance_scale: Optional[float] = 0.,
        guided_condition: Optional[torch.FloatTensor] = None,
        classifier_guidance_scale_2: Optional[float] = 0.,
        guided_condition_2: Optional[torch.FloatTensor] = None,
        num_cutouts: Optional[int] = 4,
        use_cutouts: Optional[bool] = True,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        t_start = None,
        num_cfg_steps=None,
        num_k=1,
    ):
        
        if num_cfg_steps is None:
            num_cfg_steps = num_inference_steps

        if isinstance(condition, torch.Tensor):
            batch_size = 1
        elif isinstance(condition, list):
            batch_size = len(condition)
        else:
            raise ValueError(f"`prompt` has to be of type `Tensor` or `list` but is {type(condition)}")
        
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # duplicate text embeddings for each generation per prompt
        condition = condition.repeat_interleave(num_images_per_prompt, dim=0)

        # guided_condition
        # if classifier_guidance_scale > 0:
        #     guided_condition = guided_condition / guided_condition.norm(p=2, dim=-1, keepdim=True)
        #     # duplicate text embeddings clip for each generation per prompt
        #     guided_condition = guided_condition.repeat_interleave(num_images_per_prompt, dim=0)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        do_self_attention_guidance = sag_scale > 0.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            condition = torch.cat([uncond_embeddings, condition])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (batch_size * num_images_per_prompt, self.unet.config.in_channels, height // 8, width // 8)
        latents_dtype = condition.dtype
        if latents is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        if t_start is not None:
            timesteps_tensor = timesteps_tensor[-t_start:]


        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        store_processor = CrossAttnStoreProcessor()
        self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor

        map_size = None
        def get_map_size(module, input, output):
            nonlocal map_size
            map_size = output[0].shape[-2:]

        latents_list = []
        with self.unet.mid_block.attentions[0].register_forward_hook(get_map_size):
            for i, t in enumerate(tqdm(timesteps_tensor)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=condition).sample

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # perform self-attention guidance with the stored self-attentnion map
                if do_self_attention_guidance:
                    # classifier-free guidance produces two chunks of attention map
                    # and we only use unconditional one according to equation (25)
                    # in https://arxiv.org/pdf/2210.00939.pdf
                    if do_classifier_free_guidance:
                        # DDIM-like prediction of x0
                        pred_x0 = self.pred_x0(latents, noise_pred_uncond, t)
                        # get the stored attention maps
                        uncond_attn, cond_attn = store_processor.attention_probs.chunk(2)
                        # self-attention-based degrading of latents
                        degraded_latents = self.sag_masking(
                            pred_x0, uncond_attn, map_size, t, self.pred_epsilon(latents, noise_pred_uncond, t)
                        )
                        uncond_emb, _ = condition.chunk(2)
                        # forward and give guidance
                        degraded_pred = self.unet(degraded_latents, t, encoder_hidden_states=uncond_emb).sample
                        noise_pred += sag_scale * (noise_pred_uncond - degraded_pred)
                    else:
                        # DDIM-like prediction of x0
                        pred_x0 = self.pred_x0(latents, noise_pred, t)
                        # get the stored attention maps
                        cond_attn = store_processor.attention_probs
                        # self-attention-based degrading of latents
                        degraded_latents = self.sag_masking(
                            pred_x0, cond_attn, map_size, t, self.pred_epsilon(latents, noise_pred, t)
                        )
                        # forward and give guidance
                        degraded_pred = self.unet(degraded_latents, t, encoder_hidden_states=condition).sample
                        noise_pred += sag_scale * (noise_pred - degraded_pred)

                # perform classifier guidance
                if classifier_guidance_scale > 0 and num_cfg_steps > 0:
                    num_cfg_steps -= 1
                    condition_for_guidance = (
                        condition.chunk(2)[1] if do_classifier_free_guidance else condition
                    )
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        condition_for_guidance,
                        noise_pred,
                        guided_condition,
                        classifier_guidance_scale,
                        num_cutouts,
                        use_cutouts,
                        cal_loss
                    )
                    # noise_pred, latents = self.cond_fn2(
                    #         latents,
                    #         t,
                    #         i,
                    #         condition_for_guidance,
                    #         noise_pred,
                    #         guided_condition,
                    #         classifier_guidance_scale,
                    #         num_cutouts,
                    #         use_cutouts,
                    #         cal_loss
                    #     )
                else:
                    condition_for_guidance = (
                        condition.chunk(2)[1] if do_classifier_free_guidance else condition
                    )
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        condition_for_guidance,
                        noise_pred,
                        guided_condition_2,
                        classifier_guidance_scale_2,
                        num_cutouts,
                        use_cutouts,
                        cal_loss_2
                    )

                    
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents_list.append(latents)
                
        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image_list = [self.vae.decode(1 / self.vae.config.scaling_factor * l).sample for l in latents_list]

        image_list = [(im / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() for im in image_list]

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, image_list)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=image_list)

    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        # Noise it again to match the noise level
        degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t)

        return degraded_latents

    # Modified from diffusers.schedulers.scheduling_ddim.DDIMScheduler.step
    # Note: there are some schedulers that clip or do not return x_0 (PNDMScheduler, DDIMScheduler, etc.)
    def pred_x0(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_original_sample

    def pred_epsilon(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_eps = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_eps = (sample - (alpha_prod_t**0.5) * model_output) / (beta_prod_t**0.5)
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_eps = (beta_prod_t**0.5) * sample + (alpha_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_eps


# Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img