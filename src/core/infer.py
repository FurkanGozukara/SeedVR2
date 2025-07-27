# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import time
from typing import List, Optional, Tuple, Union
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from src.optimization.memory_manager import clear_vram_cache, log_vram_usage

from src.common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from src.common.distributed import (
    get_device,
)

# from common.fs import download

from src.models.dit_v2 import na


def optimized_channels_to_last(tensor):
    """🚀 Optimized replacement for rearrange(tensor, 'b c ... -> b ... c')
    Moves channels from position 1 to last position using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, channels, spatial]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, channels, height, width]
        return tensor.permute(0, 2, 3, 1)
    elif tensor.ndim == 5:  # [batch, channels, depth, height, width]
        return tensor.permute(0, 2, 3, 4, 1)
    else:
        # Fallback for other dimensions - move channel (dim=1) to last
        dims = list(range(tensor.ndim))
        dims = [dims[0]] + dims[2:] + [dims[1]]  # [0, 2, 3, ..., 1]
        return tensor.permute(*dims)

def optimized_channels_to_second(tensor):
    """🚀 Optimized replacement for rearrange(tensor, 'b ... c -> b c ...')
    Moves channels from last position to position 1 using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, spatial, channels]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, height, width, channels]
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 5:  # [batch, depth, height, width, channels]
        return tensor.permute(0, 4, 1, 2, 3)
    else:
        # Fallback for other dimensions - move last dim to position 1
        dims = list(range(tensor.ndim))
        dims = [dims[0], dims[-1]] + dims[1:-1]  # [0, -1, 1, 2, ..., -2]
        return tensor.permute(*dims)

class VideoDiffusionInfer():
    def __init__(self, config: DictConfig, debug: bool = False):
        self.config = config
        self.debug = debug
    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError
    
    def configure_diffusion(self):
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
            device=get_device(),
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config(
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=get_device(),
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor], preserve_vram: bool = False) -> List[Tensor]:
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # Vae process by each group.
            for sample in batches:
                sample = sample.to(device, dtype)
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)
                if use_sample:
                    latent = self.vae.encode(sample).latent
                    #latent = self.vae.encode(sample, preserve_vram).latent
                else:
                    # Deterministic vae encode, only used for i2v inference (optionally)
                    latent = self.vae.encode(sample).posterior.mode().squeeze(2)
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                #latent = optimized_channels_to_last(latent)
                latent = (latent - shift) * scale
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]

        return latents
    

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor], target_dtype: torch.dtype = None, preserve_vram: bool = False) -> List[Tensor]:
        """🚀 VAE decode optimisé - décodage direct sans chunking, compatible avec autocast externe"""
        print(f"🔍 vae_decode called with preserve_vram={preserve_vram}")
        samples = []
        if len(latents) > 0:
            #t = time.time()
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)


            # 🚀 OPTIMISATION 1: Group latents intelligemment pour batch processing
            print(f"🔍 VAE Decode Start - Number of latents: {len(latents)}")
            if len(latents) > 0:
                print(f"🔍 First latent shape before grouping: {latents[0].shape}")
            
            if self.config.vae.grouping:
                latents, indices = na.pack(latents)
                print(f"🔍 After grouping: {len(latents)} groups")
            else:
                latents = [latent.unsqueeze(0) for latent in latents]
                print(f"🔍 No grouping: {len(latents)} individual latents")

            if self.debug or True:  # Always show this for debugging
                print(f"🔄 shape of latents after grouping: {latents[0].shape if latents else 'empty'}")
            #print(f"🔄 GROUPING time: {time.time() - t} seconds")
            t = time.time()
            # 🚀 OPTIMISATION 2: Traitement batch optimisé avec dtype adaptatif
            for i, latent in enumerate(latents):
                # Préparation optimisée du latent
                # Utiliser target_dtype si fourni (évite double autocast)
                effective_dtype = target_dtype if target_dtype is not None else dtype
                latent = latent.to(device, effective_dtype, non_blocking=True)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                #latent = optimized_channels_to_second(latent)
                # Check for multi-frame decoding to reduce VRAM
                print(f"🔍 VAE Decode - Latent shape: {latent.shape}, ndim: {latent.ndim}")
                temporal_frames = latent.shape[2] if latent.ndim >= 5 else 1
                print(f"🔍 Temporal frames detected: {temporal_frames}, preserve_vram: {preserve_vram}")
                
                # 🚀 OPTIMISATION 3: Frame-by-frame VAE decoding DISABLED
                # The SeedVR2 VAE requires temporal context - decoding frames individually breaks video coherence
                use_frame_by_frame = False  # Disabled due to temporal artifacts
                print(f"🔍 Frame-by-frame decision: DISABLED (temporal VAE requires context)")
                
                if use_frame_by_frame and preserve_vram and temporal_frames > 1:
                    print(f"✅ Using frame-by-frame VAE decode for {temporal_frames} frames")
                    if self.debug:
                        print(f"🔄 Frame-by-frame VAE decode: {temporal_frames} frames")
                    frame_samples = []
                    for frame_idx in range(temporal_frames):
                        # Extract single frame from temporal dimension
                        frame_latent = latent[:, :, frame_idx:frame_idx+1, :, :]
                        # Remove the temporal dimension for VAE decode
                        frame_latent = frame_latent.squeeze(2)
                        # Decode single frame
                        frame_sample = self.vae.decode(frame_latent, preserve_vram).sample
                        frame_samples.append(frame_sample)
                        # Clean up to prevent VRAM accumulation
                        if frame_idx < temporal_frames - 1:
                            torch.cuda.empty_cache()
                    # Concatenate all frames - use same dimension as input
                    sample = torch.stack(frame_samples, dim=2)
                    del frame_samples
                else:
                    # Standard batch decode when not preserving VRAM
                    latent = latent.squeeze(2)
                    sample = self.vae.decode(latent, preserve_vram).sample
                
                # 🚀 OPTIMISATION 4: Post-processing conditionnel
                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)
                    
                samples.append(sample)
                
                # 🚀 OPTIMISATION 5: Cleanup after each batch when preserve_vram is active
                # No need for aggressive cleanup with frame-by-frame decoding
            
            if self.debug:
                print(f"🔄 DECODE time: {time.time() - t} seconds")
            #t = time.time()
            # Ungroup back to individual sample with the original order.
            if self.config.vae.grouping:
                samples = na.unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]
            
            
            #print(f"🔄 UNGROUPING time: {time.time() - t} seconds")
            #t = time.time()
        return samples

    def timestep_transform(self, timesteps: Tensor, latents_shapes: Tensor):
        # Skip if not needed.
        if not self.config.diffusion.timesteps.get("transform", False):
            return timesteps

        # Compute resolution.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        # Compute shift factor.
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.schedule.T
        return timesteps

    def get_vram_usage(self):
        """Obtenir l'utilisation VRAM actuelle (allouée et réservée)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            return allocated, reserved, max_allocated
        return 0, 0, 0

    def get_vram_peak(self):
        """Obtenir le pic VRAM depuis le dernier reset"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0

    def reset_vram_peak(self):
        """Reset le compteur de pic VRAM"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        texts_neg: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        cfg_scale: Optional[float] = None,
        preserve_vram: bool = False,
        temporal_overlap: int = 0,
        use_blockswap: bool = False,
        dit_preserve_vram: bool = None,  # Separate flag for DiT offloading
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        # Monitoring VRAM initial et reset des pics
        #self.reset_vram_peak()
        
        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale

        # 🚀 OPTIMISATION: Détecter le dtype du modèle pour performance optimale
        model_dtype = next(self.dit.parameters()).dtype
        if self.debug:
            print(f"🎯 model_dtype: {model_dtype}")
        # Adapter les dtypes selon le modèle
        if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # FP8 natif: utiliser BFloat16 pour les calculs intermédiaires (compatible)
            target_dtype = torch.float16
            #print(f"🚀 FP8 model detected: using BFloat16 for intermediate calculations")
        elif model_dtype == torch.float16:
            target_dtype = torch.bfloat16
            #print(f"🎯 FP16 model: using FP16 pipeline")
        else:
            target_dtype = torch.bfloat16
            #print(f"🎯 BFloat16 model: using BFloat16 pipeline")
        if self.debug:
            print(f"🎯 target_dtype: {target_dtype}")
        # Text embeddings.
        assert type(texts_pos[0]) is type(texts_neg[0])
        if isinstance(texts_pos[0], str):
            text_pos_embeds, text_pos_shapes = self.text_encode(texts_pos)
            text_neg_embeds, text_neg_shapes = self.text_encode(texts_neg)
        elif isinstance(texts_pos[0], tuple):
            text_pos_embeds, text_pos_shapes = [], []
            text_neg_embeds, text_neg_shapes = [], []
            for pos in zip(*texts_pos):
                emb, shape = na.flatten(pos)
                text_pos_embeds.append(emb)
                text_pos_shapes.append(shape)
            for neg in zip(*texts_neg):
                emb, shape = na.flatten(neg)
                text_neg_embeds.append(emb)
                text_neg_shapes.append(shape)
        else:
            text_pos_embeds, text_pos_shapes = na.flatten(texts_pos)
            text_neg_embeds, text_neg_shapes = na.flatten(texts_neg)

        # Adapter les embeddings texte au dtype cible (compatible avec FP8)
        if isinstance(text_pos_embeds, torch.Tensor):
            text_pos_embeds = text_pos_embeds.to(target_dtype)
        if isinstance(text_neg_embeds, torch.Tensor):
            text_neg_embeds = text_neg_embeds.to(target_dtype)

        # Flatten.
        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)

        # Adapter les latents au dtype cible (compatible avec FP8)
        latents = latents.to(target_dtype) if latents.dtype != target_dtype else latents
        latents_cond = latents_cond.to(target_dtype) if latents_cond.dtype != target_dtype else latents_cond

        # Use dit_preserve_vram for DiT operations if provided, otherwise use preserve_vram
        dit_vram_flag = dit_preserve_vram if dit_preserve_vram is not None else preserve_vram
        
        if dit_vram_flag:
            # Log VRAM before any operations
            log_vram_usage("Start of Inference", "Before moving models")
            
            if conditions[0].shape[0] > 1:
                t = time.time()
                self.vae = self.vae.to("cpu")
                if self.debug:
                    print(f"🔄 VAE to CPU time: {time.time() - t} seconds")
                    
            # Log after VAE moved to CPU
            log_vram_usage("After VAE to CPU", "VAE offloaded, DiT ready for inference")
            # Before sampling, check if BlockSwap is active
            if not use_blockswap and not hasattr(self, "_blockswap_active"):
                t = time.time()
                self.dit = self.dit.to(get_device())
                if self.debug:
                    print(f"🔄 Dit to GPU time: {time.time() - t} seconds")
            else:
                # BlockSwap manages device placement
                pass

        # Log VRAM before inference
        log_vram_usage("Before DiT Inference", f"Ready to run diffusion, BlockSwap: {use_blockswap}")
        
        t = time.time()
        
        with torch.autocast("cuda", target_dtype, enabled=True):
            latents = self.sampler.sample(
                x=latents,
                f=lambda args: classifier_free_guidance_dispatcher(
                    pos=lambda: self.dit(
                        vid=torch.cat([args.x_t, latents_cond], dim=-1),
                        txt=text_pos_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_pos_shapes,
                        timestep=args.t.repeat(batch_size),
                    ).vid_sample,
                    neg=lambda: self.dit(
                        vid=torch.cat([args.x_t, latents_cond], dim=-1),
                        txt=text_neg_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_neg_shapes,
                        timestep=args.t.repeat(batch_size),
                    ).vid_sample,
                    scale=(
                        cfg_scale
                        if (args.i + 1) / len(self.sampler.timesteps)
                        <= self.config.diffusion.cfg.get("partial", 1)
                        else 1.0
                    ),
                    rescale=self.config.diffusion.cfg.rescale,
                ),
            )
        
        if self.debug:
            print(f"🔄 INFERENCE time: {time.time() - t} seconds")
            
        # Log VRAM after inference
        log_vram_usage("After DiT Inference", "Diffusion complete, before VAE decode")

        latents = na.unflatten(latents, latents_shapes)
        #print(f"🔄 UNFLATTEN time: {time.time() - t} seconds")
        
        # 🎯 Pré-calcul des dtypes (une seule fois)
        vae_dtype = getattr(torch, self.config.vae.dtype)
        decode_dtype = torch.float16 if (vae_dtype == torch.float16 or target_dtype == torch.float16) else vae_dtype
        if self.debug:
            print(f"🎯 decode_dtype: {decode_dtype}")
        if dit_vram_flag:
            t = time.time()
            self.dit = self.dit.to("cpu")
            latents_cond = latents_cond.to("cpu")
            latents_shapes = latents_shapes.to("cpu")
            if latents[0].shape[0] > 1:
                clear_vram_cache()
            if self.debug:
                print(f"🔄 Dit to CPU time: {time.time() - t} seconds")
            
            # Extra cleanup when BlockSwap was active to defragment memory
            if hasattr(self, "_blockswap_active") and self._blockswap_active:
                torch.cuda.synchronize()  # Ensure all operations complete
                # Clear any cached tensors in DiT blocks
                if hasattr(self.dit, 'blocks'):
                    for block in self.dit.blocks:
                        if hasattr(block, '_cached_tensors'):
                            del block._cached_tensors
                torch.cuda.empty_cache()  # Force memory defragmentation
                if self.debug:
                    print(f"🔄 Extra BlockSwap cleanup for VAE")
            
            if latents[0].shape[0] > 1:
                # Log before VAE to GPU
                log_vram_usage("Before VAE to GPU", "DiT offloaded, ready to move VAE")
                
                t = time.time()
                self.vae = self.vae.to(get_device())
                
                if self.debug:
                    print(f"🔄 VAE to GPU time: {time.time() - t} seconds")
                    
                # Log after VAE to GPU
                log_vram_usage("After VAE to GPU", "VAE loaded, ready for decoding")




        # Log before VAE decode
        log_vram_usage("Before VAE Decode", f"Starting decode of {len(latents)} latents")
        
        #with torch.autocast("cuda", decode_dtype, enabled=True):
        samples = self.vae_decode(latents, target_dtype=decode_dtype, preserve_vram=preserve_vram)
        
        # Log after VAE decode
        log_vram_usage("After VAE Decode", f"Decoded {len(samples)} samples")
        
        if self.debug:
            print(f"🔄 Samples shape: {samples[0].shape}")
        #print(f"🔄  ULTRA-FAST VAE DECODE time: {time.time() - t} seconds")
        #t = time.time()
        #self.dit.to(get_device())
        #self.vae.to("cpu")
        #print(f"🔄 Dit to GPU time: {time.time() - t} seconds")
        #t = time.time()
        # 🚀 CORRECTION CRITIQUE: Conversion batch Float16 pour ComfyUI (plus rapide)
        if samples and len(samples) > 0 and samples[0].dtype != torch.float16:
            if self.debug:
                print(f"🔧 Converting {len(samples)} samples from {samples[0].dtype} to Float16")
            samples = [sample.to(torch.float16, non_blocking=True) for sample in samples]
        
        #print(f"🚀 Conversion batch Float16 time: {time.time() - t} seconds")
        
        # 🚀 OPTIMISATION: Nettoyage final minimal
        #t = time.time()
        #if dit_offload:
        #    self.vae.to("cpu")
        #    torch.cuda.empty_cache()
        #    self.dit.to(get_device())
        #else:
            # Garder VAE sur GPU pour les prochains appels
        #torch.cuda.empty_cache()
        #print(f"🔄 FINAL CLEANUP time: {time.time() - t} seconds")

        
        return samples
