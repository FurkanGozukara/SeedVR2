"""
Generation Logic Module for SeedVR2

This module handles the main generation pipeline including:
- Single generation steps with adaptive dtype handling
- Complete generation loop with temporal awareness
- Context-aware batch processing with overlapping
- Video preprocessing and post-processing
- Optimized memory management during generation

Key Features:
- Native FP8 pipeline support for 2x speedup and 50% VRAM reduction
- Context-aware generation with temporal overlap for smooth transitions
- Adaptive dtype detection and optimal autocast configuration
- Intelligent batch processing with memory optimization
- Advanced video format handling (4n+1 constraint)
"""

# Required for VAE lifecycle management

import os
import gc
import torch
import time
from torchvision.transforms import Compose, Lambda, Normalize


# Import required modules
from src.optimization.memory_manager import reset_vram_peak
from src.optimization.performance import (
    optimized_video_rearrange, optimized_single_video_rearrange, 
    optimized_sample_to_image_format, temporal_latent_blending
)
from src.common.seed import set_seed
try:
    # Use local comfy copy from STAR instead of system ComfyUI
    import sys
    import os
    star_logic_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'STAR', 'logic')
    if star_logic_path not in sys.path:
        sys.path.insert(0, star_logic_path)
    import comfy.model_management
    COMFYUI_AVAILABLE = True
except:
    COMFYUI_AVAILABLE = False
    pass
# Get script directory for embeddings
script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import create_object
from src.core.model_manager import load_quantized_state_dict
# Import transforms and color fix

from src.data.image.transforms.divisible_crop import DivisibleCrop
from src.data.image.transforms.na_resize import NaResize

from src.utils.color_fix import wavelet_reconstruction



def generation_step(runner, text_embeds_dict, preserve_vram, cond_latents, temporal_overlap, 
                   tiled_vae=False, tile_size=(64, 64), tile_stride=(32, 32)):
    """
    Execute a single generation step with adaptive dtype handling
    
    Args:
        runner: VideoDiffusionInfer instance
        text_embeds_dict (dict): Text embeddings for positive and negative prompts
        preserve_vram (bool): Whether to enable VRAM optimization
        cond_latents (list): Conditional latents for generation
        temporal_overlap (int): Number of frames for temporal overlap
        
    Returns:
        tuple: (samples, last_latents) for potential temporal continuation
        
    Features:
        - Adaptive dtype detection (FP8/FP16/BFloat16)
        - Optimal autocast configuration for each model type
        - Memory-efficient noise generation and reuse
        - Automatic device placement with dtype preservation
        - Advanced inference optimization
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Adaptive dtype detection for optimal performance
    model_dtype = next(runner.dit.parameters()).dtype
    
    # Configure dtypes according to model architecture
    if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 native: use BFloat16 for intermediate calculations (optimal compatibility)
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif model_dtype == torch.float16:
        dtype = torch.float16
        autocast_dtype = torch.float16
    else:
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16

    def _move_to_cuda(x):
        """Move tensors to CUDA with adaptive optimal dtype"""
        return [i.to(device, dtype=dtype) for i in x]

    # Memory optimization: Generate noise once and reuse to save VRAM
    with torch.cuda.device(device):
        base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
    
    # Move tensors with adaptive dtype (optimized for FP8/FP16/BFloat16)
    noises, aug_noises, cond_latents = _move_to_cuda(noises), _move_to_cuda(aug_noises), _move_to_cuda(cond_latents)
    
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        # Use adaptive optimal dtype
        t = (
            torch.tensor([1000.0], device=device, dtype=dtype)
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=device)[None]
        t = runner.timestep_transform(t, shape)
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    # Generate conditions with memory optimization
    condition = runner.get_condition(
        noises[0],
        task="sr",
        latent_blur=_add_noise(cond_latents[0], aug_noises[0]),
    )
    conditions = [condition]
    t = time.time()
    
    # Check if BlockSwap is active
    use_blockswap = hasattr(runner, "_blockswap_active") and runner._blockswap_active

    # Use adaptive autocast for optimal performance
    with torch.no_grad():
        with torch.autocast("cuda", autocast_dtype, enabled=True):
            # Keep original preserve_vram value for VAE operations
            dit_preserve_vram = preserve_vram and not use_blockswap  # Disable DiT offload if BlockSwap active
            
            # Debug log tiled VAE settings
            print(f"🔍 generation.py - process_chunk: tiled_vae={tiled_vae}, tile_size={tile_size}, tile_stride={tile_stride}")
            
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                preserve_vram=preserve_vram,  # Pass original value so VAE can use it
                temporal_overlap=temporal_overlap,
                use_blockswap=use_blockswap,
                dit_preserve_vram=dit_preserve_vram,  # Separate flag for DiT offloading
                tiled_vae=tiled_vae,
                tile_size=tile_size,
                tile_stride=tile_stride,
                **text_embeds_dict,
            )
    
    print(f"🔄 INFERENCE time: {time.time() - t} seconds")
    
    # Process samples with advanced optimization
    samples = optimized_video_rearrange(video_tensors)
    #last_latents = samples[-temporal_overlap:] if temporal_overlap > 0 else samples[-1:]
    noises = noises[0].to("cpu")
    aug_noises = aug_noises[0].to("cpu")
    cond_latents = cond_latents[0].to("cpu")
    conditions = conditions[0].to("cpu")
    condition = condition.to("cpu")
    
    return samples #, last_latents


def cut_videos(videos):
    """
    Correct video cutting respecting the constraint: frames % 4 == 1
    
    Args:
        videos (torch.Tensor): Video tensor to format
        
    Returns:
        torch.Tensor: Properly formatted video tensor
        
    Features:
        - Ensures frames % 4 == 1 constraint for model compatibility
        - Intelligent padding with last frame repetition
        - Memory-efficient tensor operations
    """
    t = videos.size(1)
    
    if t % 4 == 1:
        return videos
    
    # Calculate next valid number (4n + 1)
    padding_needed = (4 - (t % 4)) % 4 + 1
    
    # Apply padding to reach 4n+1 format
    last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
    result = torch.cat([videos, last_frame], dim=1)
    
    return result


def generation_loop(runner, images, cfg_scale=1.0, seed=666, res_w=720, batch_size=90, preserve_vram=False, temporal_overlap=0, debug=False, block_swap_config=None, progress_callback=None, frame_save_callback=None, tiled_vae=False, tile_size=(64, 64), tile_stride=(32, 32)):
    """
    Main generation loop with context-aware temporal processing
    
    Args:
        runner: VideoDiffusionInfer instance
        images (torch.Tensor): Input images for upscaling
        cfg_scale (float): Classifier-free guidance scale
        seed (int): Random seed for reproducibility
        res_w (int): Target resolution width
        batch_size (int): Batch size for processing
        preserve_vram (str/bool): VRAM preservation mode
        temporal_overlap (int): Frames for temporal continuity
        debug (bool): Debug mode
        block_swap_config (dict): Optional BlockSwap configuration
        progress_callback (callable): Optional callback for progress reporting
        frame_save_callback (callable): Optional callback for saving frames as batches complete
            Called with (batch_tensor, batch_num, start_idx, end_idx)
        
    Returns:
        torch.Tensor: Generated video frames
        
    Features:
        - Context-aware generation with temporal overlap
        - Adaptive dtype pipeline (FP8/FP16/BFloat16)
        - Memory-optimized batch processing
        - Advanced video transformation pipeline
        - Intelligent VRAM management throughout process
        - Real-time progress reporting
    """
    # Debug log tiled VAE configuration
    print(f"[INFO] 📊 Entering generation_loop - tiled_vae={tiled_vae}, tile_size={tile_size}, tile_stride={tile_stride}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log BlockSwap status
    if block_swap_config:
        blocks_to_swap = block_swap_config.get("blocks_to_swap", 0)
        if blocks_to_swap > 0:
            print(f"🔄 Generation starting with BlockSwap: {blocks_to_swap} blocks")

    # Adaptive model dtype detection for maximum performance
    model_dtype = None
    try:
        # Get real dtype of loaded DiT model
        model_dtype = next(runner.dit.parameters()).dtype
        
        # Adapt dtypes according to model
        if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # For FP8, use BFloat16 for intermediate calculations (compatible)
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16  # VAE stays BFloat16 for compatibility
        elif model_dtype == torch.float16:
            compute_dtype = torch.float16
            autocast_dtype = torch.float16
            vae_dtype = torch.float16
        else:  # BFloat16 or others
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16
            
    except Exception as e:
        print(f"⚠️ Could not detect model dtype: {e}, falling back to BFloat16")
        model_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
        vae_dtype = torch.bfloat16

    # Optimization tips for users
    if torch.cuda.is_available():
        total_frames = len(images)
        optimal_batches = [x for x in [i for i in range(1, 200) if i % 4 == 1] if x <= total_frames]
        if optimal_batches:
            best_batch = max(optimal_batches)
            if best_batch != batch_size:
                print(f"\n💡 TIP: For {total_frames} frames, use batch_size={best_batch} to avoid padding")
                if batch_size not in optimal_batches:
                    padding_waste = sum(((i // 4) + 1) * 4 + 1 - i for i in range(batch_size, total_frames, batch_size))
                    print(f"   Currently: ~{padding_waste} wasted padding frames")

    # Configure classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    # Configure sampling steps
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # Set random seed
    set_seed(seed)

    # Advanced video transformation pipeline
    video_transform = Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            # Upsample image, model only trained for high res
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (faster than Rearrange)
    ])

    # Initialize generation state
    batch_samples = []
    final_tensor = None
    
    # For ETA tracking
    batch_times = []
    start_time = time.time()
    
    # Load text embeddings with adaptive dtype
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=compute_dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=compute_dtype)
    text_embeds = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
    
    
    
    # Standard processing (non-TileVAE) continues below
    # Memory optimization
    reset_vram_peak()
    
    # Calculate processing parameters
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Calculate total batches for progress reporting
    total_batches = len(range(0, len(images), step))
    
    # Move images to CPU for memory efficiency
    #t = time.time()
    #images = images.to("cpu")
    #print(f"🔄 Images to CPU time: {time.time() - t} seconds")
    
    # Use existing debugger from runner if available
    debugger = None
    if hasattr(runner, '_blockswap_debugger') and runner._blockswap_debugger is not None:
        debugger = runner._blockswap_debugger
        debugger.clear_history()
    
    try:
        # Main processing loop with context awareness
        for batch_count, batch_idx in enumerate(range(0, len(images), step)):
            # VAE Lifecycle Management: Re-create if torn down in the previous iteration
            # This is a robust way to prevent VRAM leaks between batches.
            if preserve_vram and (not hasattr(runner, 'vae') or runner.vae is None):
                print("🔧 Re-creating VAE for new batch to ensure clean VRAM state...")
                # Re-create using the original config stored in the runner
                runner.vae = create_object(runner.config.vae.model)
                runner.vae.requires_grad_(False).eval()

                # Get checkpoint path from stored base directory
                base_cache_dir = getattr(runner, '_base_cache_dir', './models')
                checkpoint_path = os.path.join(base_cache_dir, f'./{runner.config.vae.checkpoint}')

                # Load state dict to CPU since preserve_vram is on
                state = load_quantized_state_dict(checkpoint_path, "cpu", keep_native_fp8=False)
                runner.vae.load_state_dict(state)
                del state

                # Re-apply other VAE settings
                if hasattr(runner.vae, "set_causal_slicing") and hasattr(runner.config.vae, "slicing"):
                    runner.vae.set_causal_slicing(**runner.config.vae.slicing)
                if hasattr(runner.vae, "set_memory_limit"):
                    runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
                print("✅ VAE re-created on CPU.")
            # Calculate batch indices with overlap
            if COMFYUI_AVAILABLE:
                comfy.model_management.throw_exception_if_processing_interrupted()
            if batch_idx == 0:
                # First batch: no overlap
                start_idx = 0
                end_idx = min(batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = True
            else:
                # Subsequent batches: temporal overlap
                start_idx = batch_idx 
                end_idx = min(start_idx + batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = False
                if effective_batch_size <= temporal_overlap:
                    break  # Not enough new frames, stop

            tps_loop = time.time()
            batch_number = (batch_idx // step + 1) if step > 0 else 1
            current_frames = end_idx - start_idx
            percent_complete = int((batch_number / total_batches) * 100)
            frames_remaining = len(images) - end_idx
            print(f"\n🎬 Batch {batch_number}/{total_batches} ({percent_complete}%): frames {start_idx}-{end_idx-1} | {frames_remaining} frames left")
            

            
            # Process current batch
            video = images[start_idx:end_idx]
            if debug:
                print(f"🔄 video Compute dtype: {compute_dtype}")
            # Use adaptive computation dtype 
            video = video.permute(0, 3, 1, 2).to(device, dtype=compute_dtype)
            
            # Apply video transformations with memory optimization
            transformed_video = video_transform(video)
            del video
            #video = video.to("cpu")
            #del video
            ori_lengths = [transformed_video.size(1)]
            
            # Handle correct format: frames % 4 == 1
            t = transformed_video.size(1)
            print(f"📹 Sequence of {t} frames")
            
            
            if len(images) >= 5 and t % 4 != 1:
                if debug:
                    print(f"🔄 Transformed video shape before cut: {transformed_video.shape}")
                transformed_video = cut_videos(transformed_video)
                if debug:
                    print(f"🔄 Transformed video shape: {transformed_video.shape}")
            
            # Context-aware temporal strategy
            # First batch: standard complete diffusion
            tps_vae = time.time()
            runner.vae.to(device)
            if debug:
                print(f"🔄 VAE to GPU time: {time.time() - tps_vae} seconds")
            tps_vae = time.time()
            if debug:
                print(f"🔄 VAE dtype: {autocast_dtype}")
            with torch.autocast("cuda", autocast_dtype, enabled=True):
                cond_latents = runner.vae_encode([transformed_video])
            if debug:
                print(f"🔄 VAE encode time: {time.time() - tps_vae} seconds")
            
            # Move transformed_video to CPU to reduce VRAM peak when preserve_vram is enabled
            if preserve_vram:
                tps = time.time()
                transformed_video = transformed_video.to("cpu")
                if debug:
                    print(f"🔄 Transformed video to cpu time: {time.time() - tps} seconds")
            if debug:
                print(f"🔄 Cond latents shape: {cond_latents[0].shape}, time: {time.time() - tps_vae} seconds")
            
            # Normal generation
            samples = generation_step(runner, text_embeds, preserve_vram, cond_latents=cond_latents, 
                                    temporal_overlap=temporal_overlap, tiled_vae=tiled_vae, 
                                    tile_size=tile_size, tile_stride=tile_stride)
            #del cond_latents
            del cond_latents
               
            
            # Post-process samples
            sample = samples[0]
            del samples 
            #del samples
            if ori_lengths[0] < sample.shape[0]:
                sample = sample[:ori_lengths[0]]
            #if temporal_overlap > 0 and not is_first_batch and sample.shape[0] > effective_batch_size - temporal_overlap:
            #    sample = sample[temporal_overlap:]  # Remove overlap frames from output
            
            # Apply color correction if available
            tps = time.time()
            # Only move to device if it was moved to CPU (when preserve_vram is enabled)
            if transformed_video.device.type == 'cpu':
                transformed_video = transformed_video.to(device)
                if debug:
                    print(f"🔄 Transformed video to device time: {time.time() - tps} seconds")
            
            input_video = [optimized_single_video_rearrange(transformed_video)]
            del transformed_video
            #transformed_video = transformed_video.to("cpu")
            #del transformed_video
            sample = wavelet_reconstruction(sample, input_video[0][:sample.size(0)])
            del input_video

            # Convert to final image format
            sample = optimized_sample_to_image_format(sample)
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
            sample_cpu = sample.to(torch.float16).to("cpu")
            del sample
            batch_samples.append(sample_cpu)
            #del sample 
            
            # Call frame save callback if provided
            if frame_save_callback:
                # Calculate actual frame indices for this batch
                batch_start_idx = start_idx
                batch_end_idx = batch_start_idx + sample_cpu.shape[0]
                frame_save_callback(sample_cpu, batch_count, batch_start_idx, batch_end_idx)
            
            # Aggressive cleanup after each batch
            tps = time.time()
                        # Progress callback - batch start
            if progress_callback:
                # Pass frame range information for better status display
                progress_callback(batch_count+1, total_batches, current_frames, f"frames {start_idx}-{end_idx-1}")
            #transformed_video = transformed_video.to("cpu")
            #print(f"🔄 Transformed video to cpu time: {time.time() - tps} seconds")
            
            # Calculate batch time and ETA
            batch_time = time.time() - tps_loop
            batch_times.append(batch_time)
            avg_batch_time = sum(batch_times) / len(batch_times)
            batches_remaining = total_batches - batch_number
            eta_seconds = avg_batch_time * batches_remaining
            
            # Format ETA
            if eta_seconds > 60:
                eta_minutes = int(eta_seconds / 60)
                eta_secs = int(eta_seconds % 60)
                eta_str = f"{eta_minutes}m {eta_secs}s"
            else:
                eta_str = f"{int(eta_seconds)}s"
            
            if debug:
                print(f"🔄 Time batch: {batch_time:.2f} seconds")
            print(f"⏱️  Batch time: {batch_time:.1f}s | Avg: {avg_batch_time:.1f}s | ETA: {eta_str}")
            
            # Report batch time for external tracking
            if hasattr(progress_callback, '__self__') and hasattr(progress_callback.__self__, 'track_batch_time'):
                progress_callback.__self__.track_batch_time(batch_time)
            # Clean VRAM after each batch when preserve_vram is active
            # VAE Teardown: Destroy VAE object to guarantee VRAM release
            if preserve_vram:
                print("💥 Tearing down VAE to release VRAM before next batch...")
                if hasattr(runner, 'vae') and runner.vae is not None:
                    # Deleting the object is the most reliable way to free memory
                    del runner.vae
                    runner.vae = None
                gc.collect()
                torch.cuda.empty_cache()
                print("✅ VAE teardown complete.")

            #del transformed_video
            #clear_vram_cache()
            # Log memory state at the end of each batch
            if debugger:
                debugger.log_memory_state(f"Batch {batch_number} - Memory", show_tensors=False)

    finally:
        if debugger:
            debugger.log("🧹 Generation loop cleanup")

        # Final cleanup of embeddings
        del text_pos_embeds
        del text_neg_embeds

        # Final VAE cleanup in case loop was interrupted
        if preserve_vram and hasattr(runner, 'vae') and runner.vae is not None:
            del runner.vae
            runner.vae = None
            gc.collect()
        
        # IMPORTANT: Don't move models to CPU - this causes RAM leak!
        # The session manager will handle cleanup based on preserve_vram setting
        # Only clear CUDA cache
        torch.cuda.empty_cache()
        
        # Note: Models (runner.dit and runner.vae) should stay where they are
        # The cleanup will be handled by the session manager

        # Log final memory state
        if debugger:
            debugger.log_memory_state("Generation loop - After cleanup", show_tensors=False)
    
    # OPTIMISATION ULTIME : Pré-allocation et copie directe (évite les torch.cat multiples)
    print(f"💾 Processing {len(batch_samples)} batch_samples with memory-optimized pre-allocation")
    
    # 1. Calculer la taille totale finale
    total_frames = sum(batch.shape[0] for batch in batch_samples)
    if len(batch_samples) > 0:
        sample_shape = batch_samples[0].shape
        H, W, C = sample_shape[1], sample_shape[2], sample_shape[3]
        print(f"📊 Total frames: {total_frames}, shape per frame: {H}x{W}x{C}")
        
        # 2. Pré-allouer le tensor final directement sur CPU (évite concatenations)
        final_video_images = torch.empty((total_frames, H, W, C), dtype=torch.float16)
        
        # 3. Copier par blocs directement dans le tensor final
        block_size = 500
        current_idx = 0
        
        for block_start in range(0, len(batch_samples), block_size):
            block_end = min(block_start + block_size, len(batch_samples))
            block_num = block_start // block_size + 1
            total_blocks = (len(batch_samples) + block_size - 1) // block_size
            
            print(f"🔄 Block {block_num}/{total_blocks}: batch_samples {block_start}-{block_end-1}")
            
            # Charger le bloc en VRAM
            current_block = []
            for i in range(block_start, block_end):
                current_block.append(batch_samples[i].to(device))
            
            # Concatener en VRAM (rapide)
            block_result = torch.cat(current_block, dim=0)
            
            # Convertir en Float16 sur GPU
            #if block_result.dtype != torch.float16:
            #    block_result = block_result.to(torch.float16)
            
            # Copier directement dans le tensor final (pas de concatenation!)
            block_frames = block_result.shape[0]
            final_video_images[current_idx:current_idx + block_frames] = block_result.to("cpu")
            current_idx += block_frames
            
            # Nettoyage immédiat VRAM
            del current_block, block_result
            torch.cuda.empty_cache()
            
        print(f"✅ Pre-allocation strategy completed: {final_video_images.shape}")
    else:
        print("⚠️ No batch_samples to process")
        final_video_images = torch.empty((0, 0, 0, 0), dtype=torch.float16)
    
    # Cleanup batch_samples
    #del batch_samples
    
    # Calculate total processing time
    if batch_times:
        total_time = time.time() - start_time
        if total_time > 60:
            total_minutes = int(total_time / 60)
            total_secs = int(total_time % 60)
            time_str = f"{total_minutes}m {total_secs}s"
        else:
            time_str = f"{int(total_time)}s"
        
        print(f"\n✅ Generation complete! Total time: {time_str} | Processed {len(batch_times)} batches | Avg: {sum(batch_times)/len(batch_times):.1f}s per batch")
    
    return final_video_images


def prepare_video_transforms(res_w):
    """
    Prepare optimized video transformation pipeline
    
    Args:
        res_w (int): Target resolution width
        
    Returns:
        Compose: Configured transformation pipeline
        
    Features:
        - Resolution-aware upscaling (no downsampling)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
    return Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            downsample_only=False,  # Model trained for high resolution
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w
    ])


def load_text_embeddings(script_directory, device, dtype):
    """
    Load and prepare text embeddings for generation
    
    Args:
        script_directory (str): Script directory path
        device (str): Target device
        dtype (torch.dtype): Target dtype
        
    Returns:
        dict: Text embeddings dictionary
        
    Features:
        - Adaptive dtype handling
        - Device-optimized loading
        - Memory-efficient embedding preparation
    """
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=dtype)
    
    return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}


def calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap):
    """
    Calculate optimal batch processing parameters
    
    Args:
        total_frames (int): Total number of frames
        batch_size (int): Desired batch size
        temporal_overlap (int): Temporal overlap frames
        
    Returns:
        dict: Optimized parameters and recommendations
        
    Features:
        - 4n+1 constraint optimization
        - Padding waste calculation
        - Performance recommendations
    """
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Find optimal batch sizes (4n+1 constraint)
    optimal_batches = [x for x in [i for i in range(1, 200) if i % 4 == 1] if x <= total_frames]
    best_batch = max(optimal_batches) if optimal_batches else 1
    
    # Calculate potential padding waste
    padding_waste = 0
    if batch_size not in optimal_batches:
        padding_waste = sum(((i // 4) + 1) * 4 + 1 - i for i in range(batch_size, total_frames, batch_size))
    
    return {
        'step': step,
        'temporal_overlap': temporal_overlap,
        'best_batch': best_batch,
        'padding_waste': padding_waste,
        'is_optimal': batch_size in optimal_batches
    } 