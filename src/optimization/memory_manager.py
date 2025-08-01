"""
Memory management module for SeedVR2
Handles VRAM usage, cache management, and memory optimization

Extracted from: seedvr2.py (lines 373-405, 607-626, 1016-1044)
"""

import os
import torch
import gc
import time
from typing import Tuple, Optional
from src.common.cache import Cache
from src.models.dit_v2.rope import RotaryEmbeddingBase
# Use local comfy copy from STAR instead of system ComfyUI
import sys
star_logic_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'STAR', 'logic')
if star_logic_path not in sys.path:
    sys.path.insert(0, star_logic_path)
from comfy import model_management as mm

def get_basic_vram_info():
    """🔍 Méthode basique avec PyTorch natif"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # Mémoire libre et totale (en bytes)
    free_memory, total_memory = torch.cuda.mem_get_info()
    
    # Conversion en GB
    free_gb = free_memory / (1024**3)
    total_gb = total_memory / (1024**3)
    
    return {
        "free_gb": free_gb,
        "total_gb": total_gb
    }

# Utilisation
vram_info = get_basic_vram_info()
print(f"VRAM libre: {vram_info['free_gb']:.2f} GB")

def get_vram_usage() -> Tuple[float, float, float]:
    """
    Get current VRAM usage (allocated, reserved, peak)
    
    Returns:
        tuple: (allocated_gb, reserved_gb, max_allocated_gb)
               Returns (0, 0, 0) if CUDA not available
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        return allocated, reserved, max_allocated
    return 0, 0, 0


def log_vram_usage(stage: str, details: str = "") -> None:
    """
    Log comprehensive VRAM usage at a specific stage
    
    Args:
        stage: Description of the current stage (e.g., "Before Inference", "After VAE")
        details: Additional details to include in the log
    """
    if torch.cuda.is_available():
        # Get memory info
        free_memory, total_memory = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        
        # Convert to GB
        total_gb = total_memory / (1024**3)
        free_gb = free_memory / (1024**3)
        used_gb = (total_memory - free_memory) / (1024**3)
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        max_allocated_gb = max_allocated / (1024**3)
        
        # Calculate percentage
        usage_percent = (used_gb / total_gb) * 100
        
        print(f"\n{'='*60}")
        print(f"📊 VRAM Status - {stage}")
        if details:
            print(f"   {details}")
        print(f"{'='*60}")
        print(f"🎮 Total GPU VRAM:     {total_gb:.2f} GB")
        print(f"🔵 Used (Total):       {used_gb:.2f} GB ({usage_percent:.1f}%)")
        print(f"🟢 Free:               {free_gb:.2f} GB")
        print(f"🟡 PyTorch Allocated:  {allocated_gb:.2f} GB")
        print(f"🟠 PyTorch Reserved:   {reserved_gb:.2f} GB")
        print(f"🔴 Peak Allocated:     {max_allocated_gb:.2f} GB")
        print(f"{'='*60}\n")


def clear_vram_cache() -> None:
    """
    Clear VRAM cache and run garbage collection
    """
    print("🧹 Clearing VRAM cache...")
    if torch.cuda.is_available():
        # Multiple rounds of cleanup for thorough memory defragmentation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def get_gpu_profile() -> dict:
    """
    Detect GPU capabilities and return optimal memory settings.
    
    Returns:
        dict: GPU profile with recommended settings
    """
    if not torch.cuda.is_available():
        return {
            'name': 'No GPU',
            'total_memory_gb': 0,
            'profile': 'cpu',
            'memory_reserved_threshold': 4.0,
            'memory_fraction_low': 0.8,
            'memory_fraction_high': 0.6,
            'block_cleanup_threshold': 0.7,
            'io_cleanup_threshold': 0.9,
            'recommended_blocks_to_swap': 0
        }
    
    # Get GPU info
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory_gb = gpu_props.total_memory / (1024**3)
    gpu_name = gpu_props.name
    
    # Determine GPU profile based on VRAM
    if total_memory_gb >= 24:  # High-end GPUs (RTX 4090, A100, etc.)
        profile = {
            'name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'profile': 'high_end',
            'memory_reserved_threshold': 8.0,  # Can handle more reserved memory
            'memory_fraction_low': 0.9,  # Can use more memory
            'memory_fraction_high': 0.8,
            'block_cleanup_threshold': 0.85,  # Less aggressive cleanup
            'io_cleanup_threshold': 0.95,
            'recommended_blocks_to_swap': 0  # Usually no swap needed
        }
    elif total_memory_gb >= 12:  # Mid-range GPUs (RTX 3080, 4070Ti, etc.)
        profile = {
            'name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'profile': 'mid_range',
            'memory_reserved_threshold': 4.0,
            'memory_fraction_low': 0.8,
            'memory_fraction_high': 0.6,
            'block_cleanup_threshold': 0.7,
            'io_cleanup_threshold': 0.85,
            'recommended_blocks_to_swap': 8  # Moderate swap for 7B models
        }
    elif total_memory_gb >= 8:  # Entry GPUs (RTX 3060, 4060, etc.)
        profile = {
            'name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'profile': 'entry_level',
            'memory_reserved_threshold': 2.0,
            'memory_fraction_low': 0.7,
            'memory_fraction_high': 0.5,
            'block_cleanup_threshold': 0.6,
            'io_cleanup_threshold': 0.8,
            'recommended_blocks_to_swap': 16  # Aggressive swap needed
        }
    else:  # Low VRAM GPUs (< 8GB)
        profile = {
            'name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'profile': 'low_vram',
            'memory_reserved_threshold': 1.5,
            'memory_fraction_low': 0.6,
            'memory_fraction_high': 0.4,
            'block_cleanup_threshold': 0.5,
            'io_cleanup_threshold': 0.7,
            'recommended_blocks_to_swap': 24  # Maximum swap
        }
    
    return profile


def get_intelligent_memory_config(model_type='3b', batch_size=5, resolution=1024) -> dict:
    """
    Get intelligent memory configuration based on GPU, model, and task.
    
    Args:
        model_type: '3b' or '7b' model size
        batch_size: Batch size for processing
        resolution: Target resolution
        
    Returns:
        dict: Optimized memory configuration
    """
    gpu_profile = get_gpu_profile()
    
    # Estimate memory requirements
    if model_type == '7b':
        base_memory_required = 14.0  # GB for 7B model
    else:
        base_memory_required = 8.0  # GB for 3B model
    
    # Adjust for batch size and resolution
    memory_multiplier = 1.0
    if batch_size > 5:
        memory_multiplier += (batch_size - 5) * 0.1
    if resolution > 1024:
        memory_multiplier += (resolution / 1024 - 1) * 0.2
    
    total_required = base_memory_required * memory_multiplier
    
    # Determine optimal block swap based on available VRAM
    available_vram = gpu_profile['total_memory_gb']
    
    if total_required > available_vram * 0.9:
        # Need aggressive block swap
        blocks_to_swap = min(28, int((total_required - available_vram * 0.7) / 0.5))
    elif total_required > available_vram * 0.7:
        # Need moderate block swap
        blocks_to_swap = min(16, int((total_required - available_vram * 0.6) / 0.5))
    else:
        # No or minimal block swap needed
        blocks_to_swap = 0
    
    # Override with GPU profile recommendation if lower
    blocks_to_swap = max(blocks_to_swap, gpu_profile['recommended_blocks_to_swap'])
    
    return {
        'gpu_profile': gpu_profile['profile'],
        'gpu_name': gpu_profile['name'],
        'total_vram_gb': gpu_profile['total_memory_gb'],
        'blocks_to_swap': blocks_to_swap,
        'memory_settings': {
            'memory_reserved_threshold': gpu_profile['memory_reserved_threshold'],
            'memory_fraction_low_reserved': gpu_profile['memory_fraction_low'],
            'memory_fraction_high_reserved': gpu_profile['memory_fraction_high'],
            'block_memory_cleanup_threshold': gpu_profile['block_cleanup_threshold'],
            'io_memory_cleanup_threshold': gpu_profile['io_cleanup_threshold']
        },
        'estimated_memory_usage': total_required,
        'optimization_level': 'aggressive' if blocks_to_swap > 16 else 'moderate' if blocks_to_swap > 0 else 'none'
    }


def release_reserved_memory() -> tuple:
    """
    Aggressively release PyTorch reserved memory.
    Returns (allocated_before, reserved_before, allocated_after, reserved_after) in GB
    """
    if not torch.cuda.is_available():
        return (0, 0, 0, 0)
    
    # Get initial state
    allocated_before = torch.cuda.memory_allocated() / 1024**3
    reserved_before = torch.cuda.memory_reserved() / 1024**3
    
    # Step 1: Clear all caches
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Step 2: Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Step 3: Force memory release by creating and deleting a small tensor
    # This tricks PyTorch into releasing unused reserved memory
    try:
        dummy = torch.zeros(1, device='cuda')
        del dummy
    except:
        pass
    
    # Step 4: Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Get final state
    allocated_after = torch.cuda.memory_allocated() / 1024**3
    reserved_after = torch.cuda.memory_reserved() / 1024**3
    
    return (allocated_before, reserved_before, allocated_after, reserved_after)


def reset_vram_peak() -> None:
    """
    Reset VRAM peak counter for new tracking
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def preinitialize_rope_cache(runner) -> None:
    """
    🚀 Pre-initialize RoPE cache to avoid OOM at first launch
    
    Args:
        runner: The model runner containing DiT and VAE models
    """
    
    try:
        # Create dummy tensors to simulate common shapes
        # Format: [batch, channels, frames, height, width] for vid_shape
        # Format: [batch, seq_len] for txt_shape
        common_shapes = [
            # Common video resolutions
            (torch.tensor([[1, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 1 frame, 77 tokens
            (torch.tensor([[4, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 4 frames
            (torch.tensor([[5, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 5 frames (4n+1 format)
            (torch.tensor([[1, 4, 4]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # Higher resolution
        ]
        
        # Create mock cache for pre-initialization
            
        temp_cache = Cache()
        
        # Access RoPE modules in DiT (recursive search)
        def find_rope_modules(module):
            rope_modules = []
            for name, child in module.named_modules():
                if hasattr(child, 'get_freqs') and callable(getattr(child, 'get_freqs')):
                    rope_modules.append((name, child))
            return rope_modules
        
        rope_modules = find_rope_modules(runner.dit)
        
        # Pre-calculate for each RoPE module found
        for name, rope_module in rope_modules:
            # Temporarily move module to CPU if necessary
            original_device = next(rope_module.parameters()).device if list(rope_module.parameters()) else torch.device('cpu')
            rope_module.to('cpu')
            
            try:
                for vid_shape, txt_shape in common_shapes:
                    cache_key = f"720pswin_by_size_bysize_{tuple(vid_shape[0].tolist())}_sd3.mmrope_freqs_3d"
                    
                    def compute_freqs():
                        try:
                            # Calculate with reduced dimensions to avoid OOM
                            with torch.no_grad():
                                # Detect RoPE module type
                                module_type = type(rope_module).__name__
                                
                                if module_type == 'NaRotaryEmbedding3d':
                                    # NaRotaryEmbedding3d: only takes shape (vid_shape)
                                    return rope_module.get_freqs(vid_shape.cpu())
                                else:
                                    # Standard RoPE: takes vid_shape and txt_shape
                                    return rope_module.get_freqs(vid_shape.cpu(), txt_shape.cpu())
                                    
                        except Exception as e:
                            print(f"      ⚠️ Failed for {cache_key}: {e}")
                            # Return empty tensors as fallback
                            time.sleep(1)
                            clear_vram_cache()

                            return torch.zeros(1, 64)
                    
                    # Store in cache
                    temp_cache(cache_key, compute_freqs)
                
            except Exception as e:
                print(f"    ❌ Error in module {name}: {e}")
            finally:
                # Restore to original device
                rope_module.to(original_device)
        
        # Copy temporary cache to runner cache
        if hasattr(runner, 'cache'):
            runner.cache.cache.update(temp_cache.cache)
        else:
            runner.cache = temp_cache
        
    except Exception as e:
        print(f"  ⚠️ Error during RoPE pre-init: {e}")
        print("  🔄 Model will work but could have OOM at first launch")


def clear_rope_cache(runner) -> None:
    """
    🧹 Clear RoPE cache to free VRAM
    
    Args:
        runner: The model runner containing the cache
    """
    print("🧹 Cleaning RoPE cache...")
    
    if hasattr(runner, 'cache') and hasattr(runner.cache, 'cache'):
        # Count entries before cleanup
        cache_size = len(runner.cache.cache)
        
        # Free all tensors from cache
        for key, value in runner.cache.cache.items():
            if isinstance(value, (tuple, list)):
                for item in value:
                    if hasattr(item, 'cpu'):
                        item.cpu()
                        del item
            elif hasattr(value, 'cpu'):
                value.cpu()
                del value
        
        # Clear the cache
        runner.cache.cache.clear()
        print(f"  ✅ RoPE cache cleared ({cache_size} entries removed)")

    if hasattr(runner, 'dit'):
        cleared_lru_count = 0
        for module in runner.dit.modules():
            if isinstance(module, RotaryEmbeddingBase):
                if hasattr(module.get_axial_freqs, 'cache_clear'):
                    module.get_axial_freqs.cache_clear()
                    cleared_lru_count += 1
        if cleared_lru_count > 0:
            print(f"  ✅ Cleared {cleared_lru_count} LRU caches from RoPE modules.")
    # Aggressive VRAM cleanup
    # clear_vram_cache()
    #torch.cuda.empty_cache()
    #clear_vram_cache()
    
    print("🎯 RoPE cache cleanup completed!")


def clear_rope_lru_caches(model) -> int:
    """Clear ALL LRU caches from RoPE modules"""
    cleared_count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'get_axial_freqs') and hasattr(module.get_axial_freqs, 'cache_clear'):
            module.get_axial_freqs.cache_clear()
            cleared_count += 1
    
    return cleared_count


def fast_model_cleanup(model):
    """Fast model cleanup without logs"""
    if model is None:
        return
    
    # Move to CPU
    model.to("cpu")
    
    # Clear parameters and buffers recursively
    def clear_recursive(m):
        for child in m.children():
            clear_recursive(child)
        for param in m.parameters():
            if param is not None:
                param.data = param.data.cpu()
                param.grad = None
        for buffer in m.buffers():
            if buffer is not None:
                buffer.data = buffer.data.cpu()
    
    clear_recursive(model)


def fast_ram_cleanup():
    """Fast RAM cleanup without excessive logging"""
    # Garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Clear PyTorch internal caches
    try:
        torch._C._clear_cache()
    except:
        pass
    

def clear_all_caches(runner, debugger=None) -> int:
    """
    Aggressively clear all caches from runner and model.
    Optimized to only process what's necessary.
    
    Args:
        runner: The runner instance to clear caches from
        debugger: Optional BlockSwapDebugger instance for logging
    """
    if not runner:
        return 0
    
    # Try to get debugger from runner if not provided
    if debugger is None and hasattr(runner, '_blockswap_debugger'):
        debugger = runner._blockswap_debugger
    
    # Helper function for logging
    def log_message(message, level="INFO"):
        if debugger and debugger.enabled:
            debugger.log(message, level)
        
    cleaned_items = 0
    
    # Early exit if no caches to clear
    has_cache = hasattr(runner, 'cache') and hasattr(runner.cache, 'cache')
    if not has_cache and not hasattr(runner, 'dit'):
        return 0
    
    # Clear main runner cache efficiently
    if has_cache and runner.cache.cache:
        cache_entries = len(runner.cache.cache)
        
        # Process all cache items to properly free memory
        for key, value in list(runner.cache.cache.items()):
            if torch.is_tensor(value):
                # Force deallocation of tensor storage
                if value.is_cuda:
                    value.data = value.data.cpu()
                value.grad = None
                if value.numel() > 0:
                    value.data.set_()  # Release underlying storage
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if torch.is_tensor(item):
                        if item.is_cuda:
                            item.data = item.data.cpu()
                        item.grad = None
                        if item.numel() > 0:
                            item.data.set_()
        
        # Clear the cache after processing
        runner.cache.cache.clear()
        cleaned_items += cache_entries
        log_message(f"✅ Cleared {cache_entries} cache entries")
    
    # Clear any accumulated state in blocks
    if hasattr(runner, 'dit'):
        model = runner.dit
        if hasattr(model, 'dit_model'):
            model = model.dit_model

        # Clear RoPE LRU caches
        rope_caches_cleared = clear_rope_lru_caches(model)
        cleaned_items += rope_caches_cleared
        if rope_caches_cleared > 0:
            log_message(f"✅ Cleared {rope_caches_cleared} RoPE LRU caches")
            
        # Clear block attributes if needed
        if hasattr(model, 'blocks'):
            block_attrs_cleared = 0
            
            # Define PyTorch's essential attributes that must NOT be deleted
            essential_attrs = {
                '_modules', '_parameters', '_buffers', 
                '_forward_hooks', '_forward_pre_hooks', 
                '_backward_hooks', '_backward_pre_hooks',
                '_state_dict_hooks', '_state_dict_pre_hooks',
                '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks',
                '_non_persistent_buffers_set', '_version',
                '_is_full_backward_hook', 'training',
                '_original_forward',  # BlockSwap attribute
                '_is_io_wrapped',     # BlockSwap attribute
                '_block_idx',         # BlockSwap attribute
            }
            
            for idx, block in enumerate(model.blocks):
                # Get all attributes that look like caches
                attrs_to_remove = []
                for attr_name in list(block.__dict__.keys()):
                    # Only remove cache-like attributes, not essential PyTorch attributes
                    if (attr_name not in essential_attrs and 
                        ('cache' in attr_name or 
                         'temp' in attr_name or 
                         (attr_name.startswith('_') and 
                          not attr_name.startswith('__') and 
                          attr_name not in essential_attrs))):
                        attrs_to_remove.append(attr_name)
                
                # Remove the identified attributes
                for attr_name in attrs_to_remove:
                    try:
                        delattr(block, attr_name)
                        block_attrs_cleared += 1
                    except AttributeError:
                        pass  # Already deleted or doesn't exist
                            
            if block_attrs_cleared > 0:
                log_message(f"✅ Cleared {block_attrs_cleared} temporary attributes from blocks")
    
    # Clear any temporary attributes that might accumulate
    temp_attrs = ['_temp_cache', '_block_cache', '_swap_cache', '_generation_cache',
                  '_rope_cache', '_intermediate_cache', '_backward_cache']
    
    # Check both runner and model for these attributes
    for obj in [runner, getattr(runner, 'dit', None)]:
        if obj is None:
            continue
            
        # Handle wrapped models
        if hasattr(obj, 'dit_model'):
            obj = obj.dit_model
            
        for attr in temp_attrs:
            if hasattr(obj, attr):
                delattr(obj, attr)
                cleaned_items += 1
                log_message(f"✅ Cleared {attr} from {type(obj).__name__}")
                
    # Force garbage collection
    gc.collect(2)  # Collect all generations
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mm.soft_empty_cache()

    return cleaned_items