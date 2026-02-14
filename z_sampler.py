from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
import math
from PIL import Image
import numpy as np

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

# Setup logging
logger = logging.getLogger("Z-Sampling")


class ZPhase(Enum):
    """Current phase in the zigzag sampling process."""
    DENOISE = "denoise"
    INVERT = "invert"
    ZIGZAG = "zigzag"


@dataclass
class ZSamplingConfig:
    """Immutable configuration for Z-Sampling parameters."""
    gamma_1: float = 5.5
    gamma_2: float = 0.0
    lambda_step: int = -1
    t_max: int = 1
    
    def resolve_lambda_step(self, steps: int) -> int:
        """Resolve auto lambda_step (-1) to actual value."""
        if self.lambda_step < 0:
            return steps - 1
        return min(self.lambda_step, steps - 1)


# =============================================================================
# Latent2RGB Preview System (No TAESD Required)
# =============================================================================

# Standard latent-to-RGB transformation matrices
# These approximate the VAE decode by mapping latent channels to RGB directly

# SD 1.x / SD 2.x (4 channels)
LATENT_RGB_FACTORS_SD = [
    #   R        G        B
    [ 0.298,  0.207,  0.208],  # L0
    [ 0.187,  0.286,  0.173],  # L1
    [-0.158,  0.189,  0.264],  # L2
    [-0.184, -0.271, -0.473],  # L3
]

LATENT_RGB_FACTORS_SD_BIAS = [0.0, 0.0, 0.0]

# SDXL (4 channels, slightly different mapping)
LATENT_RGB_FACTORS_SDXL = [
    #   R        G        B
    [ 0.3512,  0.2297,  0.2463],  # L0
    [ 0.1843,  0.3049,  0.1705],  # L1  
    [-0.1891,  0.1795,  0.2781],  # L2
    [-0.2137, -0.2693, -0.4331],  # L3
]

LATENT_RGB_FACTORS_SDXL_BIAS = [0.1059, 0.1081, 0.0929]

# Cosmos/Video models (16 channels) - PCA-derived approximation
# This is an approximation; actual factors would need model-specific tuning
LATENT_RGB_FACTORS_16CH = [
    #   R        G        B
    [ 0.20,  0.15,  0.18],  # L0
    [ 0.15,  0.22,  0.12],  # L1
    [-0.10,  0.14,  0.20],  # L2
    [-0.14, -0.18, -0.30],  # L3
    [ 0.08,  0.05,  0.06],  # L4
    [ 0.05,  0.08,  0.04],  # L5
    [-0.03,  0.05,  0.07],  # L6
    [-0.05, -0.06, -0.10],  # L7
    [ 0.04,  0.03,  0.03],  # L8
    [ 0.03,  0.04,  0.02],  # L9
    [-0.02,  0.03,  0.04],  # L10
    [-0.03, -0.03, -0.05],  # L11
    [ 0.02,  0.01,  0.02],  # L12
    [ 0.01,  0.02,  0.01],  # L13
    [-0.01,  0.01,  0.02],  # L14
    [-0.01, -0.02, -0.03],  # L15
]

LATENT_RGB_FACTORS_16CH_BIAS = [0.08, 0.08, 0.07]


class Latent2RGBPreviewer:
    """
    Converts latent tensors to RGB images using learned transformation matrices.
    
    This is a fast approximation that doesn't require TAESD models.
    Quality is lower than TAESD but works universally.
    """
    
    def __init__(
        self, 
        latent_rgb_factors: list[list[float]],
        latent_rgb_factors_bias: Optional[list[float]] = None,
        max_preview_size: int = 512
    ):
        """
        Initialize latent2rgb previewer.
        
        Args:
            latent_rgb_factors: Matrix mapping latent channels to RGB [C, 3]
            latent_rgb_factors_bias: Optional bias for RGB output [3]
            max_preview_size: Maximum preview dimension (for memory efficiency)
        """
        self.latent_rgb_factors = torch.tensor(
            latent_rgb_factors, 
            dtype=torch.float32
        ).transpose(0, 1)  # Shape: [3, C]
        
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(
                latent_rgb_factors_bias,
                dtype=torch.float32
            )
        
        self.max_preview_size = max_preview_size
        self._device = None
        self._dtype = None
    
    def _ensure_device(self, x: Tensor) -> None:
        """Move factors to same device/dtype as input if needed."""
        if self._device != x.device or self._dtype != x.dtype:
            self.latent_rgb_factors = self.latent_rgb_factors.to(
                device=x.device, dtype=x.dtype
            )
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(
                    device=x.device, dtype=x.dtype
                )
            self._device = x.device
            self._dtype = x.dtype
    
    def decode_latent_to_preview(self, x0: Tensor) -> Image.Image:
        """
        Convert latent tensor to PIL Image.
        
        Args:
            x0: Latent tensor [B, C, H, W] or [B, C, T, H, W] for video
            
        Returns:
            PIL Image of the preview
        """
        self._ensure_device(x0)
        
        # Handle 5D video latents - take middle frame
        if x0.ndim == 5:
            mid_frame = x0.shape[2] // 2
            x0 = x0[:, :, mid_frame, :, :]
        
        # Take first batch item: [C, H, W]
        latent = x0[0]
        
        # Handle channel count mismatch
        actual_channels = latent.shape[0]
        expected_channels = self.latent_rgb_factors.shape[1]
        
        if actual_channels != expected_channels:
            # Truncate or pad channels as needed
            if actual_channels > expected_channels:
                # Use only the first N channels
                latent = latent[:expected_channels]
            else:
                # Pad with zeros (shouldn't normally happen)
                padding = torch.zeros(
                    expected_channels - actual_channels,
                    latent.shape[1],
                    latent.shape[2],
                    device=latent.device,
                    dtype=latent.dtype
                )
                latent = torch.cat([latent, padding], dim=0)
        
        # Reshape for linear transform: [H, W, C]
        latent_hwc = latent.movedim(0, -1)
        
        # Apply transformation: [H, W, C] @ [C, 3] = [H, W, 3]
        rgb = torch.nn.functional.linear(
            latent_hwc,
            self.latent_rgb_factors,
            bias=self.latent_rgb_factors_bias
        )
        
        # Scale from latent range to image range
        # Latent values are roughly in [-4, 4], we map to [0, 1]
        rgb = (rgb + 1.0) / 2.0
        rgb = rgb.clamp(0, 1)
        
        # Resize if too large (memory efficiency)
        h, w = rgb.shape[:2]
        if max(h, w) > self.max_preview_size:
            scale = self.max_preview_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
            rgb = torch.nn.functional.interpolate(
                rgb, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            rgb = rgb.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        
        # Convert to uint8 and PIL Image
        rgb_uint8 = (rgb * 255).to(dtype=torch.uint8, device='cpu')
        
        return Image.fromarray(rgb_uint8.numpy())
    
    def decode_latent_to_preview_image(
        self, 
        preview_format: str, 
        x0: Tensor
    ) -> tuple[str, Image.Image, int]:
        """
        Decode for ComfyUI preview system compatibility.
        
        Returns:
            Tuple of (format, image, max_size)
        """
        preview_image = self.decode_latent_to_preview(x0)
        return (preview_format, preview_image, self.max_preview_size)


def get_latent2rgb_previewer(
    latent_channels: int = 4,
    model_type: str = "auto",
    max_preview_size: int = 512
) -> Latent2RGBPreviewer:
    """
    Factory function to create appropriate latent2rgb previewer.
    
    Args:
        latent_channels: Number of latent channels (4 for SD/SDXL, 16 for Cosmos)
        model_type: "sd", "sdxl", "cosmos", or "auto"
        max_preview_size: Maximum preview dimension
        
    Returns:
        Configured Latent2RGBPreviewer instance
    """
    if model_type == "auto":
        if latent_channels == 16:
            model_type = "cosmos"
        elif latent_channels == 4:
            model_type = "sdxl"  # SDXL factors work well for SD too
        else:
            model_type = "sd"
    
    if model_type == "cosmos" or latent_channels == 16:
        return Latent2RGBPreviewer(
            LATENT_RGB_FACTORS_16CH,
            LATENT_RGB_FACTORS_16CH_BIAS,
            max_preview_size
        )
    elif model_type == "sdxl":
        return Latent2RGBPreviewer(
            LATENT_RGB_FACTORS_SDXL,
            LATENT_RGB_FACTORS_SDXL_BIAS,
            max_preview_size
        )
    else:  # sd
        return Latent2RGBPreviewer(
            LATENT_RGB_FACTORS_SD,
            LATENT_RGB_FACTORS_SD_BIAS,
            max_preview_size
        )


def try_get_taesd_previewer(
    device: torch.device,
    latent_format: Any
) -> Optional[Any]:
    """
    Attempt to get TAESD previewer if available.
    
    Returns None if TAESD is not installed or configured.
    """
    try:
        import latent_preview
        from comfy.cli_args import args, LatentPreviewMethod
        
        if args.preview_method == LatentPreviewMethod.NoPreviews:
            return None
        
        previewer = latent_preview.get_previewer(device, latent_format)
        return previewer
        
    except Exception as e:
        logger.debug(f"TAESD not available: {e}")
        return None


# =============================================================================
# Z-Sampling Preview Callback System
# =============================================================================

class ZSamplingPreviewCallback:
    """
    Manages live preview generation during Z-Sampling.
    
    Supports multiple preview methods with automatic fallback:
    1. TAESD (if available) - highest quality
    2. Latent2RGB - fast, no dependencies
    3. Raw channel visualization - last resort
    """
    
    def __init__(
        self,
        model: Any,
        steps: int,
        t_max: int,
        latent_channels: int = 4,
        preview_every: int = 1,
        show_zigzag_phases: bool = True,
        preview_method: str = "auto",  # "auto", "taesd", "latent2rgb"
        max_preview_size: int = 512
    ):
        """
        Initialize preview callback.
        
        Args:
            model: The model patcher for preview preparation
            steps: Total sampling steps
            t_max: Zigzag rounds per step
            latent_channels: Number of latent channels (4 or 16)
            preview_every: Show preview every N steps (1 = every step)
            show_zigzag_phases: Whether to show intermediate zigzag previews
            preview_method: Preview method to use
            max_preview_size: Maximum preview image dimension
        """
        self.steps = steps
        self.t_max = t_max
        self.latent_channels = latent_channels
        self.preview_every = max(1, preview_every)
        self.show_zigzag_phases = show_zigzag_phases
        self.max_preview_size = max_preview_size
        
        # Setup previewer with fallback chain
        self.previewer = self._setup_previewer(model, preview_method)
        self.preview_method_name = self._get_preview_method_name()
        
        # Progress tracking
        self.current_step = 0
        self.current_phase = ZPhase.DENOISE
        self.current_zigzag_round = 0
        
        # Progress bar
        self.pbar = comfy.utils.ProgressBar(steps)
        
        logger.info(f"Preview initialized: method={self.preview_method_name}, "
                   f"channels={latent_channels}, max_size={max_preview_size}")
    
    def _setup_previewer(self, model: Any, method: str) -> Any:
        """Setup previewer with fallback chain."""
        
        # Try TAESD first if requested or auto
        if method in ("auto", "taesd"):
            try:
                device = (model.load_device 
                         if hasattr(model, 'load_device') 
                         else comfy.model_management.get_torch_device())
                
                latent_format = None
                if hasattr(model, 'model') and hasattr(model.model, 'latent_format'):
                    latent_format = model.model.latent_format
                
                taesd_previewer = try_get_taesd_previewer(device, latent_format)
                
                if taesd_previewer is not None:
                    logger.info("Using TAESD previewer")
                    self._previewer_type = "taesd"
                    return taesd_previewer
                    
            except Exception as e:
                logger.debug(f"TAESD setup failed: {e}")
        
        # Fallback to latent2rgb
        logger.info("Using latent2rgb previewer (no TAESD)")
        self._previewer_type = "latent2rgb"
        return get_latent2rgb_previewer(
            latent_channels=self.latent_channels,
            model_type="auto",
            max_preview_size=self.max_preview_size
        )
    
    def _get_preview_method_name(self) -> str:
        """Get human-readable preview method name."""
        return getattr(self, '_previewer_type', 'latent2rgb')
    
    def _decode_preview(self, latent: Tensor) -> Optional[Image.Image]:
        """Decode latent to preview image."""
        if self.previewer is None:
            return None
        
        try:
            # Both TAESD and Latent2RGB have decode_latent_to_preview method
            preview = self.previewer.decode_latent_to_preview(latent)
            return preview
            
        except Exception as e:
            logger.debug(f"Preview decode failed: {e}")
            return self._raw_latent_preview(latent)
    
    def _raw_latent_preview(self, latent: Tensor) -> Image.Image:
        """
        Last-resort raw visualization of latent channels.
        
        Shows first 3 channels as RGB directly.
        """
        try:
            # Handle 5D
            if latent.ndim == 5:
                latent = latent[:, :, latent.shape[2] // 2, :, :]
            
            # Take first 3 channels for RGB
            rgb = latent[0, :3].cpu()
            
            # Normalize each channel independently
            for i in range(3):
                ch = rgb[i]
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max - ch_min > 1e-8:
                    rgb[i] = (ch - ch_min) / (ch_max - ch_min)
                else:
                    rgb[i] = torch.zeros_like(ch)
            
            # [3, H, W] -> [H, W, 3]
            rgb = rgb.permute(1, 2, 0)
            rgb_uint8 = (rgb * 255).to(torch.uint8).numpy()
            
            return Image.fromarray(rgb_uint8)
            
        except Exception as e:
            logger.warning(f"Raw preview failed: {e}")
            # Return tiny placeholder
            return Image.new('RGB', (64, 64), color='gray')
    
    def __call__(self, callback_data: dict) -> None:
        """
        Process sampling callback and generate preview.
        
        Args:
            callback_data: Dict with keys like 'i', 'denoised', 'x', 'sigma', 'phase', etc.
        """
        step = callback_data.get('i', 0)
        denoised = callback_data.get('denoised')
        x = callback_data.get('x')
        phase = callback_data.get('phase', ZPhase.DENOISE)
        zigzag_round = callback_data.get('zigzag_round', 0)
        
        self.current_step = step
        self.current_phase = phase
        self.current_zigzag_round = zigzag_round
        
        # Determine if we should show preview
        should_preview = False
        
        if phase == ZPhase.DENOISE:
            # Always preview main denoise steps at interval
            should_preview = (step % self.preview_every == 0) or (step == self.steps - 1)
        elif self.show_zigzag_phases:
            # Optionally show zigzag intermediate steps
            should_preview = (step % self.preview_every == 0)
        
        if not should_preview or denoised is None:
            # Still update progress bar
            if phase == ZPhase.DENOISE:
                self._update_progress(step, None)
            return
        
        # Generate preview
        preview_image = self._decode_preview(denoised)
        
        # Update progress bar with preview
        if phase == ZPhase.DENOISE:
            self._update_progress(step, preview_image)
    
    def _update_progress(self, step: int, preview_image: Optional[Image.Image]) -> None:
        """Update ComfyUI progress bar with optional preview."""
        try:
            preview_bytes = None
            
            if preview_image is not None:
                # Convert PIL Image to bytes for ComfyUI
                import io
                buffer = io.BytesIO()
                preview_image.save(buffer, format='JPEG', quality=85)
                preview_bytes = ("JPEG", preview_image, self.max_preview_size)
            
            self.pbar.update_absolute(
                step + 1, 
                self.steps, 
                preview_bytes
            )
            
        except Exception as e:
            logger.debug(f"Progress update failed: {e}")
            # Fallback to simple progress update
            self.pbar.update(1)
    
    def get_progress_info(self) -> dict:
        """Get current progress information."""
        return {
            "step": self.current_step,
            "total_steps": self.steps,
            "phase": self.current_phase.value,
            "zigzag_round": self.current_zigzag_round,
            "t_max": self.t_max,
            "progress": (self.current_step + 1) / self.steps,
            "preview_method": self.preview_method_name
        }


# =============================================================================
# Model Detection
# =============================================================================

def detect_model_type(model: Any) -> tuple[bool, int]:
    """
    Detect model architecture and expected latent channels.
    
    Args:
        model: ComfyUI model patcher or model object
        
    Returns:
        Tuple of (is_cosmos_type, expected_channels)
    """
    is_cosmos = False
    channels = 4  # Default for SD/SDXL
    
    try:
        model_obj = getattr(model, 'model', model)
        model_class = model_obj.__class__.__name__
        
        # Check for Cosmos/Anima/Video models
        cosmos_classes = {"Anima", "CosmosPredict2", "CosmosVideo", "MiniTrainDIT", "GeneralDIT"}
        
        if model_class in cosmos_classes:
            is_cosmos = True
            channels = 16
        
        # Check diffusion model attributes
        diffusion_model = getattr(model_obj, 'diffusion_model', None)
        if diffusion_model is not None:
            dm_class = diffusion_model.__class__.__name__
            if dm_class in cosmos_classes:
                is_cosmos = True
                channels = 16
            
            # Check for video model indicators
            if hasattr(diffusion_model, 'x_embedder') or hasattr(diffusion_model, 'patch_temporal'):
                is_cosmos = True
                channels = 16
            
            # Get actual in_channels if available
            if hasattr(diffusion_model, 'in_channels'):
                channels = diffusion_model.in_channels
        
        # Check model config for latent format
        model_config = getattr(model_obj, 'model_config', None)
        if model_config and hasattr(model_config, 'latent_format'):
            lf = model_config.latent_format
            if hasattr(lf, 'latent_channels'):
                channels = lf.latent_channels
                
    except Exception as e:
        logger.debug(f"Model detection fallback: {e}")
    
    return is_cosmos, channels


# =============================================================================
# Z-Sampling Guider
# =============================================================================

class ZSamplingGuider(comfy.samplers.CFGGuider):
    """
    Extended CFGGuider supporting dynamic CFG switching for Z-Sampling.
    """
    
    def __init__(
        self, 
        model_patcher: comfy.model_patcher.ModelPatcher,
        cfg: float = 5.5
    ):
        super().__init__(model_patcher)
        self._cfg = cfg
        
    @property
    def cfg(self) -> float:
        return self._cfg
    
    @cfg.setter
    def cfg(self, value: float) -> None:
        self._cfg = value
        
    def predict_noise(
        self, 
        x: Tensor, 
        timestep: Tensor, 
        model_options: dict = None, 
        seed: Optional[int] = None
    ) -> Tensor:
        """Predict noise using current CFG scale."""
        if model_options is None:
            model_options = {}
            
        return comfy.samplers.sampling_function(
            self.inner_model, 
            x, 
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self._cfg,
            model_options=model_options,
            seed=seed
        )


# =============================================================================
# Z-Sampler Kernel
# =============================================================================

class ZSamplerKernel(comfy.samplers.Sampler):
    """
    Core sampling kernel implementing the Z-Sampling zigzag algorithm
    with live preview support.
    """
    
    def __init__(
        self, 
        gamma_1: float,
        gamma_2: float,
        lambda_step: int,
        t_max: int,
        base_sampler: str = "euler",
        preview_callback: Optional[ZSamplingPreviewCallback] = None
    ):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.lambda_step = lambda_step
        self.t_max = t_max
        self.base_sampler = base_sampler
        self.preview_callback = preview_callback
        
    def _euler_step(
        self,
        x: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        sigma_next: Tensor,
        denoise_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Perform a single Euler step."""
        if denoise_mask is not None:
            denoised = denoised * denoise_mask + x * (1 - denoise_mask)
        
        dt = sigma_next - sigma
        
        if sigma > 1e-8:
            d = (x - denoised) / sigma
            return x + d * dt
        return denoised
    
    def _inverse_euler_step(
        self,
        x_next: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        dt: Tensor,
        denoise_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Perform inverse Euler step for zigzag."""
        if denoise_mask is not None:
            denoised = denoised * denoise_mask + x_next * (1 - denoise_mask)
        
        if sigma > 1e-8:
            d_inv = (x_next - denoised) / sigma
            return x_next - d_inv * dt
        return denoised
    
    def _send_preview(
        self,
        step: int,
        denoised: Tensor,
        x: Tensor,
        sigma: Tensor,
        phase: ZPhase,
        zigzag_round: int = 0
    ) -> None:
        """Send preview update if callback is configured."""
        if self.preview_callback is not None:
            self.preview_callback({
                'i': step,
                'denoised': denoised,
                'x': x,
                'sigma': sigma,
                'phase': phase,
                'zigzag_round': zigzag_round
            })
        
    def sample(
        self, 
        model_wrap: ZSamplingGuider,
        sigmas: Tensor,
        extra_args: Optional[dict] = None,
        callback: Optional[Callable] = None,
        noise: Tensor = None,
        latent_image: Optional[Tensor] = None,
        denoise_mask: Optional[Tensor] = None,
        disable_pbar: bool = False
    ) -> Tensor:
        """
        Execute Z-Sampling loop with zigzag denoising-inversion and live previews.
        """
        extra_args = extra_args.copy() if extra_args else {}
        seed = extra_args.get("seed", 42)
        model_options = extra_args.get("model_options", {})
        
        # Initialize x with noise scaling
        x = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], 
            noise, 
            latent_image, 
            self.max_denoise(model_wrap, sigmas)
        )
        
        total_steps = len(sigmas) - 1
        effective_lambda = min(self.lambda_step, total_steps - 1)
        
        logger.info(f"Z-Sampling: {total_steps} steps, lambda={effective_lambda}, t_max={self.t_max}")
        
        for i in range(total_steps):
            # Check for interruption
            comfy.model_management.throw_exception_if_processing_interrupted()
            
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Expand sigma for batch
            sigma_batch = sigma.view(1).expand(x.shape[0])
            dt = sigma_next - sigma
            
            # === PHASE 1: Standard denoising with gamma_1 ===
            model_wrap.cfg = self.gamma_1
            denoised = model_wrap.predict_noise(
                x, sigma_batch, 
                model_options=model_options, 
                seed=seed
            )
            
            # Send denoise preview
            self._send_preview(i, denoised, x, sigma, ZPhase.DENOISE)
            
            x_next = self._euler_step(x, denoised, sigma, sigma_next, denoise_mask)
            
            # === PHASE 2: Zigzag optimization ===
            should_zigzag = (
                i < effective_lambda and 
                sigma_next > 1e-8 and 
                sigma > 1e-8
            )
            
            if should_zigzag:
                for t_round in range(self.t_max):
                    # --- ZAG: Inversion with gamma_2 ---
                    model_wrap.cfg = self.gamma_2
                    denoised_inv = model_wrap.predict_noise(
                        x_next, sigma_batch,
                        model_options=model_options,
                        seed=seed
                    )
                    
                    # Send inversion preview
                    self._send_preview(i, denoised_inv, x_next, sigma, ZPhase.INVERT, t_round)
                    
                    x_inv = self._inverse_euler_step(
                        x_next, denoised_inv, sigma, dt, denoise_mask
                    )
                    
                    # --- ZIG: Re-denoise with gamma_1 ---
                    model_wrap.cfg = self.gamma_1
                    denoised_zig = model_wrap.predict_noise(
                        x_inv, sigma_batch,
                        model_options=model_options,
                        seed=seed
                    )
                    
                    # Send zigzag preview
                    self._send_preview(i, denoised_zig, x_inv, sigma, ZPhase.ZIGZAG, t_round)
                    
                    x_next = self._euler_step(x_inv, denoised_zig, sigma, sigma_next, denoise_mask)
            
            x = x_next
            
            # Original callback (for compatibility)
            if callback is not None:
                callback({
                    "i": i, 
                    "denoised": denoised, 
                    "x": x,
                    "sigma": sigma,
                    "sigma_next": sigma_next
                })
        
        # Final inverse noise scaling
        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x)


# =============================================================================
# Z-Sampling Settings Node
# =============================================================================

class ZSamplingSettings:
    """
    Configuration node for Z-Sampling parameters.
    """
    
    CATEGORY = "sampling/z-sampling"
    FUNCTION = "get_settings"
    RETURN_TYPES = ("ZSETTINGS",)
    RETURN_NAMES = ("z_settings",)
    DESCRIPTION = "Configure Z-Sampling zigzag diffusion parameters."
    
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "gamma_1": ("FLOAT", {
                    "default": 5.5, 
                    "min": 1.0, 
                    "max": 30.0, 
                    "step": 0.1,
                    "tooltip": "CFG scale for denoising steps (strong guidance)"
                }),
                "gamma_2": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 15.0, 
                    "step": 0.1,
                    "tooltip": "CFG scale for inversion steps (weak/no guidance)"
                }),
                "lambda_step": ("INT", {
                    "default": -1, 
                    "min": -1, 
                    "max": 500,
                    "tooltip": "Steps with zigzag optimization. -1 = auto (steps-1)"
                }),
                "t_max": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 10,
                    "tooltip": "Zigzag rounds per step. More = slower but better quality"
                }),
            },
        }
    
    def get_settings(
        self, 
        gamma_1: float, 
        gamma_2: float, 
        lambda_step: int, 
        t_max: int
    ) -> tuple[dict[str, Any]]:
        config = ZSamplingConfig(
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            lambda_step=lambda_step,
            t_max=t_max
        )
        return (config.__dict__,)


# =============================================================================
# Main Z-Sampler Node
# =============================================================================

class ZSampler:
    """
    Z-Sampler: Full Zigzag Diffusion Sampling for ComfyUI with Live Previews.
    
    Features latent2rgb preview that works without TAESD dependencies.
    """
    
    CATEGORY = "sampling/z-sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    DESCRIPTION = "Advanced sampler with zigzag diffusion, dynamic CFG, and live latent2rgb previews."
    
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt"}),
                "latent_image": ("LATENT", {"tooltip": "Input latent"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 500}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "z_settings": ("ZSETTINGS", {"tooltip": "Optional Z-Sampling config"}),
                "preview_every": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Show preview every N steps"
                }),
                "show_zigzag_phases": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show intermediate zigzag phase previews"
                }),
                "preview_method": (["auto", "latent2rgb", "taesd"], {
                    "default": "auto",
                    "tooltip": "Preview method: auto tries TAESD first, falls back to latent2rgb"
                }),
            }
        }
    
    @classmethod
    def validate_inputs(cls, model, positive, negative, latent_image, **kwargs):
        if "samples" not in latent_image:
            return "Latent image must contain 'samples' key"
        return True
    
    def _prepare_latent(
        self,
        latent_image: dict,
        is_cosmos: bool,
        expected_channels: int
    ) -> tuple[Tensor, Optional[Tensor], bool, int]:
        """Prepare latent tensor with proper shape validation."""
        samples = latent_image["samples"].clone()
        noise_mask = latent_image.get("noise_mask", None)
        original_ndim = samples.dim()
        actual_channels = samples.shape[1]
        
        logger.info(f"Input latent: shape={samples.shape}, channels={actual_channels}")
        
        if is_cosmos and actual_channels != expected_channels:
            raise ValueError(
                f"\n[Z-Sampling] Channel mismatch!\n"
                f"  Model expects: {expected_channels} channels\n"
                f"  Latent has: {actual_channels} channels\n"
                f"  SOLUTION: Use appropriate EmptyLatent node for your model.\n"
            )
        
        converted = False
        if is_cosmos and original_ndim == 4:
            samples = samples.unsqueeze(2)
            if noise_mask is not None and noise_mask.dim() == 4:
                noise_mask = noise_mask.unsqueeze(2)
            converted = True
            logger.info(f"Converted to 5D: {samples.shape}")
        
        return samples, noise_mask, converted, original_ndim
    
    def _calculate_sigmas(
        self,
        model: Any,
        scheduler: str,
        steps: int,
        denoise: float
    ) -> Tensor:
        """Calculate sigma schedule with denoise support."""
        model_sampling = model.get_model_object("model_sampling")
        
        if denoise >= 0.9999:
            return comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        elif denoise <= 0.0001:
            return torch.FloatTensor([])
        else:
            total_steps = int(steps / denoise)
            sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)
            return sigmas[-(steps + 1):]
    
    def sample(
        self,
        model: Any,
        positive: list,
        negative: list,
        latent_image: dict,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        z_settings: Optional[dict] = None,
        preview_every: int = 1,
        show_zigzag_phases: bool = False,
        preview_method: str = "auto",
    ) -> tuple[dict]:
        """Execute Z-Sampling with live latent2rgb previews."""
        
        # Parse config
        config = ZSamplingConfig(gamma_1=cfg, gamma_2=0.0, lambda_step=-1, t_max=1)
        if z_settings is not None:
            config = ZSamplingConfig(**z_settings)
        
        effective_lambda = config.resolve_lambda_step(steps)
        
        logger.info(f"Z-Sampling: gamma_1={config.gamma_1}, gamma_2={config.gamma_2}, "
                   f"lambda={effective_lambda}, t_max={config.t_max}")
        logger.info(f"Preview: method={preview_method}, every={preview_every}, zigzag={show_zigzag_phases}")
        
        # Detect model type
        is_cosmos, expected_channels = detect_model_type(model)
        logger.info(f"Model: cosmos={is_cosmos}, channels={expected_channels}")
        
        # Prepare latent
        samples, noise_mask, converted_5d, original_ndim = self._prepare_latent(
            latent_image, is_cosmos, expected_channels
        )
        
        # Calculate sigmas
        sigmas = self._calculate_sigmas(model, scheduler, steps, denoise)
        
        if len(sigmas) == 0:
            return ({"samples": samples},)
        
        device = comfy.model_management.get_torch_device()
        sigmas = sigmas.to(device)
        
        # Prepare noise
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(samples, seed, batch_inds)
        
        # Create preview callback with latent2rgb support
        preview_callback = ZSamplingPreviewCallback(
            model=model,
            steps=steps,
            t_max=config.t_max,
            latent_channels=expected_channels,
            preview_every=preview_every,
            show_zigzag_phases=show_zigzag_phases,
            preview_method=preview_method,
            max_preview_size=512
        )
        
        # Create guider and sampler
        guider = ZSamplingGuider(model, cfg=config.gamma_1)
        guider.set_conds(positive, negative)
        
        z_kernel = ZSamplerKernel(
            gamma_1=config.gamma_1,
            gamma_2=config.gamma_2,
            lambda_step=effective_lambda,
            t_max=config.t_max,
            base_sampler=sampler_name,
            preview_callback=preview_callback
        )
        
        # Execute sampling
        output = guider.sample(
            noise,
            samples,
            z_kernel,
            sigmas,
            denoise_mask=noise_mask,
            callback=None,
            disable_pbar=True,  # We handle our own progress bar
            seed=seed
        )
        
        logger.info(f"Output shape: {output.shape}")
        
        # Restore original dimensionality
        if converted_5d and output.dim() == 5 and output.shape[2] == 1:
            output = output.squeeze(2)
            logger.info(f"Restored to 4D: {output.shape}")
        
        # Build output
        result = {"samples": output.cpu()}
        for key, value in latent_image.items():
            if key not in result:
                result[key] = value
        
        logger.info("Z-Sampling complete!")
        return (result,)


# =============================================================================
# Standalone Latent Preview Node
# =============================================================================

class ZPreviewLatent:
    """
    Preview any latent using latent2rgb (no TAESD required).
    
    Passes through latent unchanged - can be used as a preview reroute.
    """
    
    CATEGORY = "sampling/z-sampling"
    FUNCTION = "preview"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_NODE = True
    DESCRIPTION = "Preview latent using latent2rgb. No TAESD required."
    
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "model_type": (["auto", "sd", "sdxl", "cosmos"], {
                    "default": "auto",
                    "tooltip": "Model type for correct RGB factors"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    def preview(
        self,
        latent: dict,
        model_type: str = "auto",
        prompt: Any = None,
        extra_pnginfo: Any = None
    ) -> dict:
        """Preview the latent and pass it through."""
        import os
        
        samples = latent["samples"]
        channels = samples.shape[1]
        
        # Create appropriate previewer
        previewer = get_latent2rgb_previewer(
            latent_channels=channels,
            model_type=model_type,
            max_preview_size=512
        )
        
        # Handle 5D video latents - preview middle frame
        if samples.dim() == 5:
            mid_frame = samples.shape[2] // 2
            samples_2d = samples[:, :, mid_frame, :, :]
        else:
            samples_2d = samples
        
        # Decode to preview
        preview_img = previewer.decode_latent_to_preview(samples_2d[:1])
        
        # Save to temp directory
        output_dir = folder_paths.get_temp_directory()
        filename = f"zpreview_{hash(samples.data_ptr()) % 100000:05d}.png"
        filepath = os.path.join(output_dir, filename)
        preview_img.save(filepath)
        
        return {
            "ui": {
                "images": [{
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp"
                }]
            },
            "result": (latent,)
        }


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "ZSamplingSettings": ZSamplingSettings,
    "ZSampler": ZSampler,
    "ZPreviewLatent": ZPreviewLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZSamplingSettings": "Z-Sampling Settings ⚡",
    "ZSampler": "Z-Sampler (Zigzag + Preview) ⚡",
    "ZPreviewLatent": "Z-Preview Latent (RGB) 👁️",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]