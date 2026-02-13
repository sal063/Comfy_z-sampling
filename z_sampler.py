import torch
import math
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.sampler_helpers
import comfy.model_patcher


class ZSamplingSettings:
    """Configuration node for Z-Sampling parameters."""
    
    CATEGORY = "sampling/z-sampling"
    FUNCTION = "get_settings"
    RETURN_TYPES = ("ZSETTINGS",)
    RETURN_NAMES = ("z_settings",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gamma_1": ("FLOAT", {
                    "default": 5.5, 
                    "min": 1.0, 
                    "max": 20.0, 
                    "step": 0.1,
                    "tooltip": "CFG scale for denoising steps (strong guidance)"
                }),
                "gamma_2": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "CFG scale for inversion steps (weak/no guidance)"
                }),
                "lambda_step": ("INT", {
                    "default": -1, 
                    "min": -1, 
                    "max": 200,
                    "tooltip": "Steps with zigzag (-1 = auto: steps-1)"
                }),
                "t_max": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 5,
                    "tooltip": "Zigzag rounds per step (more = slower but better)"
                }),
            },
        }
    
    def get_settings(self, gamma_1, gamma_2, lambda_step, t_max):
        return ({
            "gamma_1": gamma_1,
            "gamma_2": gamma_2,
            "lambda_step": lambda_step,
            "t_max": t_max,
        },)


def is_cosmos_model(model):
    """Check if model is Cosmos/Anima based (requires 5D latents with 16 channels)."""
    try:
        model_obj = model.model if hasattr(model, 'model') else model
        model_class = model_obj.__class__.__name__
        if model_class in ["Anima", "CosmosPredict2", "CosmosVideo"]:
            return True
        
        diffusion_model = getattr(model_obj, 'diffusion_model', None)
        if diffusion_model is not None:
            dm_class = diffusion_model.__class__.__name__
            if dm_class in ["MiniTrainDIT", "Anima", "GeneralDIT"]:
                return True
            if hasattr(diffusion_model, 'x_embedder') or hasattr(diffusion_model, 'patch_temporal'):
                return True
    except:
        pass
    return False


def get_model_latent_channels(model):
    """Get expected latent channel count from model."""
    try:
        model_obj = model.model if hasattr(model, 'model') else model
        
        if hasattr(model_obj, 'model_config'):
            config = model_obj.model_config
            if hasattr(config, 'latent_format'):
                lf = config.latent_format
                if hasattr(lf, 'latent_channels'):
                    return lf.latent_channels
        
        diffusion_model = getattr(model_obj, 'diffusion_model', None)
        if diffusion_model is not None:
            if hasattr(diffusion_model, 'in_channels'):
                return diffusion_model.in_channels
                
        if is_cosmos_model(model):
            return 16
            
    except:
        pass
    return 4


class ZSamplingGuider(comfy.samplers.CFGGuider):
    """Extended CFGGuider that supports Z-Sampling with variable CFG."""
    
    def __init__(self, model_patcher, gamma_1, gamma_2, lambda_step, t_max):
        super().__init__(model_patcher)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.lambda_step = lambda_step
        self.t_max = t_max
        self.current_cfg = gamma_1
        
    def set_current_cfg(self, cfg):
        self.current_cfg = cfg
        
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return comfy.samplers.sampling_function(
            self.inner_model, x, timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.current_cfg,
            model_options=model_options,
            seed=seed
        )


class ZSampler:
    """Z-Sampler: Full Zigzag Diffusion Sampling implementation for ComfyUI."""
    
    CATEGORY = "sampling/z-sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "z_settings": ("ZSETTINGS",),
            }
        }
    
    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        z_settings=None,
    ):
        # Parse Z-Sampling settings
        gamma_1 = cfg
        gamma_2 = 0.0
        t_max = 1
        lambda_step = steps - 1
        
        if z_settings is not None:
            gamma_1 = z_settings.get("gamma_1", cfg)
            gamma_2 = z_settings.get("gamma_2", 0.0)
            t_max = z_settings.get("t_max", 1)
            ls = z_settings.get("lambda_step", -1)
            lambda_step = ls if ls >= 0 else steps - 1
        
        lambda_step = min(lambda_step, steps - 1)
        
        print(f"[Z-Sampling] Configuration:")
        print(f"  gamma_1 (denoise CFG): {gamma_1}")
        print(f"  gamma_2 (invert CFG): {gamma_2}")
        print(f"  lambda_step: {lambda_step}")
        print(f"  t_max (zigzag rounds): {t_max}")
        print(f"  steps: {steps}, scheduler: {scheduler}, sampler: {sampler_name}")
        
        # Detect Cosmos/Anima model
        needs_cosmos = is_cosmos_model(model)
        expected_channels = get_model_latent_channels(model)
        print(f"[Z-Sampling] Cosmos/Anima model: {needs_cosmos}, expected channels: {expected_channels}")
        
        # Prepare latent
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        noise_mask = latent.get("noise_mask", None)
        
        # Store original shape info
        original_ndim = latent_samples.dim()
        actual_channels = latent_samples.shape[1]
        print(f"[Z-Sampling] Input latent shape: {latent_samples.shape}, channels: {actual_channels}")
        
        # Check channel mismatch for Cosmos/Anima
        if needs_cosmos and actual_channels != expected_channels:
            error_msg = (
                f"\n[Z-Sampling ERROR] Channel mismatch!\n"
                f"  Model expects: {expected_channels} channels (Cosmos/Anima format)\n"
                f"  Latent has: {actual_channels} channels\n\n"
                f"  SOLUTION: Use 'EmptyCosmosLatentVideo' node instead of 'EmptyLatentImage'\n"
                f"  Or use an image encoded with the Cosmos VAE.\n"
            )
            print(error_msg)
            raise ValueError(error_msg)
        
        # Convert to 5D if needed for Cosmos/Anima (but input was 4D)
        converted_to_5d = False
        if needs_cosmos and original_ndim == 4:
            latent_samples = latent_samples.unsqueeze(2)
            if noise_mask is not None and noise_mask.dim() == 4:
                noise_mask = noise_mask.unsqueeze(2)
            converted_to_5d = True
            print(f"[Z-Sampling] Converted to 5D: {latent_samples.shape}")
        
        device = comfy.model_management.get_torch_device()
        
        # Prepare noise
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        
        # Calculate sigmas
        model_sampling = model.get_model_object("model_sampling")
        
        if denoise is None or denoise > 0.9999:
            sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        elif denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        else:
            new_steps = int(steps / denoise)
            sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, new_steps)
            sigmas = sigmas[-(steps + 1):]
        
        if len(sigmas) == 0:
            latent["samples"] = latent_samples
            return (latent,)
        
        print(f"[Z-Sampling] Starting sampling with {len(sigmas)-1} steps")
        
        # Create Z-Sampling guider
        guider = ZSamplingGuider(model, gamma_1, gamma_2, lambda_step, t_max)
        guider.set_conds(positive, negative)
        
        # Create custom sampler
        z_sampler = ZSamplerFunction(gamma_1, gamma_2, lambda_step, t_max, sampler_name)
        
        # Run sampling
        samples = guider.sample(
            noise, 
            latent_samples, 
            z_sampler, 
            sigmas.to(device),
            denoise_mask=noise_mask,
            callback=None,
            disable_pbar=False,
            seed=seed
        )
        
        print(f"[Z-Sampling] Output shape: {samples.shape}")
        
        # IMPORTANT: Keep output in same format as input for VAE compatibility
        # If input was 5D, keep 5D. If input was 4D but we converted, restore to 4D
        if converted_to_5d and samples.dim() == 5 and samples.shape[2] == 1:
            samples = samples.squeeze(2)
            print(f"[Z-Sampling] Restored to 4D: {samples.shape}")
        
        # For Cosmos models with native 5D input, keep 5D output
        # The VAE decoder expects the same format as input
        
        latent_out = {"samples": samples.cpu()}
        
        # Preserve any additional keys from input latent
        for key in latent:
            if key not in latent_out:
                latent_out[key] = latent[key]
        
        print("[Z-Sampling] Complete!")
        
        return (latent_out,)


class ZSamplerFunction(comfy.samplers.Sampler):
    """Custom sampler implementing the Z-Sampling zigzag algorithm."""
    
    def __init__(self, gamma_1, gamma_2, lambda_step, t_max, base_sampler_name):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.lambda_step = lambda_step
        self.t_max = t_max
        self.base_sampler_name = base_sampler_name
        
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        """Z-Sampling loop with zigzag denoising-inversion."""
        
        # Scale initial noise
        x = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, 
            self.max_denoise(model_wrap, sigmas)
        )
        
        extra_args = extra_args.copy() if extra_args else {}
        seed = extra_args.get("seed", 42)
        model_options = extra_args.get("model_options", {})
        
        total_steps = len(sigmas) - 1
        
        for i in range(total_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            batch_size = x.shape[0]
            sigma_expanded = sigma.view(1).expand(batch_size)
            
            # === STEP 1: Standard denoising with gamma_1 ===
            if hasattr(model_wrap, 'set_current_cfg'):
                model_wrap.set_current_cfg(self.gamma_1)
            
            denoised = model_wrap.outer_predict_noise(x, sigma_expanded, model_options=model_options, seed=seed)
            
            if denoise_mask is not None:
                denoised = denoised * denoise_mask + x * (1 - denoise_mask)
            
            dt = sigma_next - sigma
            if sigma > 0:
                d = (x - denoised) / sigma
                x_next = x + d * dt
            else:
                x_next = denoised
            
            # === STEP 2: Zigzag optimization ===
            if i < self.lambda_step and sigma_next > 0 and sigma > 0:
                for t_round in range(self.t_max):
                    # --- ZAG: Inversion with gamma_2 ---
                    if hasattr(model_wrap, 'set_current_cfg'):
                        model_wrap.set_current_cfg(self.gamma_2)
                    
                    denoised_inv = model_wrap.outer_predict_noise(x_next, sigma_expanded, model_options=model_options, seed=seed)
                    
                    if denoise_mask is not None:
                        denoised_inv = denoised_inv * denoise_mask + x_next * (1 - denoise_mask)
                    
                    d_inv = (x_next - denoised_inv) / sigma
                    x_inv = x_next - d_inv * dt
                    
                    # --- ZIG: Denoise with gamma_1 ---
                    if hasattr(model_wrap, 'set_current_cfg'):
                        model_wrap.set_current_cfg(self.gamma_1)
                    
                    denoised_zig = model_wrap.outer_predict_noise(x_inv, sigma_expanded, model_options=model_options, seed=seed)
                    
                    if denoise_mask is not None:
                        denoised_zig = denoised_zig * denoise_mask + x_inv * (1 - denoise_mask)
                    
                    d_zig = (x_inv - denoised_zig) / sigma
                    x_next = x_inv + d_zig * dt
            
            x = x_next
            
            if callback is not None:
                callback({"i": i, "denoised": denoised, "x": x})
            
            if (i + 1) % 10 == 0 or i == total_steps - 1:
                print(f"[Z-Sampling] Step {i+1}/{total_steps}")
        
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x)
        return samples


# Node registration
NODE_CLASS_MAPPINGS = {
    "ZSamplingSettings": ZSamplingSettings,
    "ZSampler": ZSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZSamplingSettings": "Z-Sampling Settings",
    "ZSampler": "Z-Sampler (Zigzag Diffusion)",
}