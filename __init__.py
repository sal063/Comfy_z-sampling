from .z_sampler import ZSampler, ZSamplingSettings

NODE_CLASS_MAPPINGS = {
    "ZSampler": ZSampler,
    "ZSamplingSettings": ZSamplingSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZSampler": "Z-Sampler (Zigzag)",
    "ZSamplingSettings": "Z-Sampling Settings",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]