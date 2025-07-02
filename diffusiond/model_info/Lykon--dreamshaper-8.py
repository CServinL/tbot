# ModelInfo structure for DreamShaper8_pruned for use in diffusiond

MODEL_INFO = {
    "name": "DreamShaper8_pruned",
    "model_id": "Lykon/dreamshaper-8",
    "model_hash": "879db523c3",
    "version": "v1.4.1",
    "prompt_example": (
        "(masterpiece), (extremely intricate:1.3), (realistic), portrait of a girl, the most beautiful in the world, "
        "(medieval armor), metal reflections, upper body, outdoors, intense sunlight, far away castle, professional photograph of a stunning woman detailed, "
        "sharp focus, dramatic, award winning, cinematic lighting, octane render  unreal engine,  volumetrics dtx, "
        "(film grain, blurry background, blurry foreground, bokeh, depth of field, sunset, motion blur:1.3), chainmail"
    ),
    "negative_prompt_example": "BadDream, (UnrealisticDream:1.3)",
    "default_steps": 30,
    "default_cfg_scale": 9,
    "default_sampler": "DPM++ SDE Karras",
    "default_seed": 5775713,
    "default_size": "512x832",
    "hires_steps": 20,
    "hires_upscale": 2.2,
    "hires_upscaler": "8x_NMKD-Superscale_150000_G",
    "adetailer": {
        "model": "face_yolov8n.pt",
        "prompt": "photo of a blonde girl, (film grain)",
        "negative_prompt": "BadDream",
        "version": "23.6.1.post0",
        "mask_blur": 4,
        "model_2nd": "hand_yolov8n.pt",
        "confidence": 0.3,
        "dilate_erode": 4,
        "mask_blur_2nd": 4,
        "confidence_2nd": 0.3,
        "inpaint_padding": 0,
        "denoising_strength": 0.46,
        "dilate_erode_2nd": 4,
        "denoising_strength_2nd": 0.3,
        "inpaint_only_masked": True,
        "inpaint_padding_2nd": 32,
        "negative_prompt_2nd": "BadDream",
        "inpaint_only_masked_2nd": True,
    },
    "clip_skip": 2,
}
