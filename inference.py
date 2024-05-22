import os.path
import pdb

import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL, ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetPipeline
from PIL import Image
import argparse

from garment_adapter.garment_diffusion import ClothAdapter
from pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='oms diffusion')
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--enable_cloth_guidance', action="store_true")
    parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
    parser.add_argument('--output_path', type=str, default="./output_img")
    parser.add_argument('--person_path', type=str, required=False)

    args = parser.parse_args()

    device = "cuda"
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cloth_image = Image.open(args.cloth_path).convert("RGB")
    control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    if args.enable_cloth_guidance:
        pipe = OmsDiffusionControlNetPipeline.from_pretrained(args.pipe_path, vae=vae, controlnet=control_net_openpose, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(args.pipe_path, vae=vae, controlnet=control_net_openpose, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    full_net = ClothAdapter(pipe, args.model_path, device, args.enable_cloth_guidance, False, person_path=args.person_path)
    images = full_net.generate(cloth_image)
    for i, image in enumerate(images[0]):
        image.save(os.path.join(output_path, "out_" + str(i) + ".png"))
