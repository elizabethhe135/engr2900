import copy
import pdb
from PIL import Image, ImageOps
import numpy as np
import cv2

import torch
from safetensors import safe_open
from garment_seg.process import load_seg_model, generate_mask
from utils.utils import is_torch2_available, prepare_image, prepare_mask
from diffusers import UNet2DConditionModel

if is_torch2_available():
    from .attention_processor import REFAttnProcessor2_0 as REFAttnProcessor
    from .attention_processor import AttnProcessor2_0 as AttnProcessor
    from .attention_processor import REFAnimateDiffAttnProcessor2_0 as REFAnimateDiffAttnProcessor
else:
    from .attention_processor import REFAttnProcessor, AttnProcessor


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


class ClothAdapter:
    def __init__(self, sd_pipe, ref_path, device, enable_cloth_guidance, use_independent_condition, set_seg_model=True, 
    person_path="./images/p0.png"):
        self.enable_cloth_guidance = enable_cloth_guidance
        self.use_independent_condition = use_independent_condition
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.set_adapter(self.pipe.unet, "write")
        self.person_path = person_path

        ref_unet = copy.deepcopy(sd_pipe.unet)
        if ref_unet.config.in_channels == 9:
            ref_unet.conv_in = torch.nn.Conv2d(4, 320, ref_unet.conv_in.kernel_size, ref_unet.conv_in.stride, ref_unet.conv_in.padding)
            ref_unet.register_to_config(in_channels=4)
        state_dict = {}
        with safe_open(ref_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        ref_unet.load_state_dict(state_dict, strict=False)

        self.ref_unet = ref_unet.to(self.device, dtype=self.pipe.dtype)
        self.set_adapter(self.ref_unet, "read")
        if set_seg_model:
            self.set_seg_model()
        self.attn_store = {}

    def set_seg_model(self, ):
        checkpoint_path = '/home/lizcar/MagicClothing/MagicClothing/cloth_segm.pth'
        self.seg_net = load_seg_model(checkpoint_path, device=self.device)

    def set_adapter(self, unet, type):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name:
                attn_procs[name] = REFAttnProcessor(name=name, type=type)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def generate(
            self,
            cloth_image,
            cloth_mask_image=None,
            prompt=None,
            a_prompt="best quality, high quality",
            num_images_per_prompt=4,
            negative_prompt="bare, monochrome, lowres, bad anatomy, worst quality, low quality",
            seed=-1,
            guidance_scale=2.5,
            cloth_guidance_scale=2.5,
            num_inference_steps=20,
            height=512,
            width=384,
            image = None,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(cloth_image, net=self.seg_net, device=self.device)

        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        if 1:
        # if self.person_path is not None:
            person_image = Image.open(self.person_path).convert("RGB")
            person_mask_image = generate_mask(person_image, net=self.seg_net, device=self.device)
            # person_mask_image = ImageOps.invert(person_mask_image)
            person = prepare_image(person_image, height, width)
            person_mask = prepare_mask(person_mask_image, height, width)
            # person = (person * person_mask).to(self.device, dtype=torch.float16)
            person = person.squeeze(0).cpu()
            if person.is_floating_point():
                person = (person * 255).byte()
            person = person.permute(1, 2, 0).numpy()
            person_image.save('./output_img/person.png')
            cloth_mask_image.save('./output_img/cloth_mask.png')
            inpaint_image = make_inpaint_condition(person_image, person_mask_image)
            # print(image - inpaint_image)
            
            # inpaint_image_pil = Image.fromarray(inpaint_image.squeeze().permute(1,2,0).cpu().numpy())
            inpaint_img_pil = inpaint_image.cpu()
            if inpaint_img_pil.dim() > 3:
                inpaint_img_pil = inpaint_img_pil.squeeze(0)  # Remove batch dimension if necessary

            # Convert to 'uint8' if it's not already
            if inpaint_img_pil.is_floating_point():
                # Normalize and scale to 0-255 if your float tensor is normalized to [0, 1]
                inpaint_img_pil = (inpaint_img_pil * 255).byte()

            # Permute to change the order from (C, H, W) to (H, W, C)
            # if inpaint_img_pil.dim() == 3 and inpaint_img_pil.size(0) == 3:  # Ensure it is 3 channels
            #     inpaint_img_pil = inpaint_img_pil.permute(1, 2, 0)

            # Convert to numpy array
            # inpaint_img_pil_np = Image.fromarray(inpaint_img_pil.numpy())
            # inpaint_img_pil_np.save('./output_img/inpaint.png')
            # inpaint_image = make_inpaint_condition(Image.fromarray(person), Image.fromarray(person_mask.squeeze().cpu().numpy()))
            # tmp_output = inpaint_image.squeeze().permute(1,2,0).cpu().numpy()
            # cv2.imwrite('./output_img/tmp.png', tmp_output)
            # cv2.imwrite('./output_img/person.png', person)
            # pdb.set_trace()
            # cv2.imwrite('./output_img/person_mask.png', person_mask.squeeze().permute(1,2,0).cpu().numpy())

        if prompt is None:
            prompt = "a photography of a model"
        prompt = prompt + ", " + a_prompt
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_null = self.pipe.encode_prompt([""], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = self.pipe.vae.encode(cloth).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})
        if seed == -1:
            seed = None
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        if self.enable_cloth_guidance:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                cloth_guidance_scale=cloth_guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0, "enable_cloth_guidance": self.enable_cloth_guidance, "use_independent_condition": 1},
                image=inpaint_image,
            ).images
        else:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0, "enable_cloth_guidance": self.enable_cloth_guidance, "use_independent_condition": self.use_independent_condition},
                image=inpaint_image,
                # **kwargs,
            ).images

        return images, cloth_mask_image

    def generate_inpainting(
            self,
            cloth_image,
            cloth_mask_image=None,
            num_images_per_prompt=4,
            seed=-1,
            cloth_guidance_scale=2.5,
            num_inference_steps=20,
            height=512,
            width=384,
            **kwargs,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(cloth_image, net=self.seg_net, device=self.device)

        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        with torch.inference_mode():
            prompt_embeds_null = self.pipe.encode_prompt(["a photography of a model"], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = self.pipe.vae.encode(cloth).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds_null,
            cloth_guidance_scale=cloth_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": cloth_guidance_scale > 1.0, "enable_cloth_guidance": False},
            **kwargs,
        ).images

        return images, cloth_mask_image


class ClothAdapter_AnimateDiff:
    def __init__(self, sd_pipe, pipe_path, ref_path, self_ip_path, device, set_seg_model=True):
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.set_ori_adapter(self.pipe.unet)

        ref_unet = UNet2DConditionModel.from_pretrained(pipe_path, subfolder='unet', torch_dtype=sd_pipe.dtype)
        state_dict = {}
        with safe_open(ref_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        ref_unet.load_state_dict(state_dict, strict=False)

        self.ref_unet = ref_unet.to(self.device)
        self.set_ref_adapter(self.ref_unet)
        if set_seg_model:
            self.set_seg_model()
        self.attn_store = {}

        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers_stores = torch.nn.ModuleList([])
        for i in range(len(ip_layers)):
            if isinstance(ip_layers[i], REFAnimateDiffAttnProcessor):
                ip_layers_stores.append(ip_layers[i])
                ip_layers_stores.append(torch.nn.Identity())
        ip_layers_stores.load_state_dict(torch.load(self_ip_path, map_location="cpu"))
        ip_layers_stores.to(self.device)

    def set_seg_model(self, ):
        checkpoint_path = 'checkpoints/cloth_segm.pth'
        self.seg_net = load_seg_model(checkpoint_path, device=self.device)


    def set_ori_adapter(self, unet):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if "attn1" in name and "motion_modules" not in name:
                attn_procs[name] = REFAnimateDiffAttnProcessor(hidden_size=hidden_size, cross_attention_dim=hidden_size,name=name)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def set_ref_adapter(self, unet):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name:
                attn_procs[name] = REFAttnProcessor(name=name, type="read")
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def generate(
            self,
            cloth_image,
            cloth_mask_image=None,
            prompt=None,
            a_prompt="best quality, high quality, masterpiece, bestquality, highlydetailed,",
            num_images_per_prompt=4,
            negative_prompt=None,
            seed=-1,
            guidance_scale=5.,
            cloth_guidance_scale=2.5,
            num_inference_steps=20,
            height=768,
            width=576,
            **kwargs,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(cloth_image, net=self.seg_net, device=self.device)

        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        if prompt is None:
            prompt = "a photography of a model"
        prompt = prompt + ", " + a_prompt
        if negative_prompt is None:
            negative_prompt = "worst quality, low quality"

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_null = self.pipe.encode_prompt([""], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = self.pipe.vae.encode(cloth).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        frames = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            cloth_guidance_scale=cloth_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0},
            **kwargs,
        ).frames

        return frames, cloth_mask_image
