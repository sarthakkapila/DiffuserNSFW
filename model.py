from diffusers import ControlNetModel, AutoPipelineForImage2Image
import torch
from diffusers.utils import load_image, make_image_grid

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)


# Usage of ControlNetModel

# Preparing image
url = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.theverge.com%2F2021%2F12%2F5%2F22819328%2Fhumanoid-robot-eerily-lifelike-facial-expressions&psig=AOvVaw03T7glGS3NLsms3DxBr57t&ust=1715873845632000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCNDfiqr-j4YDFQAAAAAdAAAAABAE"
init_image = load_image(url)
init_image = init_image.resize((958, 960)) # resize to depth image dimensions
scribble = load_image("/Users/sarthakkapila/NSFW/skkibidi.jpeg")
make_image_grid([init_image, scribble], rows=1, cols=2)

prompt = "A Humanoid robot eating Butter chicken with Garlic naan in an indian restaurant, 8k"
image_control_net = pipeline(prompt, image=init_image, control_image=scribble).images[0]
make_image_grid([init_image, scribble, image_control_net], rows=1, cols=3)
