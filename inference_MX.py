import json
from pathlib import Path
from PIL import Image

import torch
from sconf import Config
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

from base.dataset import read_font, render
from base.utils import save_tensor_to_image, load_reference
from MX.models import Generator
from inference import infer_MX

weight_path = "MonoGlyph_60000.pth"
n_experts = 6

cfg = Config("cfgs/MX/default.yaml")

g_kwargs = cfg.get('g_args', {})
gen = Generator(n_experts=n_experts, n_emb=2).cuda().eval()
weight = torch.load(weight_path)
if "generator_ema" in weight:
    weight = weight["generator_ema"]
gen.load_state_dict(weight, strict=False)
gen.load_state_dict(weight)

ref_path = "font_data"
extension = "png"
ref_chars = "가나다라마사아자차카타파하"

ref_dict, load_img = load_reference(ref_path, extension, ref_chars)

kor_json_path = "kor_string.json"

with open(kor_json_path, "r", encoding="utf-8") as f:
    gen_chars = json.load(f)

if not isinstance(gen_chars, str):
    raise ValueError('kor_string.json은 "가나다라" 같은 단일 JSON 문자열이어야 합니다.')

save_dir = "./result"
source_path = "data/kor/source.ttf"
source_ext = "ttf"
batch_size = 16

infer_MX(gen, save_dir, source_path, source_ext, gen_chars, ref_dict, load_img, batch_size)