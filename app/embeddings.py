import os
from typing import List

from openai import OpenAI
import torch
import open_clip
from PIL import Image


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# NOTE: Check and adjust TEXT_EMBED_DIM to match the actual dimension of text-embedding-3-large.
TEXT_EMBED_DIM = 3072


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=device,
)

# This depends on the specific CLIP backbone; for ViT-B-32 it's typically 512 or 768.
CLIP_EMBED_DIM = clip_model.text_projection.shape[1]


def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return resp.data[0].embedding


def embed_image(path: str) -> List[float]:
    image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().tolist()

