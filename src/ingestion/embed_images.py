from transformers import CLIPModel, CLIPProcessor

from shared.embedding_config import IMAGE_MODEL_NAME


def load_image_model():
    """(CLIP model, processor)."""
    return CLIPModel.from_pretrained(IMAGE_MODEL_NAME), CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)


IMAGE_BATCH_SIZE = 16  # pictures per forward pass


def embed_pil_images(model, processor, images: list, batch_size: int = IMAGE_BATCH_SIZE) -> list:
    """The normalised 512-number embedding of every picture (PIL images), `batch_size` pictures per forward pass,
    in the order given."""
    vectors = []
    for start in range(0, len(images), batch_size):
        batch = [img.convert("RGB") for img in images[start:start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt")
        # model.get_image_features() returns the wrong object (raw vision
        # encoder output, not the projected 512-dim embedding) in this
        # transformers version - call the vision encoder + projection
        # layer directly instead, which is what get_image_features is
        # supposed to do internally.
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        embeddings = model.visual_projection(vision_outputs.pooler_output)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        vectors.extend(embeddings.tolist())
    return vectors


def embed_pil_image(model, processor, img) -> list:
    """The normalised 512-number embedding of one picture (a PIL image)."""
    return embed_pil_images(model, processor, [img])[0]
