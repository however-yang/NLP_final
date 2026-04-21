from __future__ import annotations


def ensure_image_placeholders_in_text(processor, text: str, *, num_images: int = 1) -> str:
    """
    Qwen3-VL（及同类）要求：传入 images 时，text 中必须包含与图像数量一致的 image_token
    （如 <|image_pad|>），processor 再将其展开为与视觉特征对齐的占位长度。
    否则会出现：ValueError: Image features and image tokens do not match.
    """
    token = getattr(processor, "image_token", None)
    if token is None or num_images <= 0:
        return text
    present = text.count(token)
    if present >= num_images:
        return text
    missing = num_images - present
    prefix = (token + "\n") * missing
    return prefix + text
