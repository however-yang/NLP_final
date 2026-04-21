from __future__ import annotations

from contextlib import nullcontext

from text_rich_mllm.models.vision_prompt import ensure_image_placeholders_in_text


def _move_to_device(payload, device):
    moved = {}
    for key, value in payload.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def strip_prompt_from_output(decoded: str, prompt: str) -> str:
    decoded = decoded.strip()
    prompt = prompt.strip()
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded


def open_image_as_rgb(image_path: str):
    """与推理一致：调色板透明 PNG 先转 RGBA 再 RGB，避免训练 collator 读图与推理不一致。"""
    from PIL import Image

    img = Image.open(image_path)
    if getattr(img, "mode", None) == "P" and "transparency" in getattr(img, "info", {}):
        img = img.convert("RGBA")
    return img.convert("RGB")


def take_answer_tail_after_marker(text: str) -> str:
    """
    Qwen-VL 等常在解码里复述整段 user prompt；若无法用前缀切掉，则按最后一次「Answer:」截取，
    否则 DocVQA/ChartQA/MCQ 的评测会把整段长文本拿去算 ANLS，得分接近全 0。
    """
    t = text.strip()
    for marker in ("\nAnswer:", "\n答案:", "Answer:", "答案:", "答："):
        if marker in t:
            tail = t.rsplit(marker, 1)[-1].strip()
            if tail:
                return tail
    return t


def run_generation(model, processor, image_path: str, prompt: str, generation_config: dict) -> str:
    image = open_image_as_rgb(image_path)
    prompt_for_model = ensure_image_placeholders_in_text(processor, prompt, num_images=1)
    inputs = processor(images=image, text=prompt_for_model, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = _move_to_device(inputs, model_device)
    if hasattr(model, "eval"):
        model.eval()
    try:
        import torch

        context = torch.inference_mode()
    except ImportError:
        context = nullcontext()
    with context:
        generated = model.generate(**inputs, **generation_config)
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    out = strip_prompt_from_output(decoded, prompt_for_model)
    return take_answer_tail_after_marker(out)
