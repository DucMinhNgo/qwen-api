import os
import warnings

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated! Use `dtype` instead!",
    category=FutureWarning
)

from typing import Union
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    use_fast=True,
    min_pixels=256 * 28 * 28,
    max_pixels=384 * 28 * 28,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.eval()


def refine_ocr(image: Union[str, Image.Image], ocr_text: str) -> str:
    if isinstance(image, str):
        try:
            image = Image.open(image).convert("RGB")
        except Exception as e:
            raise ValueError(f"Không mở được ảnh: {image}\nLỗi: {e}")

    if not isinstance(image, Image.Image):
        raise ValueError("image phải là PIL.Image hoặc đường dẫn file hợp lệ")

    max_side = 672
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    prompt = f"""Bạn là chuyên gia sửa lỗi OCR chính xác tuyệt đối.

QUY TẮC BẮT BUỘC:
1. CHỈ sửa ký tự/từ sai rõ ràng dựa trên ảnh
2. KHÔNG thêm bất kỳ nội dung nào không thấy rõ
3. KHÔNG bỏ bớt nội dung hiện có
4. Giữ nguyên định dạng, xuống dòng sát bản gốc
5. Trả về DUY NHẤT văn bản đã sửa - không giải thích

OCR TEXT:
{ocr_text}

Văn bản đã sửa:"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to("cpu")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_length = inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_length:]

    corrected = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()

    return corrected


# ─── Test nhanh ─────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     test_image_path = "duong_dan_anh_test.jpg"  # Thay bằng ảnh thật

#     raw_ocr = """Thls ls an exampl3 wlth s0me typ1cal OCR err0rs.
# C0mpany: ABC C0rp0rat!on"""

#     print("OCR gốc:")
#     print(raw_ocr)
#     print("\nĐang sửa trên CPU... (có thể mất vài phút)\n")

#     try:
#         result = refine_ocr(test_image_path, raw_ocr)
#         print("Kết quả sau khi sửa:")
#         print(result)
#     except Exception as e:
#         print("Lỗi:", str(e))
