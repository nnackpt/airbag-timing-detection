import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer
from threading import Thread

DEFAULT_MAX_NEW_TOKENS = 32
MAX_INPUT_TOKEN_LENGTH = 2048
MODEL_PATH = "./ocr"  # หรือ path เต็ม

class OCR:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device).eval()

    def extract_ms_time(self, image: Image.Image, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        if image is None:
            return "No image provided."

        query = (
            "Read the time (in milliseconds) shown at the top-left corner of the image. "
            "Only return the number, for example: 12345"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKEN_LENGTH
        ).to(self.device)

        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 30,
            "repetition_penalty": 1.1
        }

        Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        output_text = "".join(token for token in streamer).strip()

        return output_text.replace("<|im_end|>", "").strip()


# ===== ตัวอย่างการเรียกใช้ =====
if __name__ == "__main__":
    image_path = r"Output\Screenshots\N1WB-E042D95-AEDYACB25115110423_23_Side\FR1_frame86_N1WB-E042D95-AEDYACB25115110423_23_Side.png"
    image = Image.open(image_path).convert("RGB")

    ocr = OCR()
    result = ocr.extract_ms_time(image)

    print("Time in ms extracted:", result)
