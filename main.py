from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from io import BytesIO

from model import refine_ocr

app = FastAPI(title="OCR Refine API")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/refine")
async def refine(
    image: UploadFile = File(...),
    ocr_text: str = Form(...)
):
    img_bytes = await image.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    refined = refine_ocr(img, ocr_text)

    return {
        "refined_text": refined
    }
