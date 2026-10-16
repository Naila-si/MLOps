import base64
from datetime import date
import io
import json
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import AsyncOpenAI
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# App Configuration & Client Initialization
app = FastAPI(title="Receipt Finance API", version="1.0.0")

client = AsyncOpenAI(
    base_url=os.getenv("MODEL_BASE_URL", "http://127.0.0.1:1234/v1"),
    api_key=os.getenv("MODEL_API_KEY", "lm-studio"),
    timeout=120.0,
)

MODEL_NAME = os.getenv("MODEL_NAME", "")
MAX_BYTES = int(os.getenv("MAX_FILE_MB", "8")) * 1024 * 1024


# Schemas
class ReceiptItem(BaseModel):
    name: str
    quantity: float
    unit_price: Optional[float] = None
    subtotal: Optional[float] = None


class ReceiptData(BaseModel):
    merchant: Optional[str] = None
    transaction_date: Optional[date] = None
    currency: str = "IDR"
    items: list[ReceiptItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None


class ReceiptResponse(BaseModel):
    success: bool
    data: ReceiptData
    metadata: dict


# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/receipts/extract", response_model=ReceiptResponse)
async def extract_receipt(file: UploadFile = File(...)):
    if not MODEL_NAME:
        raise HTTPException(500, "MODEL_NAME belum dikonfigurasi")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Format gambar tidak didukung")

    raw = await file.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(413, "Gambar kosong atau melebihi batas ukuran")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(415, "Format gambar tidak didukung")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Berkas bukan gambar yang valid")

    mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }[image.format]

    encoded = base64.b64encode(raw).decode("ascii")
    image_url = f"data:{mime};base64,{encoded}"

    prompt = """
Ekstrak data nota keuangan dari gambar.
Kembalikan HANYA satu objek JSON murni tanpa markdown, tanpa teks pembuka atau penutup.
Struktur JSON yang harus dikembalikan:
{
  "merchant": string atau null,
  "transaction_date": "YYYY-MM-DD" atau null,
  "currency": "IDR",
  "items": [
    {
      "name": string,
      "quantity": float/number,
      "unit_price": float/number atau null,
      "subtotal": float/number atau null
    }
  ],
  "subtotal": float/number atau null,
  "tax": float/number atau null,
  "discount": float/number atau null,
  "total": float/number atau null,
  "payment_method": string atau null
}

Gunakan angka murni untuk nominal (tanpa simbol mata uang seperti Rp), tanggal wajib format YYYY-MM-DD, dan null jika data tidak ditemukan.
Jangan mengarang informasi.
"""

    try:
        result = await client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
        )
        content = result.choices[0].message.content
        if not content:
            raise ValueError("Output model kosong")

        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise ValueError("JSON tidak ditemukan")

        payload = json.loads(content[start : end + 1])
        data = ReceiptData.model_validate(payload)

        return ReceiptResponse(
            success=True,
            data=data,
            metadata={
                "model": MODEL_NAME,
                "processing_status": "success",
            },
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            502, "Inferensi gagal atau output model tidak valid"
        )
    finally:
        await file.close()
