import base64, json, mimetypes, os, re
from pathlib import Path
from openai import OpenAI

IMAGE = Path("data/raw/nota-sample.jpg")
MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen/qwen2.5-vl-7b")

client = OpenAI(base_url="http://172.24.192.1:1234/v1", api_key="lm-studio")

mime = mimetypes.guess_type(IMAGE)[0] or "image/jpg"
b64 = base64.b64encode(IMAGE.read_bytes()).decode("utf-8")

prompt = (
    "Baca nota ini. Ekstrak merchant, tanggal, item, subtotal, pajak, dan total. "
    "Keluarkan JSON valid saja, tanpa penjelasan. "
    "Jika pajak tidak terlihat, isi 0. Jangan mengarang."
)

resp = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ],
)

text = resp.choices[0].message.content.strip()
text = re.sub(r"^(?:json)?\s*|\s*$", "", text)
try:
    result = json.loads(text)
except json.JSONDecodeError:
    print("Jawaban model bukan JSON valid:")
    print(text)
    raise SystemExit(1)

Path("reports").mkdir(exist_ok=True)
Path("reports/receipt.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(result, indent=2, ensure_ascii=False))
