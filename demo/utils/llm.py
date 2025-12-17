import pandas as pd
from google.genai import types
from google import genai
import re, os, json
from dotenv import load_dotenv

load_dotenv("../.env")
API_KEY = os.getenv("GEMINI_API_KEY7")

client = genai.Client(
    api_key=API_KEY
)

def return_prompt(row):
    prompt = (
        "Teks berikut adalah hasil OCR:\n\n"
        f"{row}\n\n"
        "Instruksi:\n"
        "1. Jangan susun ulang teks.\n"
        "2. Perbaiki typo dan kesalahan yang penyebabnya adalah OCR.\n"
        "3. Jangan menambah, mengurangi, atau mengubah informasi apa pun.\n"
        "4. Jangan menambah komentar, penjelasan, atau catatan.\n"
        "5. Output harus berupa teks final saja, tanpa markdown, tanpa format tambahan.\n"
        "6. Jangan melakukan asumsi atau mengisi bagian teks yang hilang.\n"
        "7. Teks output **harus berisi kata-kata yang sama dengan input**, kecuali kata yang memang diperbaiki karena typo.\n"
        "8. Jangan mengubah struktur kalimat secara berlebihan—hanya perbaiki ejaan.\n\n"
        "PERINTAH PENTING:\n"
        "• Jangan membuat kalimat baru.\n"
        "• Jangan menghilangkan kata.\n"
        "• Jangan melakukan halusinasi.\n"
        "• Kembalikan hanya teks yang sudah diperbaiki."
    )
    return prompt


def llm_clean(row):
    try:
        model = "gemini-2.5-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=return_prompt(row)),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        response_text=""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ): response_text += chunk.text

        # print(index)
        # print(response_text)
        
        # with open(f"./final_LLM_res/res_{index}.txt", "w",encoding="utf-8", errors="ignore") as file:
        #     file.write(response_text)
        return response_text

    except Exception as e:
        print(f"Row - Error processing column :", e)