import streamlit as st
import os
import tempfile
from utils.ocr import scan_image
from utils.yolo_cleaning import yolo_clean
from utils.symspell import symspell_clean
from utils.llm import llm_clean
from utils.tts import text_to_speech_bytes


def final_clean_text(src, use_yolo=False, text_clean_choice=None) -> str:
    """
    Demo of OCR and applying the OCR post-processing

    The method first takes an image and apply OCR to it using PyTesseract, then
    we apply OCR post-processing techniques using YOLO and text correction methods.
    User can choose to apply the techniques or not.  
    
    Args:
        src (str): The path of the image file
        use_yolo (bool, optional): Choose only bwteen True or False, specify whether the user wants to use YOLO or not. 
        text_clean_choice (str, optional): Choose only between None, symspell, or llm. Specify whether the user wants to apply text correction or not. 
    
    Example:
        >>> final_clean_text('test.png', True, 'llm')
    """
    final_text, ocr_data = scan_image(src)

    results = {
        "baseline": final_text,
        "yolo": None,
        "text-correction" : None,
        "final": final_text
    }

    if use_yolo:
        final_text = yolo_clean(src)
        results["yolo"] = final_text
        results["final"] = final_text

    if text_clean_choice == "symspell":
        final_text = symspell_clean(final_text)
        results["text-correction"] = final_text
        results["final"] = final_text

    elif text_clean_choice == "llm":
        final_text = llm_clean(final_text)
        results["text-correction"] = final_text
        results["final"] = final_text

    return results

# streamlit
st.set_page_config(
    page_title="OCR Post-Processing Demo",
    layout="wide"
)
st.title("OCR Post-Processing Demo")
st.write("OCR to YOLO to SymSpell / LLM")


st.sidebar.header("Settings")
use_yolo = st.sidebar.checkbox("Use YOLO cleaning", value=False)
text_clean_choice = st.sidebar.radio(
    "Text correction method",
    options=[None, "symspell", "llm"],
    index=0,
    format_func=lambda x: "None" if x is None else x.upper()
)

save_output = st.sidebar.checkbox("Save output to file")
output_filename = st.sidebar.text_input(
    "Output filename",
    value="results/output.txt",
    disabled=not save_output
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "tiff", "bmp"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    assert os.path.exists(img_path), "Temporary image file was not created"

    if st.button("🚀 Run OCR Pipeline"):
        with st.spinner("Processing..."):
            results = final_clean_text(
                src=img_path,
                use_yolo=use_yolo,
                text_clean_choice=text_clean_choice
            )

        st.subheader("📌 Baseline OCR")
        st.text_area(
            "Baseline OCR Output",
            results["baseline"],
            height=200
        )

        if results["yolo"] is not None:
            st.subheader("YOLO Cleaned Text")
            st.text_area(
                "YOLO Output",
                results["yolo"],
                height=200
            )
        
        if results["text-correction"] is not None:
            st.subheader("Typo Corrected Text")
            st.text_area(
                "Text Correction Output",
                results["text-correction"],
                height=200
            )

        st.subheader("✅ Final Output")
        st.text_area(
            "Final Cleaned Text",
            results["final"],
            height=250
        )

        if save_output:
            output_dir = os.path.dirname(output_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(results["final"])

            st.success(f"Output saved to `{output_filename}`")

        st.download_button(
            label="Download Final Text",
            data=results["final"],
            file_name="final_output.txt",
            mime="text/plain"
        )


        st.subheader("[Text to Speech]")
        if st.button("Generate Speech for Final Text"):
            with st.spinner("Generating speech..."):
                audio_bytes = text_to_speech_bytes(results["final"])
                st.session_state["tts_audio"] = audio_bytes
        
        if "tts_audio" in st.session_state:
            st.download_button(
                label="Download Speech",
                data=st.session_state["tts_audio"],
                file_name="final_output.wav",
                mime="audio/wav"
            )
            st.audio(st.session_state["tts_audio"], format="audio/wav")


    # Cleanup temp file
    os.unlink(img_path)
