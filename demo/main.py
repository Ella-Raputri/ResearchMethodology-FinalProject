import argparse
import os

from utils.ocr import scan_image
from utils.yolo_cleaning import yolo_clean
from utils.symspell import symspell_clean
from utils.llm import llm_clean


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

    if(use_yolo):
        final_text = yolo_clean(src)

    if (not text_clean_choice): return final_text


    if (text_clean_choice == 'symspell'):
        final_text = symspell_clean(final_text)
    elif (text_clean_choice == 'llm'):
        final_text = llm_clean(final_text)
    else:
        print('Invalid choice')
        raise ValueError("Invalid text_clean_choice")

    return final_text

def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR + post-processing demo (YOLO / SymSpell / LLM)"
    )

    parser.add_argument("src", type=str, help="Path to input image file")
    parser.add_argument("--yolo", action="store_true", help="Enable YOLO-based OCR cleaning")
    parser.add_argument("--text-clean", choices=["symspell", "llm"], default=None, help="Text cleaning method")
    parser.add_argument("--save-output", type=str,default=None,  help="Path to save final output text")
    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.exists(args.src), f"File not found: {args.src}"
    assert args.src.lower().endswith(
        (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    ), "Input must be an image file"

    if args.text_clean == "llm":
        assert callable(llm_clean), "llm_clean function is not callable"

    result = final_clean_text(
        src=args.src,
        use_yolo=args.yolo,
        text_clean_choice=args.text_clean
    )

    print("\n[FINAL OUTPUT]")
    print(result)

    if args.save_output:
        output_dir = os.path.dirname(args.save_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.save_output, "w", encoding="utf-8") as f:
            f.write(result)

        print(f"\n[OUTPUT SAVED TO]")
        print(args.save_output)


if __name__ == "__main__":
    main()