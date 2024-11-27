import os
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import json

# Paths
input_folder = "input"
output_folder_images = "D:/repos/image-vison/e-invoice/data/upload/images"
output_folder_text = "output/text"
output_json = "config/annotations.json"

# Ensure output folders exist
os.makedirs(output_folder_images, exist_ok=True)
os.makedirs(output_folder_text, exist_ok=True)

# Process all PDFs in the input folder
def process_all_pdfs(input_folder, output_folder_images, output_folder_text, output_json):
    annotations = []  # Collect all annotations for JSON output

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_name}...")

            # Extract images
            images = convert_from_path(pdf_path, dpi=300)
            pdf_base_name = os.path.splitext(file_name)[0]

            for i, image in enumerate(images):
                image_name = f"{pdf_base_name}_page_{i + 1}.png"
                image_path = os.path.join(output_folder_images, image_name)
                image.save(image_path, "PNG")
                print(f"Saved image: {image_path}")

            # Extract text
            doc = fitz.open(pdf_path)
            text_output = []
            for page_number, page in enumerate(doc):
                text = page.get_text()
                text_output.append({"page": page_number + 1, "text": text})

            # Save text to file
            text_file_path = os.path.join(output_folder_text, f"{pdf_base_name}.txt")
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                for text_data in text_output:
                    text_file.write(f"Page {text_data['page']}:\n{text_data['text']}\n")
                print(f"Saved text: {text_file_path}")

            # Add annotations for JSON
            annotations.append({
                "data": {
                    "document": f"{output_folder_images}/{pdf_base_name}_page_{i + 1}.png" for i in range(len(images))
                }
            })

    # Save annotations to JSON
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(annotations, json_file, ensure_ascii=False, indent=4)
        print(f"Annotations saved to {output_json}")

# Run the script
process_all_pdfs(input_folder, output_folder_images, output_folder_text, output_json)
