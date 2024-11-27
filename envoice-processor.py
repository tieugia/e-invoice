import os
import pytesseract
from pdf2image import convert_from_path
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
import pandas as pd

# Load mô hình và tokenizer
model_path = "fine_tuned_layoutlmv3"
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_path)

# Đường dẫn Tesseract (nếu cần)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Trích xuất văn bản và bounding boxes từ PDF
def extract_text_and_bbox(pdf_path):
    images = convert_from_path(pdf_path)
    text_lines = []
    bboxes = []

    for img in images:
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            if ocr_data['text'][i].strip():
                text_lines.append(ocr_data['text'][i])
                bbox = [
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['left'][i] + ocr_data['width'][i],
                    ocr_data['top'][i] + ocr_data['height'][i],
                ]
                bboxes.append(bbox)

    return text_lines, bboxes

# Chuẩn bị dữ liệu đầu vào
def prepare_inputs(text_lines, bboxes):
    inputs = tokenizer(
        text_lines,
        boxes=bboxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
    )
    return inputs

# Dự đoán nhãn cho từng đoạn văn bản
def classify_pdf(pdf_path):
    text_lines, bboxes = extract_text_and_bbox(pdf_path)
    inputs = prepare_inputs(text_lines, bboxes)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.argmax(dim=-1).squeeze().tolist()

    # Kết hợp nhãn với văn bản
    results = [{"text": text, "label": model.config.id2label[pred]} for text, pred in zip(text_lines, predictions)]
    return results

# Lưu kết quả vào CSV
def save_results_to_csv(results, pdf_path, output_csv):
    rows = []
    for result in results:
        rows.append({
            "file_name": os.path.basename(pdf_path),
            "text": result["text"],
            "label": result["label"]
        })

    # Ghi dữ liệu vào file CSV
    df = pd.DataFrame(rows)
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8-sig')  # Ghi thêm nếu file tồn tại
    else:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')  # Ghi mới nếu file chưa tồn tại
    print(f"Results saved to {output_csv}")

# Chạy dự đoán trên tất cả file PDF trong thư mục
def process_all_pdfs(input_folder, output_csv):
    pdf_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        results = classify_pdf(pdf_file)
        save_results_to_csv(results, pdf_file, output_csv)

# Thực thi chương trình
if __name__ == "__main__":
    input_folder = "input"  # Thư mục chứa các file PDF
    output_csv = "output/predicted_data.csv"  # File CSV đầu ra
    process_all_pdfs(input_folder, output_csv)
