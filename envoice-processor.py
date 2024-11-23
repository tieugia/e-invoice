import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import pandas as pd

# Danh sách từ đồng nghĩa
synonyms = {
    "invoice_number": ["Số (No.)"],
    "seller_company": ["Đơn vị bán hàng (Seller)"],
    "buyer_company": ["Tên đơn vị (Company's name)", "Tên đơn vị (Company name)"],
    "tax_code": ['Mã số thuế (Tax code)'],
    "total_amount": ["Tổng cộng tiền thanh toán (Total payment)"],
    "date": ["Ngày phát hành", "Ngày xuất hóa đơn", "Date", "Invoice Date", "Issue Date"]
}

# Ánh xạ nhãn trong bảng về nhãn chung
header_mapping = {
    "Name of goods, services": ["Tên hàng hóa, dịch vụ\n(Name of goods, services)",  "Tên hàng hóa, dịch vụ\n(Description)"],
    "Unit": ["Đơn vị tính\n(Unit)"],
    "Quantity": ["Số lượng (Quantity)"],
    "Unit price": ["Đơn giá\n(Unit price)"],
    "Amount": ["Thành tiền\n(Amount)"]
}


# Trích xuất text từ PDF
def extract_text_from_pdf(pdf_path, is_scanned=False):
    if is_scanned:
        images = convert_from_path(pdf_path)  # Chuyển PDF sang hình ảnh
        text = []
        for image in images:
            text.append(pytesseract.image_to_string(image, lang='vie+eng'))  # OCR trên từng ảnh
        return "\n".join(text)
    else:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text.strip())
            return "\n".join(all_text)

# Chuẩn hóa văn bản
def normalize_text(text):
    text = text.lower()  # Chuyển tất cả về chữ thường
    return text

# Trích xuất thông tin từ văn bản
def extract_info(text):
    normalized_text = normalize_text(text)
    extracted_data = {}

    # Lặp qua danh sách từ đồng nghĩa để tìm giá trị
    for key, keywords in synonyms.items():
        for keyword in keywords:
            keyword_pattern = re.escape(keyword.lower()).replace("\\ ", "\\s*")
            regex = rf"{keyword_pattern}:?\s*(.+)"
            match = re.search(regex, normalized_text)
            if match:
                value = match.group(1).strip()
                if "\n" in value:
                    value = value.split("\n")[0].strip()
                extracted_data[key] = value

    # Xử lý trường hợp đặc biệt (ví dụ: số tiền)
    if "total_amount" in extracted_data:
        total_amount = extracted_data["total_amount"].replace(",", "").replace(".", "")
        match = re.search(r"[\d\.]+", total_amount)
        if match:
            extracted_data["total_amount"] = float(match.group(0))

    return extracted_data

# Trích xuất dữ liệu từ bảng trong PDF
def extract_table_from_pdf_with_labels(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        table_rows = []
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                # Dòng đầu tiên là tiêu đề, ánh xạ về nhãn chung
                headers = table[0]
                mapped_headers = {}
                for i, header in enumerate(headers):
                    if header:  # Kiểm tra tiêu đề không rỗng
                        for key, synonyms in header_mapping.items():
                            for synonym in synonyms:
                                # Kiểm tra khớp bằng regex
                                if re.search(rf"{re.escape(synonym)}", header.strip(), re.IGNORECASE):
                                    mapped_headers[i] = key  # Ánh xạ cột về nhãn chung
                                    break

                # Xử lý các dòng dữ liệu
                for row in table[1:]:
                    if any(row):  # Bỏ qua các dòng trống
                        row_data = {mapped_headers[i]: row[i] for i in mapped_headers if row[i]}
                        table_rows.append(row_data)
        return table_rows


# Kết hợp thông tin văn bản và bảng
def process_pdfs_with_table_and_labels(input_folder, output_excel, is_scanned=False):
    all_data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_name}")

            # Trích xuất văn bản từ file PDF
            text = extract_text_from_pdf(pdf_path, is_scanned)
            info_data = extract_info(text)  # Trích xuất thông tin từ văn bản

            # Trích xuất bảng với nhãn và giá trị
            table_data = extract_table_from_pdf_with_labels(pdf_path)
            if table_data:
                info_data['table'] = table_data  # Lưu bảng dưới dạng danh sách các từ điển

            # Gắn tên file để đối chiếu
            info_data['file_name'] = file_name
            all_data.append(info_data)

    # Lưu tất cả dữ liệu vào Excel
    df_main = pd.DataFrame(all_data)

    # Tách từng bảng thành các hàng riêng biệt
    all_rows = []
    for idx, row in df_main.iterrows():
        table = row.get('table', [])
        if isinstance(table, list):  # Nếu có dữ liệu bảng
            for table_row in table:
                combined_row = row.to_dict()
                combined_row.pop('table')  # Loại bỏ trường bảng gốc
                combined_row.update(table_row)  # Kết hợp với dữ liệu bảng
                all_rows.append(combined_row)
        else:
            all_rows.append(row.to_dict())

    # Tạo DataFrame từ dữ liệu kết hợp
    final_df = pd.DataFrame(all_rows)
    final_df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"Data saved to {output_excel}")

# Thực thi
input_folder = "input"  # Thư mục chứa các file PDF
output_excel = "output/invoice_data_combined.xlsx"  # File Excel đầu ra
is_scanned_pdf = False  # Đặt True nếu file PDF là dạng scanned

process_pdfs_with_table_and_labels(input_folder, output_excel, is_scanned=is_scanned_pdf)
