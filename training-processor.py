import os
import json
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer, Trainer, TrainingArguments
import torch

# Load dữ liệu huấn luyện từ JSON
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, bboxes, labels = [], [], []
    for item in data:
        annotations = item["annotations"][0]["result"]
        for annotation in annotations:
            if annotation["type"] == "textarea":
                texts.append(annotation["value"]["text"])
                bboxes.append(annotation["value"]["bbox"])
                labels.append(annotation["value"]["labels"][0])  # Lấy nhãn đầu tiên
    return texts, bboxes, labels

# Chuẩn bị dữ liệu huấn luyện
def prepare_data(texts, bboxes, labels, tokenizer, label2id):
    encoded_inputs = tokenizer(
        texts,
        boxes=bboxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    encoded_inputs["labels"] = torch.tensor([label2id[label] for label in labels])
    return encoded_inputs

# Tạo Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Đường dẫn
json_path = "train/train.csv"  # File JSON chứa dữ liệu huấn luyện
output_dir = "fine_tuned_layoutlmv3"

# Nhãn và ID
label2id = {"buyer_company": 0, "seller_company": 1, "invoice_number": 2, "date": 3, "total_amount": 4, "vat_amount": 5, "product": 6}
id2label = {v: k for k, v in label2id.items()}

# Load model và tokenizer
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(label2id))
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
model.config.id2label = id2label
model.config.label2id = label2id

# Load dữ liệu
texts, bboxes, labels = load_data(json_path)
encodings = prepare_data(texts, bboxes, labels, tokenizer, label2id)

# Tạo Dataset
dataset = CustomDataset(encodings)

# Huấn luyện
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Bắt đầu huấn luyện
trainer.train()

# Lưu model sau khi huấn luyện
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
