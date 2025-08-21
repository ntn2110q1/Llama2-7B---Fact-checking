import json
import os

import glob


def preprocess_liar_data(input_path, output_path):

    with open(input_path) as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        evidence = []
        for report in item.get("reports", []):
            for sent in report.get("tokenized", []):
                if sent.get("is_evidence", 0) == 1:
                    evidence.append(sent.get("sent", ""))
        evidence_text = "\n".join(evidence) if evidence else "No evidence provided."

        # Tạo mục dữ liệu với cấu trúc yêu cầu
        cleaned_item = {
            "instruction": "Evaluate the statement and classify it as one of the following based on the provided evidence: True, Mostly True, Half True, Barely True, False, or Pants on Fire.",
            "input": item.get("claim", ""),
            "evidence": evidence_text,
            "output": item.get("label", "")
        }
        cleaned_data.append(cleaned_item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)

    return output_path


def preprocess_rawfc_data(input_dir, output_path):
    cleaned_data = []

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    print(f"Found {len(json_files)} JSON files in {input_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                item = json.load(f)

            # Thu thập evidence từ reports
            evidence = []
            for report in item.get("reports", []):
                for sent in report.get("tokenized", []):
                    if sent.get("is_evidence", 0) == 1:
                        evidence.append(sent.get("sent", ""))
            evidence_text = "\n".join(evidence) if evidence else "No evidence provided."

            # Tạo mục dữ liệu với cấu trúc yêu cầu
            cleaned_item = {
                "instruction": "Evaluate the statement and classify it as one of the following based on the provided evidence: True, Half , False",
                "input": item.get("claim", ""),
                "evidence": evidence_text,
                "output": item.get("label", "")
            }
            cleaned_data.append(cleaned_item)

        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
            continue

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Lưu dữ liệu vào file output
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(cleaned_data)} items and saved to {output_path}")
    return output_path

# LiarRaw
liar_train_path = ".../LIAR-RAW/train.json"
liar_test_path = "/.../LIAR-RAW/test.json"
liar_val_path = ".../LIAR-RAW/val.json"
liar_cleaned_train_path = ".../liar_cleaned_train.json"
liar_cleaned_test_path = ".../liar_cleaned_test.json"
liar_cleaned_val_path = ".../liar_cleaned_val.json"

preprocess_liar_data(liar_train_path, liar_cleaned_train_path)
preprocess_liar_data(liar_test_path, liar_cleaned_test_path)
preprocess_liar_data(liar_val_path, liar_cleaned_val_path)

# RawFC
rawfc_train_path = ".../RAWFC/train.json"
rawfc_test_path = "/.../RAWFC/test.json"
rawfc_val_path = ".../RAWFC/val.json"
rawfc_cleaned_train_path = ".../rawfc_cleaned_train.json"
rawfc_cleaned_test_path = ".../rawfc_cleaned_test.json"
rawfc_cleaned_val_path = ".../rawfc_cleaned_val.json"

preprocess_rawfc_data(rawfc_train_path, rawfc_cleaned_train_path)
preprocess_rawfc_data(rawfc_test_path, rawfc_cleaned_test_path)
preprocess_rawfc_data(rawfc_val_path, rawfc_cleaned_val_path)