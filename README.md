# Llama2-7B for Fact-checking
Triển khai Llama2-7B cho bài toán xác thực tin tức

# Overview
Xây dựng và triển khai mô hình Llama2-7B, tinh chỉnh bằng qLoRA cho bài toán xác thực tin tức. Mô hình có hiệu suất đáng kể so với các mô hình học sâu cơ sở và mô hình Llama2-7B chưa tinh chỉnh trên hai tập dữ liệu công khai về xác thực tin tức là LIAR dataset và RAWFC dataset.

## Baselines Evaluation
Performance Table of My Model and Baseline Models. This shows that my model has an improvement in Precision, Recall and F1-score :
| Model         | LIAR (Precision) | LIAR (Recall) | LIAR (F1) | RAWFC (Precision) | RAWFC (Recall) | RAWFC (F1) |
|---------------|-----------|----------|-----------|----------|----------|----------|
| DeClarE    | 22.86     | 20.55    | 18.43     | 43.39    | 43.52   | 42.18    |
| HAN   | 22.64     | 19.96    | 18.46     | 45.66   | 45.54   | 44.25   |
| GenFE   | 28.01     | 26.16    | 26.49     | 44.29    | 44.74    | 44.43   |
| dEFEND   | 23.09     | 18.56    | 17.51     | 44.93    | 43.26   | 44.07    |
| CofCED   | 29.48      | 29.55    | 28.93     | 52.99    | 50.99   | 51.07    |
| Llama2-7B    | 15.87     | 20.69    | 12.24     | 33.50    | 32.55    | 26.43    |
| Llama2-7B (qLoRA)   | **32.24**    | **31.96**   | **30.32**   | **55.11** | **54.50** | **55.40** |

## Dataset
My model was trained and evaluated on the `LIAR-RAW` and `RAWFC` datasets. For detailed information about these two datasets, please refer to the following paper: [links](https://arxiv.org/pdf/2209.14642).

The raw datasets can be downloaded at: [CofCED](https://github.com/Nicozwy/CofCED)

Tôi đã tiền xử lý và làm sạch dữ liệu tại `dataProcessing.py`. Dữ liệu được làm sạch có tại `cleanData`.

## Models
Mô hình của tôi sử dụng 'Llama2-7B' đã được tinh chỉnh bằng tinh chỉnh có hướng dẫn dựa trên phương pháp lượng tử hóa LoRA 8-bit. 
