# Llama2-7B for Fact-checking
Implementation of Llama2-7B for evidence-based fact-checking.

# Overview
Building and deploying a Llama2-7B model fine-tuned with QLoRA for the task of fact-checking. The instruction-tuning dataset was constructed and preprocessed based on two public fact-checking datasets: the LIAR dataset and the RAWFC dataset. The resulting model demonstrates significant performance improvements compared to baseline deep learning models and the original (unfine-tuned) Llama2-7B.

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
| Llama2-7B (QLoRA)   | **32.24**    | **31.96**   | **30.32**   | **55.11** | **54.50** | **55.40** |

## Dataset
My model was trained and evaluated on the `LIAR-RAW` and `RAWFC` datasets. For detailed information about these two datasets, please refer to the following paper: [links](https://arxiv.org/pdf/2209.14642).

The raw datasets can be downloaded at: [CofCED](https://github.com/Nicozwy/CofCED)

Data preprocessing and cleaning were performed in `dataProcessing.py`. The instruction-tuning dataset is available in the `cleanData`.

## Models
My model uses the 'Llama2-7B' model fine-tuned with instruction tuning based on 8-bit quantized LoRA.

To apply the quantization-based LoRA method (`qLoRA`), you need to have [Bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library installed.
<p align="center">
    <br>
    <a href="[https://github.com/safe](https://github.com/ntn2110q1/Llama2-7B---Fact-checking)">
        <img src="https://github.com/ntn2110q1/Llama2-7B---Fact-checking/blob/main/frameWork.png" width="1000"/>
    </a>
    <br>
<p>

## Acknowledgment
This project is built based on [Alpaca-Lora](https://github.com/tloen/alpaca-lora.git).
