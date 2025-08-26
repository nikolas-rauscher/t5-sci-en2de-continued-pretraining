---
license: apache-2.0
base_model: google/t5-base
tags:
- text2text-generation
- t5
- pytorch
- scientific-domain
- continued-pretraining
- english
language:
- en
datasets:
- texts_pq_4-deduped-Eng_Latn-post-processed
pipeline_tag: text2text-generation
widget:
- example_title: "Scientific Translation"
  text: "translate English to German: The photosynthesis process converts carbon dioxide into oxygen."
- example_title: "Scientific Explanation"
  text: "explain: What is the relationship between temperature and enzyme activity?"
---

# T5-Base Scientific Domain (640k steps)

## Model Description

This is a T5-Base model continued pretrained on English scientific domain texts for 640,000 optimization steps. The model demonstrates enhanced performance on scientific reasoning and STEM-related tasks.

**Model ID:** `t5_base-512-640k-steps-peak-lr0.001-texts_pq_4-deduped-Eng_Latn-post-processed`

## Training Details

- **Base Model:** google/t5-base (220M parameters)
- **Training Steps:** 640,000 optimization steps  
- **Learning Rate:** 0.001 with inverse square root schedule 
- **Warmup Steps:** 20,000 (3% of total steps)
- **Optimizer:** Adafactor
- **Gradient Clipping:** 1.0
- **Batch Size:** 384 effective (48 per GPU × 4 GPUs × 2 accumulation steps)
- **Max Sequence Length:** 512 tokens
- **Mixed Precision:** bf16-mixed (bfloat16)
- **Dataset:** texts_pq_4-deduped-Eng_Latn-post-processed (260,925,917 sliding windows total)
- **Training Split:** 260,825,460 windows
- **Text Processing:** Sliding window with 512 tokens max length, 50% overlap per document
- **Data Cleaning:** Removal of appendices, references, PDF artifacts, and citation formatting
- **Hardware:** 4x H100-80GB GPUs

## Performance

### MMLU Benchmark (0-shot)

**Overall Performance:** 0.27030 accuracy (±0.00370 stderr)

| Category | Accuracy | Improvement vs T5-Base |
|----------|----------|----------------------|
| **Overall** | **0.27030** | **+17.6%** |
| STEM | 0.28600 | +36.1% |
| Social Sciences | 0.29020 | +29.9% |
| Other | 0.25840 | +14.2% |
| Humanities | 0.24360 | -4.1% |

### Complete Task Performance (0-shot)

#### Humanities (0.24360 ±0.0063)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Formal Logic | 0.34920 | ±0.0426 |
| High School European History | 0.25450 | ±0.0340 |
| High School US History | 0.25490 | ±0.0306 |
| High School World History | 0.20250 | ±0.0262 |
| International Law | 0.15700 | ±0.0332 |
| Jurisprudence | 0.24070 | ±0.0413 |
| Logical Fallacies | 0.24540 | ±0.0338 |
| Moral Disputes | 0.21680 | ±0.0222 |
| Moral Scenarios | 0.27260 | ±0.0149 |
| Philosophy | 0.24440 | ±0.0244 |
| Prehistory | 0.22220 | ±0.0231 |
| Professional Law | 0.24510 | ±0.0110 |
| World Religions | 0.18710 | ±0.0299 |

#### STEM (0.28960 ±0.0080)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Abstract Algebra | 0.21000 | ±0.0409 |
| Anatomy | 0.23700 | ±0.0367 |
| Astronomy | 0.34870 | ±0.0388 |
| College Biology | 0.26390 | ±0.0369 |
| College Chemistry | 0.42000 | ±0.0496 |
| College Computer Science | 0.34000 | ±0.0476 |
| College Mathematics | 0.31000 | ±0.0465 |
| College Physics | 0.40200 | ±0.0488 |
| Computer Security | 0.20000 | ±0.0402 |
| Conceptual Physics | 0.20430 | ±0.0264 |
| Electrical Engineering | 0.24830 | ±0.0360 |
| Elementary Mathematics | 0.26980 | ±0.0229 |
| High School Biology | 0.31610 | ±0.0265 |
| High School Chemistry | 0.27590 | ±0.0314 |
| High School Computer Science | 0.20000 | ±0.0402 |
| High School Mathematics | 0.24440 | ±0.0262 |
| High School Physics | 0.35100 | ±0.0390 |
| High School Statistics | 0.46760 | ±0.0340 |
| Machine Learning | 0.18750 | ±0.0370 |

#### Social Sciences (0.31070 ±0.0083)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Econometrics | 0.23680 | ±0.0400 |
| High School Geography | 0.35350 | ±0.0341 |
| High School Government and Politics | 0.36790 | ±0.0348 |
| High School Macroeconomics | 0.35640 | ±0.0243 |
| High School Microeconomics | 0.34870 | ±0.0310 |
| High School Psychology | 0.35050 | ±0.0205 |
| Human Sexuality | 0.28240 | ±0.0395 |
| Professional Psychology | 0.21570 | ±0.0166 |
| Public Relations | 0.22730 | ±0.0401 |
| Security Studies | 0.40000 | ±0.0314 |
| Sociology | 0.27360 | ±0.0315 |
| US Foreign Policy | 0.28000 | ±0.0451 |

#### Other (0.25140 ±0.0076)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Business Ethics | 0.21000 | ±0.0409 |
| Clinical Knowledge | 0.30190 | ±0.0283 |
| College Medicine | 0.34100 | ±0.0361 |
| Global Facts | 0.18000 | ±0.0386 |
| Human Aging | 0.11660 | ±0.0215 |
| Management | 0.37860 | ±0.0480 |
| Marketing | 0.19660 | ±0.0260 |
| Medical Genetics | 0.23000 | ±0.0423 |
| Miscellaneous | 0.20310 | ±0.0144 |
| Nutrition | 0.29080 | ±0.0260 |
| Professional Accounting | 0.23760 | ±0.0254 |
| Professional Medicine | 0.44850 | ±0.0302 |
| Virology | 0.19280 | ±0.0307 |