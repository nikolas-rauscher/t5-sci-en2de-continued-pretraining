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

# T5-Base Scientific Domain Conservative (605k steps)

## Model Description

This is a T5-Base model continued pretrained on English scientific domain texts for 605,000 optimization steps using a conservative training approach. The model demonstrates enhanced performance on scientific reasoning tasks with stable training dynamics.

**Model ID:** `t5_base-512-605k-steps-peak-lr0.0001-texts_pq_4-deduped-Eng_Latn-post-processed`

## Training Details

- **Base Model:** google/t5-base (220M parameters)
- **Training Steps:** 605,000 optimization steps  
- **Learning Rate:** 0.0001 with inverse square root schedule 
- **Warmup Steps:** 15,000 (2.5% of total steps)
- **Optimizer:** Adafactor
- **Gradient Clipping:** 0.5
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

**Overall Performance:** 0.26980 accuracy (±0.00370 stderr)

| Category | Accuracy | Improvement vs T5-Base |
|----------|----------|----------------------|
| **Overall** | **0.26980** | **+17.3%** |
| STEM | 0.28230 | +31.7% |
| Social Sciences | 0.30550 | +27.6% |
| Other | 0.26070 | +14.3% |
| Humanities | 0.24420 | -0.7% |

### Complete Task Performance (0-shot)

#### Humanities (0.24420 ±0.0063)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Formal Logic | 0.36510 | ±0.0431 |
| High School European History | 0.25450 | ±0.0340 |
| High School US History | 0.25980 | ±0.0308 |
| High School World History | 0.20250 | ±0.0262 |
| International Law | 0.14050 | ±0.0317 |
| Jurisprudence | 0.20370 | ±0.0389 |
| Logical Fallacies | 0.25770 | ±0.0344 |
| Moral Disputes | 0.22540 | ±0.0225 |
| Moral Scenarios | 0.27260 | ±0.0149 |
| Philosophy | 0.24760 | ±0.0245 |
| Prehistory | 0.21910 | ±0.0230 |
| Professional Law | 0.24450 | ±0.0110 |
| World Religions | 0.19880 | ±0.0306 |

#### STEM (0.28230 ±0.0079)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Abstract Algebra | 0.21000 | ±0.0409 |
| Anatomy | 0.20000 | ±0.0346 |
| Astronomy | 0.31580 | ±0.0378 |
| College Biology | 0.25000 | ±0.0362 |
| College Chemistry | 0.37000 | ±0.0485 |
| College Computer Science | 0.32000 | ±0.0469 |
| College Mathematics | 0.29000 | ±0.0456 |
| College Physics | 0.36270 | ±0.0478 |
| Computer Security | 0.18000 | ±0.0386 |
| Conceptual Physics | 0.22980 | ±0.0275 |
| Electrical Engineering | 0.26210 | ±0.0366 |
| Elementary Mathematics | 0.24340 | ±0.0221 |
| High School Biology | 0.30970 | ±0.0263 |
| High School Chemistry | 0.31030 | ±0.0326 |
| High School Computer Science | 0.17000 | ±0.0378 |
| High School Mathematics | 0.26670 | ±0.0270 |
| High School Physics | 0.35100 | ±0.0390 |
| High School Statistics | 0.47220 | ±0.0340 |
| Machine Learning | 0.16070 | ±0.0349 |

#### Social Sciences (0.30550 ±0.0083)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Econometrics | 0.24560 | ±0.0405 |
| High School Geography | 0.34850 | ±0.0339 |
| High School Government and Politics | 0.34720 | ±0.0344 |
| High School Macroeconomics | 0.32820 | ±0.0238 |
| High School Microeconomics | 0.33610 | ±0.0307 |
| High School Psychology | 0.34500 | ±0.0204 |
| Human Sexuality | 0.27480 | ±0.0392 |
| Professional Psychology | 0.22390 | ±0.0169 |
| Public Relations | 0.25450 | ±0.0417 |
| Security Studies | 0.40410 | ±0.0314 |
| Sociology | 0.27360 | ±0.0315 |
| US Foreign Policy | 0.25000 | ±0.0435 |

#### Other (0.26070 ±0.0078)
| Task | Accuracy | Stderr |
|------|----------|--------|
| Business Ethics | 0.25000 | ±0.0435 |
| Clinical Knowledge | 0.29060 | ±0.0279 |
| College Medicine | 0.33530 | ±0.0360 |
| Global Facts | 0.17000 | ±0.0378 |
| Human Aging | 0.16590 | ±0.0250 |
| Management | 0.33010 | ±0.0466 |
| Marketing | 0.20510 | ±0.0265 |
| Medical Genetics | 0.27000 | ±0.0446 |
| Miscellaneous | 0.20950 | ±0.0146 |
| Nutrition | 0.31050 | ±0.0265 |
| Professional Accounting | 0.24110 | ±0.0255 |
| Professional Medicine | 0.44850 | ±0.0302 |
| Virology | 0.22890 | ±0.0327 |