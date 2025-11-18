# Cross-Lingual Transfer MMLU Task Analysis

## Overall MMLU Performance

- **AA_wechsel-robust-german-from-640k-en**: 0.24106
- **german-t5-base-nl36**: 0.23757
- **green-run-640k-english-source**: 0.24619
- **t5-base-original**: 0.22995

## Best & Worst Tasks per Model

### AA_wechsel-robust-german-from-640k-en
- **Best**: astronomy (0.303)
- **Worst**: high_school_physics (0.166)
- **Average**: 0.239 ± 0.031

### german-t5-base-nl36
- **Best**: human_aging (0.336)
- **Worst**: professional_medicine (0.173)
- **Average**: 0.239 ± 0.035

### green-run-640k-english-source
- **Best**: abstract_algebra (0.330)
- **Worst**: high_school_statistics (0.167)
- **Average**: 0.246 ± 0.038

## Top 10 Best Performing Tasks (Average)

- human_aging: 0.297
- world_religions: 0.288
- formal_logic: 0.288
- machine_learning: 0.281
- computer_security: 0.275
- high_school_world_history: 0.274
- abstract_algebra: 0.273
- jurisprudence: 0.266
- business_ethics: 0.265
- high_school_computer_science: 0.265

## Top 10 Worst Performing Tasks (Average)

- high_school_physics: 0.184
- high_school_statistics: 0.189
- professional_medicine: 0.199
- security_studies: 0.207
- college_physics: 0.208
- high_school_geography: 0.211
- high_school_microeconomics: 0.213
- high_school_biology: 0.215
- college_chemistry: 0.215
- college_mathematics: 0.217
