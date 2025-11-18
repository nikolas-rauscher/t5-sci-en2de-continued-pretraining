# Detaillierte MMLU Subtask Analyse - Cross-Lingual Transfer

## Gesamtperformance Vergleich

| Modell | MMLU Score |
|--------|------------|
| T5-Base Original | 0.22995 |
| German T5-Base | 0.23757 |
| **Wechsel German Transfer** | **0.24106** |
| Green Run English | 0.24619 |

## Performance nach Kategorien

| Kategorie | T5-Base | German T5 | **Wechsel German** | English Source |
|-----------|---------|-----------|-------------------|----------------|
| STEM | 0.220 | 0.226 | **0.241** | 0.246 |
| Humanities | 0.247 | 0.251 | **0.254** | 0.256 |
| Social Sciences | 0.233 | 0.244 | **0.229** | 0.241 |
| Other/Medical | 0.233 | 0.240 | **0.229** | 0.233 |

## Top 20 Beste Wechsel Verbesserungen (vs T5-Base)

| Rank | Task | Wechsel Score | Verbesserung % |
|------|------|---------------|----------------|
| 1 | Astronomy | 0.303 | +64.3% |
| 2 | High School Statistics | 0.255 | +61.8% |
| 3 | High School Chemistry | 0.232 | +51.6% |
| 4 | Security Studies | 0.273 | +45.7% |
| 5 | Philosophy | 0.238 | +39.6% |
| 6 | High School Biology | 0.235 | +30.4% |
| 7 | Professional Medicine | 0.239 | +30.0% |
| 8 | Anatomy | 0.237 | +28.0% |
| 9 | High School European History | 0.279 | +27.8% |
| 10 | Global Facts | 0.230 | +27.8% |
| 11 | Abstract Algebra | 0.280 | +27.3% |
| 12 | College Medicine | 0.277 | +26.3% |
| 13 | High School Geography | 0.222 | +25.7% |
| 14 | High School Psychology | 0.244 | +24.3% |
| 15 | International Law | 0.289 | +20.7% |
| 16 | Electrical Engineering | 0.276 | +17.6% |
| 17 | High School Microeconomics | 0.239 | +14.0% |
| 18 | Miscellaneous | 0.269 | +13.4% |
| 19 | High School Government And Politics | 0.218 | +10.5% |
| 20 | College Physics | 0.225 | +9.5% |

## Top 20 Schwächste Wechsel Performance (vs T5-Base)

| Rank | Task | Wechsel Score | Verschlechterung % |
|------|------|---------------|-------------------|
| 1 | Medical Genetics | 0.200 | -33.3% |
| 2 | Human Aging | 0.211 | -32.9% |
| 3 | Virology | 0.199 | -29.8% |
| 4 | Business Ethics | 0.230 | -23.3% |
| 5 | College Computer Science | 0.200 | -23.1% |
| 6 | Marketing | 0.222 | -22.4% |
| 7 | Machine Learning | 0.250 | -20.0% |
| 8 | Us Foreign Policy | 0.240 | -20.0% |
| 9 | High School Physics | 0.166 | -16.7% |
| 10 | College Chemistry | 0.180 | -14.3% |
| 11 | High School Us History | 0.216 | -13.7% |
| 12 | Nutrition | 0.199 | -12.9% |
| 13 | World Religions | 0.287 | -10.9% |
| 14 | Formal Logic | 0.286 | -10.0% |
| 15 | Professional Accounting | 0.213 | -9.1% |
| 16 | Logical Fallacies | 0.209 | -8.1% |
| 17 | Jurisprudence | 0.241 | -7.1% |
| 18 | Professional Psychology | 0.232 | -6.6% |
| 19 | Public Relations | 0.209 | -4.2% |
| 20 | Econometrics | 0.228 | -3.7% |

## Cross-Lingual Transfer Erfolg (Wechsel vs English Source)

### Top 20 Tasks wo German Transfer besser als English Source

| Rank | Task | German Score | English Score | Vorteil |
|------|------|--------------|---------------|---------|
| 1 | Security Studies | 0.273 | 0.171 | +0.102 |
| 2 | High School Statistics | 0.255 | 0.167 | +0.088 |
| 3 | Formal Logic | 0.286 | 0.230 | +0.056 |
| 4 | Astronomy | 0.303 | 0.250 | +0.053 |
| 5 | High School Macroeconomics | 0.231 | 0.179 | +0.051 |
| 6 | College Physics | 0.225 | 0.176 | +0.049 |
| 7 | Human Sexuality | 0.260 | 0.221 | +0.038 |
| 8 | Professional Medicine | 0.239 | 0.202 | +0.037 |
| 9 | International Law | 0.289 | 0.256 | +0.033 |
| 10 | High School World History | 0.291 | 0.262 | +0.030 |
| 11 | High School Microeconomics | 0.239 | 0.210 | +0.029 |
| 12 | World Religions | 0.287 | 0.257 | +0.029 |
| 13 | College Medicine | 0.277 | 0.254 | +0.023 |
| 14 | College Biology | 0.264 | 0.243 | +0.021 |
| 15 | Medical Genetics | 0.200 | 0.180 | +0.020 |
| 16 | Computer Security | 0.290 | 0.270 | +0.020 |
| 17 | Sociology | 0.244 | 0.229 | +0.015 |
| 18 | Professional Law | 0.256 | 0.243 | +0.013 |
| 19 | High School Psychology | 0.244 | 0.233 | +0.011 |
| 20 | College Computer Science | 0.200 | 0.190 | +0.010 |

### Top 20 Tasks mit größtem Transfer-Verlust

| Rank | Task | German Score | English Score | Verlust |
|------|------|--------------|---------------|---------|
| 1 | Human Aging | 0.211 | 0.327 | -0.117 |
| 2 | Public Relations | 0.209 | 0.318 | -0.109 |
| 3 | Machine Learning | 0.250 | 0.312 | -0.062 |
| 4 | Virology | 0.199 | 0.259 | -0.060 |
| 5 | High School Chemistry | 0.232 | 0.291 | -0.059 |
| 6 | Abstract Algebra | 0.280 | 0.330 | -0.050 |
| 7 | Nutrition | 0.199 | 0.248 | -0.049 |
| 8 | Management | 0.175 | 0.223 | -0.049 |
| 9 | High School Mathematics | 0.230 | 0.274 | -0.044 |
| 10 | Logical Fallacies | 0.209 | 0.252 | -0.043 |
| 11 | Philosophy | 0.238 | 0.280 | -0.042 |
| 12 | Global Facts | 0.230 | 0.270 | -0.040 |
| 13 | Jurisprudence | 0.241 | 0.278 | -0.037 |
| 14 | Professional Accounting | 0.213 | 0.248 | -0.035 |
| 15 | High School Computer Science | 0.270 | 0.300 | -0.030 |
| 16 | Elementary Mathematics | 0.217 | 0.246 | -0.029 |
| 17 | Professional Psychology | 0.232 | 0.258 | -0.026 |
| 18 | Prehistory | 0.241 | 0.265 | -0.025 |
| 19 | Anatomy | 0.237 | 0.259 | -0.022 |
| 20 | Marketing | 0.222 | 0.244 | -0.021 |

## Alle MMLU Subtasks - Detaillierte Scores

| Task | T5-Base | German T5 | **Wechsel German** | English Source | Wechsel vs T5 % | Wechsel vs German % |
|------|---------|-----------|-------------------|----------------|-----------------|-------------------|
| Astronomy | 0.184 | 0.184 | **0.303** | 0.250 | +64.3% | +64.3% |
| High School World History | 0.274 | 0.270 | **0.291** | 0.262 | +6.2% | +7.8% |
| Computer Security | 0.280 | 0.260 | **0.290** | 0.270 | +3.6% | +11.5% |
| International Law | 0.240 | 0.264 | **0.289** | 0.256 | +20.7% | +9.4% |
| World Religions | 0.322 | 0.287 | **0.287** | 0.257 | -10.9% | +0.0% |
| Formal Logic | 0.317 | 0.317 | **0.286** | 0.230 | -10.0% | -10.0% |
| Abstract Algebra | 0.220 | 0.260 | **0.280** | 0.330 | +27.3% | +7.7% |
| High School European History | 0.218 | 0.212 | **0.279** | 0.279 | +27.8% | +31.4% |
| College Medicine | 0.220 | 0.277 | **0.277** | 0.254 | +26.3% | +0.0% |
| Electrical Engineering | 0.234 | 0.214 | **0.276** | 0.269 | +17.6% | +29.0% |
| Security Studies | 0.188 | 0.196 | **0.273** | 0.171 | +45.7% | +39.6% |
| High School Computer Science | 0.250 | 0.240 | **0.270** | 0.300 | +8.0% | +12.5% |
| Miscellaneous | 0.238 | 0.262 | **0.269** | 0.278 | +13.4% | +2.9% |
| College Biology | 0.257 | 0.222 | **0.264** | 0.243 | +2.7% | +18.8% |
| Conceptual Physics | 0.260 | 0.264 | **0.260** | 0.277 | +0.0% | -1.6% |
| Human Sexuality | 0.252 | 0.237 | **0.260** | 0.221 | +3.0% | +9.7% |
| Professional Law | 0.246 | 0.248 | **0.256** | 0.243 | +3.7% | +3.2% |
| High School Statistics | 0.157 | 0.176 | **0.255** | 0.167 | +61.8% | +44.7% |
| Machine Learning | 0.312 | 0.250 | **0.250** | 0.312 | -20.0% | +0.0% |
| High School Psychology | 0.196 | 0.229 | **0.244** | 0.233 | +24.3% | +6.4% |
| Sociology | 0.244 | 0.244 | **0.244** | 0.229 | +0.0% | +0.0% |
| Prehistory | 0.222 | 0.219 | **0.241** | 0.265 | +8.3% | +9.9% |
| Jurisprudence | 0.259 | 0.287 | **0.241** | 0.278 | -7.1% | -16.1% |
| Us Foreign Policy | 0.300 | 0.250 | **0.240** | 0.230 | -20.0% | -4.0% |
| Moral Disputes | 0.234 | 0.228 | **0.240** | 0.237 | +2.5% | +5.1% |
| High School Microeconomics | 0.210 | 0.193 | **0.239** | 0.210 | +14.0% | +23.9% |
| Professional Medicine | 0.184 | 0.173 | **0.239** | 0.202 | +30.0% | +38.3% |
| Philosophy | 0.170 | 0.225 | **0.238** | 0.280 | +39.6% | +5.7% |
| Anatomy | 0.185 | 0.244 | **0.237** | 0.259 | +28.0% | -3.0% |
| Moral Scenarios | 0.237 | 0.249 | **0.237** | 0.256 | +0.0% | -4.9% |
| High School Biology | 0.181 | 0.200 | **0.235** | 0.242 | +30.4% | +17.7% |
| Professional Psychology | 0.248 | 0.217 | **0.232** | 0.258 | -6.6% | +6.8% |
| High School Chemistry | 0.153 | 0.217 | **0.232** | 0.291 | +51.6% | +6.8% |
| High School Macroeconomics | 0.215 | 0.249 | **0.231** | 0.179 | +7.1% | -7.2% |
| Business Ethics | 0.300 | 0.280 | **0.230** | 0.250 | -23.3% | -17.9% |
| Global Facts | 0.180 | 0.190 | **0.230** | 0.270 | +27.8% | +21.1% |
| High School Mathematics | 0.215 | 0.207 | **0.230** | 0.274 | +6.9% | +10.7% |
| Econometrics | 0.237 | 0.228 | **0.228** | 0.246 | -3.7% | +0.0% |
| Clinical Knowledge | 0.208 | 0.260 | **0.226** | 0.238 | +9.1% | -13.0% |
| College Physics | 0.206 | 0.225 | **0.225** | 0.176 | +9.5% | +0.0% |
| High School Geography | 0.177 | 0.207 | **0.222** | 0.237 | +25.7% | +7.3% |
| Marketing | 0.286 | 0.252 | **0.222** | 0.244 | -22.4% | -11.9% |
| High School Government And Politics | 0.197 | 0.254 | **0.218** | 0.228 | +10.5% | -14.3% |
| Elementary Mathematics | 0.209 | 0.212 | **0.217** | 0.246 | +3.8% | +2.5% |
| High School Us History | 0.250 | 0.250 | **0.216** | 0.235 | -13.7% | -13.7% |
| Professional Accounting | 0.234 | 0.241 | **0.213** | 0.248 | -9.1% | -11.8% |
| Human Aging | 0.314 | 0.336 | **0.211** | 0.327 | -32.9% | -37.3% |
| College Mathematics | 0.210 | 0.250 | **0.210** | 0.200 | +0.0% | -16.0% |
| Public Relations | 0.218 | 0.227 | **0.209** | 0.318 | -4.2% | -8.0% |
| Logical Fallacies | 0.227 | 0.209 | **0.209** | 0.252 | -8.1% | +0.0% |
| College Computer Science | 0.260 | 0.220 | **0.200** | 0.190 | -23.1% | -9.1% |
| Medical Genetics | 0.300 | 0.300 | **0.200** | 0.180 | -33.3% | -33.3% |
| Nutrition | 0.229 | 0.209 | **0.199** | 0.248 | -12.9% | -4.7% |
| Virology | 0.283 | 0.259 | **0.199** | 0.259 | -29.8% | -23.3% |
| College Chemistry | 0.210 | 0.270 | **0.180** | 0.200 | -14.3% | -33.3% |
| Management | 0.175 | 0.311 | **0.175** | 0.223 | +0.0% | -43.8% |
| High School Physics | 0.199 | 0.185 | **0.166** | 0.185 | -16.7% | -10.7% |