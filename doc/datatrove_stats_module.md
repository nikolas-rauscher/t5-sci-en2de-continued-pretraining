# Erklärung der DataTrove Statistikmodule

## Pipeline-Übersicht

Die Statistik-Pipeline ist in zwei Stufen aufgeteilt:
1. Standard-Stats: DocStats, LineStats, ParagraphStats, TokenStats, PerplexityStats (NumPy 2.0 kompatibel)
2. spaCy-Stats: SentenceStats, WordStats, LangStats (NumPy 1.x erforderlich)

Jedes Modul erbt von `BaseStats` und sammelt Metriken in verschiedenen Gruppen:
- summary: Gesamtstatistiken (Durchschnitt, Minimum, Maximum, Anzahl) über alle Dokumente
- histogram: Verteilungen der Metrikwerte, gerundet auf 3 Nachkommastellen (`histogram_round_digits: 3`)

## Standard-Stats Module (NumPy 2.0 kompatibel)

### 1. DocStats - Dokumentebene Statistiken
Berechnet:

 **Generierte Metriken:**
- `length`: Dokumentlänge in Zeichen
- `white_space_ratio`: Anteil von Leerzeichen
- `digit_ratio`: Anteil von Ziffern 
- `uppercase_ratio`: Anteil von Großbuchstaben
- `punctuation_ratio`: Anteil von Satzzeichen
- `non_alpha_digit_ratio`: Anteil von Sonderzeichen (weder Buchstaben noch Zahlen)
- `elipsis_ratio`: Anteil von Auslassungspunkten ("..." oder "…")

### 2. LineStats - Zeilenebenene Analyse
Analysiert die Zeilenstruktur innerhalb der Dokumente.

**Konfigurierte Parameter:**
- `max_k_chars_per_line_tresholds: [10, 30]`: Schwellenwerte für kurze Zeilen
- `min_k_chars_per_line_thresholds: [2000, 10000]`: Schwellenwerte für lange Zeilen
- `ignore_empty_lines: false`: Leere Zeilen werden mitgezählt

**Generierte Metriken:**
- `n_lines`: Anzahl der Zeilen pro Dokument
- `avg_line_length`: Durchschnittliche Zeilenlänge
- `short_line_ratio_chars_10`: Anteil der Zeilen mit ≤10 Zeichen
- `short_line_ratio_chars_30`: Anteil der Zeilen mit ≤30 Zeichen
- `long_line_ratio_chars_2000`: Anteil der Zeilen mit ≥2000 Zeichen
- `long_line_ratio_chars_10000`: Anteil der Zeilen mit ≥10000 Zeichen
- `lines_ending_with_terminal_mark_ratio`: Anteil der Zeilen mit Satzendezeichen
- `bullet_point_lines_ratio`: Anteil der Zeilen mit Aufzählungspunkten
- `line_duplicates`: Anteil doppelter Zeilen
- `line_char_duplicates`: Zeichenanteil in doppelten Zeilen

### 3. ParagraphStats - Absatzebene Statistiken
Analysiert Absatzstrukturen (getrennt durch `\n\n`).

**Konfigurierte Parameter:**
- `short_paragraph_max_chars_threshold: [100]`: Schwellenwert für kurze Absätze
- `long_paragraph_max_chars_threshold: [1000]`: Schwellenwert für lange Absätze
- `ignore_empty_paragraphs: false`: Leere Absätze werden mitgezählt

**Generierte Metriken:**
- `n_paragraphs`: Anzahl der Absätze pro Dokument
- `avg_paragraph_length`: Durchschnittliche Absatzlänge
- `short_paragraph_ratio_100`: Anteil der Absätze mit ≤100 Zeichen
- `long_paragraph_ratio_1000`: Anteil der Absätze mit ≥1000 Zeichen
- `paragraph_duplicates`: Anteil doppelter Absätze
- `paragraph_char_duplicates`: Zeichenanteil in doppelten Absätzen

### 4. TokenStats - Tokenisierung
Berechnet tokenbasierte Statistiken mit dem konfigurierten Tokenizer.

**Konfigurierte Parameter:**
- `tokenizer_name_or_path: "gpt2"`: Verwendeter Tokenizer

**Generierte Metriken:**
- `token_count`: Anzahl der Tokens pro Dokument

### 5. PerplexityStats - Textqualität via KenLM
Berechnet Perplexitätswerte mit KenLM-Sprachmodellen.

**Konfigurierte Variante:**
- `perplexity_stats_wikipedia`: Verwendet Wikipedia-trainiertes Modell
- `model_dataset: "wikipedia"`: Formale/akademische Qualitätsbaseline
- `language: "en"`: Englisches Sprachmodell

**Generierte Metriken:**
- `ccnet_perplexity_wikipedia_en`: Perplexitätswert mit Wikipedia-Modell

**Hinweis:** `perplexity_stats_oscar` ist auskommentiert (Web-Content Baseline).

## spaCy-Stats Module (NumPy 1.x erforderlich)

### 6. SentenceStats - Satzebene Analyse
Analysiert Satzstrukturen mit spaCy-Satzsegmentierung.

**Konfigurierte Parameter:**
- `short_sentence_max_chars_threshold: [20]`: Schwellenwert für kurze Sätze
- `long_sentence_max_chars_threshold: [75]`: Schwellenwert für lange Sätze
- `language: "en"`: Sprache für Satzsegmentierung

**Generierte Metriken:**
- `n_sentences`: Anzahl der Sätze pro Dokument
- `avg_sentence_length`: Durchschnittliche Satzlänge in Zeichen
- `short_sentence_ratio_20`: Anteil der Sätze mit ≤20 Zeichen
- `long_sentence_ratio_75`: Anteil der Sätze mit ≥75 Zeichen

### 7. WordStats - Wortebene Analyse
Detaillierte Wortanalyse mit spaCy-Tokenisierung.

**Konfigurierte Parameter:**
- `short_word_max_chars_threshold: [3]`: Schwellenwert für kurze Wörter
- `long_word_max_chars_threshold: [7]`: Schwellenwert für lange Wörter
- `language: "en"`: Sprache für Wort-Tokenisierung

**Generierte Metriken:**
- `n_words`: Anzahl der Wörter pro Dokument
- `avg_word_length`: Durchschnittliche Wortlänge
- `avg_words_per_line`: Durchschnittliche Wörter pro Zeile
- `short_word_ratio_3`: Anteil der Wörter mit ≤3 Zeichen
- `long_word_ratio_7`: Anteil der Wörter mit ≥7 Zeichen
- `type_token_ratio`: Verhältnis einzigartiger Wörter zu Gesamtwörtern (lexikalische Vielfalt)
- `uppercase_word_ratio`: Anteil komplett großgeschriebener Wörter
- `capitalized_word_ratio`: Anteil großgeschriebener Wörter (Titel-Fall)
- `stop_word_ratio`: Anteil der Stoppwörter

### 8. LangStats - Spracherkennung
Berechnet Spracherkennungsmetriken mit FastText.

**Konfigurierte Parameter:**
- `language: "en"`: Zielsprache für Konfidenzberechnung

**Generierte Metriken:**
- `fasttext_en`: Konfidenzwert für englische Sprache (0-1)

## Pipeline-Konfiguration

**Ausführungsparameter:**
- `tasks: 200`: Gesamtanzahl der Tasks für Parallelisierung
- `workers: 48`: Anzahl gleichzeitiger Worker-Prozesse
- `limit_documents: -1`: Unbegrenzte Dokumentanzahl (überschreibbar via CLI)

**Enriched Documents:**
- `save_enriched_docs: true`: Speichert Parquet-Dateien mit Stats als Metadaten
- Ermöglicht spätere Filterung basierend auf Statistikwerten

**Dual-Environment Setup:**
- `.venv`: NumPy 2.0 kompatible Module (Standard-Stats)
- `.venv_spacy_stats`: NumPy 1.x erforderliche Module (spaCy-Stats)

**Output-Struktur:**
- Primary: `outputs/YYYY-MM-DD/HH-MM-SS/stats/` (Hydra-Output mit Historie)
- Central: `data/statistics/` (Zentrale Sammlung der neuesten Stats)