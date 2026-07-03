# BlackboxNLP 2026 Submission — Review & ACL Formatting Reference

Generated: 2026-06-30  
Paper: `submissions/blackboxnlp2026/paper.tex`  
Deadline: 2026-07-17  
Venue: BlackboxNLP (EMNLP 2026 workshop)  
Source: https://acl-org.github.io/ACLPUB/formatting.html

---

## 1. ACL Formatting Requirements (Official Summary)

### 1.1 Style Files
- Download from: https://github.com/acl-org/acl-style-files
- Required files: `acl.sty`, `acl_natbib.bst`
- LaTeX invocation: `\documentclass[11pt]{article}` + `\usepackage[review]{acl}` for submissions
- Review mode (`[review]`): adds line rulers, page numbers, "Anonymous ACL submission" header
- Final mode (`[final]`): removes line numbers, removes page numbers, shows author names
- Preprint mode (`[preprint]`): keeps page numbers, removes line numbers, shows author names

### 1.2 Paper Length
| Version | Content pages | Other |
|---------|--------------|-------|
| Long paper (review) | 8 | + unlimited references |
| Long paper (camera-ready) | 9 | + acknowledgments + unlimited references |
| Short paper (review) | 4 | + unlimited references |
| Short paper (camera-ready) | 5 | + unlimited references |

**Sections that do NOT count toward page limit:**
- References
- Acknowledgments (camera-ready only; omit from review)
- Limitations
- Ethical Considerations
- Appendices

### 1.3 File Format
- Format: PDF only
- Paper size: **A4** (21 cm × 29.7 cm / 595.276 × 841.89 pts)
- All fonts must be **embedded** — verify with `pdffonts paper.pdf`

### 1.4 Paper Format

**Margins:** 2.5 cm on all sides (set by `acl.sty` via geometry package — do not override)

**Layout:**
- Two-column body
- Column width: 7.7 cm
- Column gap: 0.6 cm
- Column height: 24.7 cm

**Typography:**
| Element | Size |
|---------|------|
| Body text | 11 pt roman (Times Roman preferred) |
| Title | 15 pt bold |
| Authors | 12 pt bold |
| Affiliations | 12 pt roman |
| Section headings | 12 pt bold |
| Subsection headings | 11 pt bold |
| Abstract | 10 pt roman |
| Captions | 10 pt roman |
| References | 10 pt roman |
| Footnotes | 9 pt roman |

**Figures and Tables:**
- Place near first reference in text
- Minimum one column width
- Vector formats (PDF/EPS) preferred; PNG acceptable
- Must be grayscale-readable; color can be used if it is not the sole differentiator
- Captions: 10 pt font; below figures, above tables
- Captions: centered if one line, left-aligned if longer
- Numbered sequentially (Figure 1, Table 1, ...)

**Citations:**
- Author-date format: `(Gusfield, 1997)` via `\citep{}`
- Author-in-text: `Gusfield (1997)` via `\citet{}`
- Two authors: both names; 3+ authors: first author + "et al."
- DOIs required where available; ACL Anthology URL as fallback
- Citation style: `acl_natbib.bst` (set by `acl.sty`, do NOT add `\bibliographystyle{}`)

**Hyperlinks:** Dark blue (#000099), not underlined (set by `acl.sty`)

### 1.5 Special Sections (order matters)

```
... main text sections ...
[Conclusion]
[Limitations]          ← required; no new experiments/figures
[Ethical Considerations] ← optional; no new experiments/figures
[Acknowledgments]      ← camera-ready only; omit from review
References
Appendix A, B, ...     ← after references; follow anonymity rules
```

### 1.6 Anonymization Rules (Review Version)
- No author names, affiliations, or identifying URLs anywhere
- No acknowledgments section
- Self-citations must not reveal authorship
  - OK: `\citet{anonymous2026system}` → renders as "Anonymous (2026)"
  - Avoid phrasing: "We previously showed (Anonymous, 2026)" — use third person
  - Avoid: "in our prior work" or "our codebase"
- Preliminary work at another venue: mention in submission metadata only, not in paper body
- Appendices must also follow anonymity rules

### 1.7 Metadata (Submission System)
- Title, author names, and abstract in the START system must be **plain text** (no LaTeX/HTML)
- Use unicodeit.net to convert math symbols to Unicode for the abstract metadata field
- The PDF abstract can still use LaTeX commands

---

## 2. Pre-Submission Compliance Checklist

### 2.1 Format

| Check | Status | Notes |
|-------|--------|-------|
| `\documentclass[11pt]{article}` | ✅ PASS | Line 1 |
| `\usepackage[review]{acl}` | ✅ PASS | Line 2 |
| `\usepackage{times}` for Times Roman | ✅ PASS | Line 6 |
| A4 paper size | ✅ PASS | pdfinfo: 595.276 × 841.89 pts |
| All fonts embedded | ✅ PASS | pdffonts: all 16 fonts emb=yes |
| Content ≤ 8 pages | ✅ PASS | pdfinfo: 8 pages total |
| Two-column layout | ✅ PASS | via acl.sty |
| Line numbers in margins | ✅ PASS | review mode |
| Page numbers present | ✅ PASS | review mode |
| Anonymous header | ✅ PASS | review mode |

### 2.2 Anonymization

| Check | Status | Notes |
|-------|--------|-------|
| No author names | ✅ PASS | "Anonymous Authors" |
| No institution names | ✅ PASS | "Anonymous Institution" |
| No author URLs or repo links | ✅ PASS | |
| Self-citations anonymized | ✅ PASS | `anonymous2026system`, `anonymous2026benchmark` |
| No acknowledgments section | ✅ PASS | |
| No identifying phrasing ("our prior work") | ✅ PASS | Uses third-person for cited system |

### 2.3 Content Structure

| Check | Status | Notes |
|-------|--------|-------|
| Numbered sections (Arabic) | ✅ PASS | Sections 1–7 |
| Limitations section present | ✅ PASS | After Conclusion, before References |
| Limitations: no new figures/experiments | ✅ PASS | Text-only limitations |
| References before any appendix | ✅ PASS | No appendix in this version |
| `\bibliography{references}` (no `\bibliographystyle`) | ✅ PASS | acl.sty sets bibstyle |
| Full author names in references | ✅ PASS | Verified all 23 entries |
| DOIs / ACL Anthology URLs in references | ✅ PASS | All entries have URL or DOI |
| Author-date citation format throughout | ✅ PASS | `\citep{}` and `\citet{}` used |

### 2.4 Figures and Tables

| Check | Status | Notes |
|-------|--------|-------|
| Figure captions below figures | ✅ PASS | All figures |
| Table captions above tables | ✅ PASS | All tables |
| Figures grayscale-readable | ✅ PASS | Color not sole differentiator; labels on all zones |
| Wide tables use `\resizebox` | ✅ PASS | Tables 1 and 4 |
| Full-width figures use `figure*` | ✅ PASS | Figures 5 and 6 |
| No figure/table overlap | ✅ PASS | Verified in generated PDF |
| TikZ figure fits in column | ✅ PASS | xscale=0.80 confirmed |

### 2.5 Minor Notes (non-blocking)

| Item | Status | Notes |
|------|--------|-------|
| `\usepackage{url}` | ⚠️ REDUNDANT | acl.sty loads hyperref; `url` package harmless but unnecessary |
| `\usepackage{float}` | ⚠️ EXTRA | Not standard ACL; harmless |
| PDF Title metadata empty | ⚠️ MINOR | Not required; can add `\hypersetup{pdftitle={...}}` |
| `\LogAwareStd` macro naming | ⚠️ NAMING | Value = 0.997 is LogAware performance on standard pairs, not σ |

---

## 3. Experiment Value Audit

All values in the paper that are produced by macros were verified against `paper/numbers.tex`. Values that are hardcoded in prose come from original evaluation logs and are consistent internally.

### 3.1 Macro Values (from `numbers.tex`)

| Macro | Value | Used in paper | Correct |
|-------|-------|---------------|---------|
| `\BaselineMF` | 0.385 | Trust-blind GNN macro-F1 | ✅ |
| `\BaselineStd` | 0.007 | ±0.007 across 3 runs | ✅ |
| `\VtwoMF` | 0.918 | v2-HGNN macro-F1 | ✅ |
| `\VtwoStd` | 0.017 | ±0.017 across 3 runs | ✅ |
| `\VthreeMF` | 0.907 | v3-NLI macro-F1 | ✅ |
| `\VthreeStd` | 0.019 | ±0.019 across 3 runs | ✅ |
| `\GapMF` | 0.533 | VtwoMF − BaselineMF | ✅ 0.918−0.385=0.533 |
| `\LogBlindMF` | 0.389 | Trust-blind LogReg probe | ✅ |
| `\LogAwareMF` | 0.882 | Trust-aware LogReg probe (overall) | ✅ |
| `\LogAwareStd` | 0.997 | Trust-aware probe on standard pairs | ✅ (naming confusing but correct) |
| `\MidTierMF` | 0.204 | Trust-aware LogReg probe on mid-tier | ✅ |
| `\MidTierBlindMF` | 0.016 | Trust-blind LogReg on mid-tier | ✅ |
| `\AlwaysNEEMF` | 0.250 | Always-NEE predictor macro-F1 | ✅ |
| `\ECHigh` | 0.781 | EC at ST=0.85, EW=0.80, IS=1.0 | ✅ computed: 1−0.15^0.80=0.781 |
| `\ECLowMax` | 0.380 | EC at ST=0.45, EW=0.80, IS=1.0 | ✅ computed: 1−0.55^0.80=0.380 |
| `\ECMidA` | 0.426 | EC at ST=0.50 | ✅ computed: 1−0.50^0.80=0.426 |
| `\ECMidB` | 0.539 | EC at ST=0.62 | ✅ computed: 1−0.38^0.80=0.539 |
| `\ECMidC` | 0.639 | EC at ST=0.72 | ✅ computed: 1−0.28^0.80=0.639 |
| `\VtwoMidMF` | 0.252 | v2-HGNN mid-tier macro-F1 | ✅ |
| `\VtwoMidStd` | 0.024 | v2-HGNN mid-tier σ | ✅ |
| `\VthreeMidMF` | 0.237 | v3-NLI mid-tier macro-F1 | ✅ |
| `\VthreeMidStd` | 0.026 | v3-NLI mid-tier σ | ✅ |
| `\VtwoStdMF` | 1.000 | v2-HGNN standard-records macro-F1 | ✅ |
| `\VtwoVHAcc` | 0.870 | v2-HGNN HybridVerdictHead accuracy | ✅ |
| `\VthreeVHAcc` | 0.852 | v3-NLI HybridVerdictHead accuracy | ✅ |
| `\VtwoVHErr` | 47 | v2-HGNN errors/run on 360 NEE records | ✅ 360×(1−0.870)=46.8≈47 |
| `\VthreeVHErr` | 53 | v3-NLI errors/run on 360 NEE records | ✅ 360×(1−0.852)=53.3≈53 |
| `\VHTotal` | 360 | NEE records in test set | ✅ NTest×0.60=360 |
| `\NTotal` | 2200 | Total benchmark records | ✅ |
| `\NTrain` | 1600 | Training records | ✅ |
| `\NTest` | 600 | Test records | ✅ NStandard+NMidTier=480+120 |
| `\NStandard` | 480 | Standard-pair test records | ✅ |
| `\NMidTier` | 120 | Mid-tier probe test records | ✅ |

### 3.2 Cross-Checks (Derived Values)

| Check | Formula | Result | Status |
|-------|---------|--------|--------|
| GapMF = VtwoMF − BaselineMF | 0.918 − 0.385 | 0.533 ✅ | PASS |
| NTest = NStandard + NMidTier | 480 + 120 | 600 ✅ | PASS |
| VHTotal = 60% of NTest | 600 × 0.60 | 360 ✅ | PASS |
| VtwoVHErr ≈ VHTotal × (1 − VtwoVHAcc) | 360 × 0.130 | 46.8 ≈ 47 ✅ | PASS |
| VthreeVHErr ≈ VHTotal × (1 − VthreeVHAcc) | 360 × 0.148 | 53.3 ≈ 53 ✅ | PASS |
| Table 2: EC(ST=0.85) | 1−(0.15)^0.80 | 0.781 = ECHigh ✅ | PASS |
| Table 2: EC(ST=0.30) | 1−(0.70)^0.80 | 0.248 ✅ | PASS |

### 3.3 Hardcoded Values (from Original Evaluation Logs)

These values are not in `numbers.tex`; they come from the audited system's own evaluation logs.

| Value | Location | Context |
|-------|----------|---------|
| 0.996 | Abstract, Section 4, Table 1 | Trust-blind accuracy on leaky split |
| 0.889 | Section 4, Table 4 | Trust-aware accuracy on leaky split |
| 0.99 | Section 4, Table 1 | Template field verdict lookup |
| 0.82 | Section 4, Table 1 | Source-id verdict lookup |
| 0.92 | Section 4, Table 1 | Evidence-text verdict lookup |
| 0.54 | Section 4, Table 1 | Stance verdict lookup |
| 0.660 / 0.265 | Section 4, Table 1 | AVeriTeC majority-class acc / macro-F1 |
| 0.656 / 0.354 | Section 4, Table 1 | AVeriTeC text-classifier acc / macro-F1 |
| 0.665 | Section 4 prose | Reported system accuracy on AVeriTeC |
| 462 | Section 4 prose | AVeriTeC dev set size |
| 1.000 | Section 4, Table 1 | Simulation split accuracy |
| 0.82 (first-version leak) | Section 5 prose | Trust-blind on first-version benchmark |
| 13 of 14 template types | Section 4 prose | 100%-predictive templates |
| 40% / 60% | Section 6 prose | EC symbolic path / HybridVerdictHead split |
| 1028 of 1068 distinct texts | Section 5 prose | Texts with multi-label coverage in benchmark |

### 3.4 EC Formula Verification

Formula: EC = 1 − (1 − ST)^(EW × IS)  
Benchmark conditions: EW = 0.80 (testimony), IS = 1.0

| ST | Computed EC | Macro value | Match |
|----|-------------|-------------|-------|
| 0.85 | 1 − 0.15^0.80 = **0.781** | `\ECHigh` = 0.781 | ✅ |
| 0.72 | 1 − 0.28^0.80 = **0.639** | `\ECMidC` = 0.639 | ✅ |
| 0.62 | 1 − 0.38^0.80 = **0.539** | `\ECMidB` = 0.539 | ✅ |
| 0.50 | 1 − 0.50^0.80 = **0.426** | `\ECMidA` = 0.426 | ✅ |
| 0.45 | 1 − 0.55^0.80 = **0.380** | `\ECLowMax` = 0.380 | ✅ |
| 0.30 | 1 − 0.70^0.80 = **0.248** | Table 2 hardcoded | ✅ |

All EC values confirmed correct.

---

## 4. Issues Found

### 4.1 Critical Issues
**None.** The paper is ready for submission.

### 4.2 Non-Blocking Issues (worth addressing)

**[FORMAT-1] PDF metadata title is empty**  
`pdfinfo` shows blank Title field. Add to preamble after `\title{...}`:
```latex
\hypersetup{
  pdftitle={When Trust Doesn't Transfer: Diagnosing Source-Credibility Shortcuts in Fact Verification Models},
  pdfsubject={BlackboxNLP 2026},
}
```
Note: the author field should stay blank in review version.

**[FORMAT-2] Redundant `\usepackage{url}`**  
`acl.sty` loads hyperref (which supersedes url). The explicit `\usepackage{url}` at line 14 is harmless but can be removed.

**[NARRATIVE-1] `\LogAwareStd` macro name is misleading**  
In `paper/numbers.tex`, `\LogAwareStd = 0.997` is used as the LogAware probe's macro-F1 on standard pairs. The suffix `Std` is used elsewhere for σ (e.g., `\VtwoMidStd = 0.024`). No fix needed for submission but worth renaming before camera-ready (e.g., `\LogAwareStdMF` or `\LogAwareStdPairs`).

**[NARRATIVE-2] "mid-tier" label in TikZ uses approximate range**  
The TikZ figure shows "EC 0.43--0.64" while the exact values are 0.426, 0.539, 0.639. This is an intentional approximation in the figure label; the caption uses exact macros. No change needed.

---

## 5. Camera-Ready Checklist (post-acceptance)

When the paper is accepted, these changes are required before final submission:

- [ ] Change `\usepackage[review]{acl}` → `\usepackage[final]{acl}`
- [ ] Replace "Anonymous Authors" with real author names (full names, no initials)
- [ ] Replace "Anonymous Institution" with real affiliations
- [ ] De-anonymize self-citations: replace `anonymous2026system` and `anonymous2026benchmark` with real BibTeX keys
- [ ] Add `\section*{Acknowledgments}` before `\bibliography{}` (funding, compute, dataset credits)
- [ ] Optionally expand content by 1 page (camera-ready allows 9 pages)
- [ ] Add `\hypersetup{pdfauthor={...}}` with real author names
- [ ] Enter real metadata (title, authors, abstract) in the START submission system
- [ ] Re-run `pdffonts` and verify all fonts still embedded
- [ ] Complete ACL Copyright Transfer Agreement
- [ ] Review and incorporate reviewer feedback

---

## 6. Build Commands

```bash
# Compile (from epistemic-factkg-benchmark/)
just pdf-blackbox

# Clean + full rebuild
latexmk -C -cd submissions/blackboxnlp2026/paper.tex
just pdf-blackbox

# Verify font embedding
pdffonts submissions/blackboxnlp2026/paper.pdf

# Verify page count and paper size
pdfinfo submissions/blackboxnlp2026/paper.pdf
```

---

## 7. PDF Verification Results (2026-06-30)

```
Pages:     8
Page size: 595.276 × 841.89 pts (A4)  ✅
Fonts:     16 fonts, all emb=yes sub=yes uni=yes  ✅
```

Font list: NimbusRomNo9L (Times equivalent), NimbusSanL (Helvetica, for line numbers),
CMSY10, CMR10, CMMI10 (math), NimbusMonL (monospace for \texttt). All embedded and subsetted.
