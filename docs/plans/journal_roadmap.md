# Journal Paper Roadmap

Extended paper for MDPI Technologies special issue, building on the AIC 2025 conference paper.

## Target Journal

- **Journal:** MDPI Technologies (Impact Factor 3.6, 5-yr IF 4.2; Phil, Jul 2026: "impact factor >5")
- **Special issue:** "AI-Driven Color Models for Imaging, Formulation, Appearance Measurement and Computer Vision"
- **Guest editors:** Eric J.J. Kirchner, Stephen Westland
- **Submission deadline:** 30 August 2026
- **Open access APC:** ~$1250 (funding TBD)
- **Special issue page:** https://www.mdpi.com/journal/technologies/special_issues/0O2229T6RE
- **Follow-up talk:** ICC Expert Day, Gjøvik, 21 September 2026 — Hamza presents an outline (after the submission deadline; avoid disclosing too much before the article is published)

## Background

The AIC 2025 paper evaluated 14 ML methods for CMY→XYZ color prediction on 3 printer datasets (PC10, PC11, FOGRA51) using CIEDE2000. Phil presented in Taipei (Oct 2025). Eric Kirchner was interested in the results and invited submission to this special issue. The AIC paper answered the question for ML with n=3 — the journal paper extends this.

## Core Research Questions (from Phil)

> "The main contribution could be: a) can ML/AI give as good or better results than existing methods on n≤4, avoiding the need for gray component algorithms etc; and b) can AI handle n>4 successfully. Either of these has potential to be a quite significant contribution in the field."

## Research Extensions (prioritized)

### High Priority — Core Contributions

1. **n>4 colorant systems (CMYKOGV, etc.)**
   - The key innovation per Phil
   - Can AI/ML handle higher dimensionality where traditional polynomial methods struggle?
   - *Data:* Waiting on Phil for n-colour datasets (n>4). He'll also check if they can print new targets depending on RIP control.

2. **LLM as direct color predictor**
   - Feed CMY/CMYK sample data directly to Claude/GPT and have it predict XYZ values
   - Suggested at ICC GASIG meeting (Jul 2025) — "Claude would do a better job than ChatGPT models"
   - Compare LLM prediction accuracy against the 14 traditional ML methods
   - Test with different models: Claude (Sonnet/Opus), GPT-4o, o1, o3-mini

3. **CMYK source space**
   - Extend from CMY (3-channel) to CMYK (4-channel)
   - Stepping stone to n>4
   - Data already available (datasets have CMYK_K column, currently filtered out)

### Medium Priority

4. **Larger / combined datasets**
   - Phil has a large newsprint dataset (~200 reproductions of the same test target by different printers)
   - Train across multiple datasets to improve generalization
   - Compare single-dataset vs multi-dataset training

5. **Direct ΔE2000 minimization**
   - Current pipeline optimizes MSE on XYZ, then evaluates with ΔE₀₀
   - Use CIEDE2000 directly as the loss/objective function
   - Custom loss for neural networks, or wrap in optimization loop for other methods

6. **Additional optimization methods**
   - Genetic algorithms (students previously found worse than other approaches)
   - Nelder-Mead simplex
   - GRG (Generalized Reduced Gradient)
   - Novel challenge: increasing dimensionality for n>3

### Lower Priority (if time permits)

7. **Linearization preprocessing** — standard industry practice, do in conjunction with other methods. Phil: "could possibly be skipped if time is limited"

8. **Colorimetric density as input domain** — alternative to raw CMYK values

## Meeting Notes — 16 April 2026 (with Phil)

- Benchmark against prior published results: the PhD work at https://scholar.google.com/citations?user=e19-J04AAAAJ&hl=en, and Kiran's Optics Express paper on n-colour.
- Will (ex-Kodak) is the contact for the n>4 dataset — Phil to ask.
- Lookup tables become inefficient at larger numbers of colorants (supports the ML-for-n>4 argument).
- Claude/GPT-based approaches: check for overfitting — cross-fold validation; train on a proportion of the data and check errors on the rest.
- AIC 2025 proceedings available at https://www.aic2025.org/.

## Writing Guidance (from Phil)

- Give full details of methods including equations
- Check results carefully — some in AIC paper looked anomalous
- Avoid excess precision: 2-3 decimal places is enough for color differences
- Polynomial regression: cap at 3rd order (higher degrees overfit)
- Report three summary statistics: central tendency (median), range (max), distribution (95th percentile)
- Median and 95th percentile preferred over mean and stdev (errors not normally distributed)

## Data Status

| Dataset | Channels | Status |
|---------|----------|--------|
| APTEC PC10 (cardboard) | CMY(K) | Available |
| APTEC PC11 (coated paper) | CMY(K) | Available |
| FOGRA51 (reference) | CMY(K) | Available |
| IFRA newsprint (wb/bb) | CMYK + spectral | Received 29 Mar 2026 (`journal/data/raw/Ifra-{wb,bb}.zip`); keep wb/bb separate — wb is the industry norm, bb has more runs |
| n>4 colorant datasets | CMYKOGV+ | Not yet received — Phil to ask Will (ex-Kodak) |
| Potential new prints | n-colour | Depends on RIP control |

## Status (1 Aug 2026) — what's DONE

- **AIC evaluation error found & corrected (Jul 2026).** v1 computed ΔE00 on normalized XYZ (~5× understatement). Verified 4 independent ways; Phil informed 23 Jul and agreed: **no AIC erratum; journal = new analysis; consider mentioning the error in the article.** Published state preserved at git tag `aic2025-published`; v1 at HEAD is corrected. Materials sent to Phil: `~/Desktop/AIC-correction-for-Phil/`.
- **Journal pipeline built** (`journal/pipeline/`): n-channel-generic, 5-fold CV, per-fold scaling, ΔE00 on denormalized XYZ, XYZ→Lab roundtrip tripwire on every load.
- **First results committed** (`journal/results/`, 6 variants × 14 models). Headlines: GP median ≈ 0.05–0.07 everywhere (needs grouped-split control before it's a paper claim); poly3 second at n=3 (0.28–0.37) but degrades ~3× at n=4 (0.82–0.94, max ≈ 30) while GP barely moves — first evidence for core question (b).
- **Design rule (agreed):** CMY experiments → 818 K=0 rows; CMYK → all 1,617 rows with K as input; never drop a column while keeping its rows.

## Execution plan — 4 weeks to deadline (30 Aug)

| Week | Work |
|------|------|
| **W1 (1–8 Aug)** | GP verification (grouped CV vs duplicate patches; compare to instrument repeatability ~0.2–0.3 ΔE00). IFRA ingestion (CGATS latin-1 parse, spectral→XYZ D50/2°, register as IFRA-wb/-bb) + first generalization runs (train-on-one-press-run→test-on-others; combined vs single training). **Reply to Phil: meeting + results table + chase n>4 data (Will, ex-Kodak).** |
| **W2 (9–15 Aug)** | LLM-as-color-predictor track (`journal/llm/`): in-context patches → predict XYZ, same ΔE00 scoring, vs the 14 methods. Direct-ΔE00 loss where it can matter (MLP custom loss; Nelder-Mead on poly coefficients — also covers "other optimizers"). CMYKOGV run if data arrives (pipeline is n-generic; ingestion + one command). **n>4 data cutoff ~15 Aug:** if not in hand, reframe paper as n≤4 + newsprint generalization, n>4 as future work. |
| **W3 (16–22 Aug)** | Writing takes over: full draft, all figures from results CSVs, correction paragraph + evaluation-pitfall section (tone per Phil). Experiments only to fill reviewer-visible gaps. |
| **W4 (23–30 Aug)** | Phil review round, revisions, MDPI formatting, submit. Nothing new lands here. |
| **21 Sep** | ICC Expert Day, Gjøvik — present outline. |

**Cut list if time runs out** (per Phil): linearization preprocessing, colorimetric density domain, genetic algorithms.

## Key Contacts

- **Phil Green** (philip.green@ntnu.no) — co-author, supervisor, at NTNU in May
- **Hamza Zafar** (muhammad.h.zafar@ntnu.no) — lead author, guest researcher at NTNU
