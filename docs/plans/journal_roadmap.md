# Journal Paper Roadmap

Extended paper for MDPI Technologies special issue, building on the AIC 2025 conference paper.

## Target Journal

- **Journal:** MDPI Technologies (Impact Factor 3.6, 5-yr IF 4.2)
- **Special issue:** "AI-Driven Color Models for Imaging, Formulation, Appearance Measurement and Computer Vision"
- **Guest editors:** Eric J.J. Kirchner, Stephen Westland
- **Submission deadline:** 30 August 2026
- **Open access APC:** ~$1250 (funding TBD)
- **Special issue page:** https://www.mdpi.com/journal/technologies/special_issues/0O2229T6RE

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
| Newsprint (~200 targets) | ? | Phil to provide |
| n>4 colorant datasets | CMYKOGV+ | Phil to provide |
| Potential new prints | n-colour | Depends on RIP control |

## Timeline

| Period | Focus |
|--------|-------|
| **Mar–Apr 2026** | CMYK extension (data ready), LLM predictor experiments, set up n>4 framework |
| **May 2026** | Phil at NTNU — in-person collaboration, review progress, receive datasets |
| **Jun–Jul 2026** | n>4 experiments, additional methods, write-up, generate figures |
| **Aug 2026** | Final revisions, submit by Aug 30 |

## Key Contacts

- **Phil Green** (philip.green@ntnu.no) — co-author, supervisor, at NTNU in May
- **Hamza Zafar** (muhammad.h.zafar@ntnu.no) — lead author, guest researcher at NTNU
