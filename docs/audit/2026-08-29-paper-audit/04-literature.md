# Paper audit (29 Aug 2026) — literature positioning

Checked the Introduction, Related Work, Discussion and Conclusions against the published literature. Every DOI below was resolved via the Crossref REST API and matched to the stated title/authors (arXiv DOIs confirmed via doi.org). No file edited.

**Headline: the experiments are sound and the related-work section is thin. Five domain citations (Bala 2003, Kang 1992, Tominaga 1993, Cheung 2004, Deshpande 2014) carry the whole positioning, and nothing after 2004 in the paper's own field is cited. Two "to the authors' knowledge a first" claims are falsifiable with one search and must be narrowed.**

## Point-by-point

### 1. "ML for characterization since the early 1990s; whether modern methods succeed remains unsettled" — OVERSTATED
There is a continuous 2002–2025 literature on NNs, ensembles and deep models for CMYK→CIELAB printer models that the paper does not cite. Fair to say it is fragmented and rarely uses a corrected ΔE00 protocol across ink counts; not fair to imply the question has sat idle since 1993. Bala's chapter itself summarises Kang & Anderson 1992 as "superior fits to training data but inferior performance for independent test data" (verified in the chapter text, p. 319).

| Ref | DOI | Relevance |
|---|---|---|
| Littlewood, Drakopoulos, Subbarayan 2002, ACM TOG 21(2) | 10.1145/508357.508361 | ANN CIELAB↔CMYK for printers |
| Littlewood & Subbarayan 2006, JIST 50(6) | 10.2352/j.imagingsci.technol.(2006)50:6(556) | CMYK model maintenance |
| Artusi, Campadelli, Schettini 1999 | 10.1007/978-1-4471-0811-5_30 | Boosting for printer characterization |
| Su et al. 2021, Coloration Technology 137 | 10.1111/cote.12529 | Wavelet NN CMYK→CIELab, mean ΔE 3.47 |
| Zhan et al. 2025, Color Res. Appl. 50 | 10.1002/col.22971 | Stacking ensemble CMYK→CIELab; most recent direct competitor |
| He, Xiao, Pointer, Bressler, Liu 2023, LIM 4 | 10.2352/lim.2023.4.1.18 | 3rd-order PR vs DNN, CMYK→CIELAB; **compares XYZ / log XYZ / CIELAB output spaces** |
| Chen & Urban 2021, Opt. Express 29 | 10.1364/oe.410796 | Deep learning forward model, six-material 3D printer |
| Chen & Urban 2023, Opt. Express 31 | 10.1364/oe.487526 | Multi-printer transfer learning (relevant to IFRA) |
| Molada-Tebar et al. 2019, Sensors 19 | 10.3390/s19214610 | **GP for device characterization** (camera). No GP-for-printer paper found; cite this before claiming GP novelty |
| Kucuk, Finlayson, Mantiuk, Ashraf 2023, J. Imaging 9 | 10.3390/jimaging9100214 | NN advantage over regression shrinks once regression is optimised for perceptual error; corroborates Sec 4.6 |
| Balasubramanian 1999, JEI 8 | 10.1117/1.482694 | Neugebauer optimisation minimising CIELAB error |
| Bala, Sharma, Monga, Van de Capelle 2005, IEEE TIP 14 | 10.1109/tip.2005.851678 | Modern classical baseline |

### 2. "Fitting in CIELAB rather than XYZ is long-standing practice" — SOUND BUT UNDER-CITED, AND THE BALA CITATION IS WEAK
Direct check of Bala 2003 (chapter PDF, pp. 269–384):
- Sec 5.4.3 polynomial regression: the worked example is **RGB → XYZ tristimulus** ("mapping device RGB space to XYZ tristimulus space"). No statement that regression should be posed in CIELAB.
- Sec 5.5 metrics: "The error is not always calculated in a visually meaningful color space … evaluation of errors with visually relevant metrics is strongly recommended." This is about *evaluation*, not fitting.
- Sec 5.10.3: empirical LUT procedure maps "CMY to CIELAB space"; Tominaga's NN trained "from CMYK to CIELAB". Supports "CIELAB is the conventional output space for printer models" indirectly.
- Sec 5.10.2.3 (Neugebauer), verified: "An alternative technique for determining dot areas is to minimize the error in CIELAB rather than spectral coordinates … a nonlinear optimization problem." Supports *model-parameter fitting on a CIELAB objective* for physical models, which is the closest thing in the chapter to the paper's claim.
- **A summary tool's claimed quote "regression should be performed in CIELAB space rather than XYZ" does not exist in the chapter.** Do not trust that sentence.
Verdict on l.79 "Regression-based characterization is conventionally posed with the colorimetric target expressed in a perceptually uniform space \citep{bala2003device}": **not supported as a regression statement**; Bala's regression example targets XYZ. Reword to "printer characterization models conventionally take CIELAB as the output space \citep{bala2003device}, and evaluation in a perceptually uniform space is strongly recommended there" or swap in the sources below.

Confirmed supporting sources: Balasubramanian 1999 (10.1117/1.482694); Gerhardt & Hardeberg 2008, CRA 33 (10.1002/col.20444); Westland, Ripamonti, Cheung 2012, Computational Colour Science using MATLAB (10.1002/9780470710890); Ji et al. 2020, CRA 45 (10.1002/col.22563).

**Has anyone quantified the XYZ-vs-Lab penalty?** Yes, partially: He et al. 2023 compared XYZ, log XYZ, CIELAB, spectra as output for 3rd-order PR and DNN on a CMYK 3D printer; log XYZ / CIELAB beat raw XYZ. Pointer is a co-author; colour-community reviewers will notice. The claim "what is new is the measurement of how large the penalty is" must narrow to "on physical CMYK(+N) press charts across n = 3..7".

### 3. "n>4 uses analytical models; ML for n>4 is a first" — WRONG AS WRITTEN
(a) The n-colour literature is mostly spectral Neugebauer, not SCOP: Chen, Berns, Taplin 2004, JIST 48(6) (10.2352/j.imagingsci.technol.2004.48.6.art00009); **Babaei & Hersch 2016, IEEE TIP 25(7), "N-ink printer characterization with barycentric subdivision" (10.1109/tip.2016.2560526)**, the standard n-ink reference; Boll 1994, SPIE 2170 (10.1117/12.173839); Coppel, Sole, Hardeberg 2014, CIC 22 (10.2352/cic.2014.22.1.art00043), NTNU's own multi-channel programme; Deshpande, Green, Pointer 2015, CRA 40 (10.1002/col.21909).
(b) **ML has been applied beyond four colorants**: Shi et al. 2018, ACM TOG 37(6) (10.1145/3272127.3275057), neural spectral predictor for 10-ink stacks; Chen & Urban 2021, six-material printer. The "first" clause (Intro and Related Work) must go. Defensible claim: first systematic comparison of a broad regression suite against a properly fitted polynomial on conventional halftone press charts at n = 5 and 7.

### 4. "LLM as zero-shot color predictor is a first" — OVERSTATED
No paper asks an LLM for CMYK→CIELAB specifically, so a narrow first is defensible, but: Vacareanu et al. 2024, COLM (10.48550/arXiv.2404.07544), LLMs as in-context regressors; **Shojaee et al., LLM-SR, ICLR 2025 (10.48550/arXiv.2404.18400)**, LLM writes equation skeletons + code, which is exactly the paper's code-execution condition; Gruver et al. 2023, NeurIPS (10.52202/075280-0861); Fukushima et al. 2025, Discover AI (10.1007/s44163-025-00323-8); Mukherjee et al. 2026, Cognitive Science (10.1111/cogs.70219).

### 5. Degree cap at 3 "following standard practice" — SOUND, easy to source
Bala 2003 Sec 5.4.3 (verified in text): worked example is third order; "we recommend using the smallest number of polynomial terms that adequately fits the curvature of the function while still smoothing out the noise … obtained by experimentation, intuition, and experience." Hong, Luo, Rhodes 2001, CRA 26(1) (10.1002/1520-6378(200102)26:1<76::AID-COL8>3.0.CO;2-3): up to 3×20 third-order, 3×11 recommended. No peer-reviewed printer paper using degree ≥4 found, which strengthens Sec 4.6's degree-4 result as a contribution.

### 6. Direct ΔE00 loss and the cube-root relation — SOUND, missing closest relatives
Finlayson, Mackiewicz, Hurlbert 2015, IEEE TIP 24(5), root-polynomial regression (10.1109/tip.2015.2405336): roots of *input* monomials for exposure invariance, vs this paper's cube root of the *output*. Must be distinguished explicitly or a reviewer will assume they are the same idea. Bianco et al. 2008, JEI 17(4) (10.1117/1.2982004): polynomial coefficients optimised on colour-error statistics for scanners. Gerhardt & Hardeberg 2008; Kucuk et al. 2023.

### 7. IFRA press-to-press variation — SOUND, uncited
**Nussbaum & Hardeberg 2012, CRA 37(2) (10.1002/col.20674)**: eight Norwegian coldset presses; inter-press variation exceeded the difference between custom and standard ICC profiles. Same finding, same substrate class, second author's own institution. Must be cited. Also ISO 12647-3:2013; Wyble & Rich 2007, CRA 32 (10.1002/col.20308); Chen & Urban 2023.

## Overall
Not yet adequate for MDPI Technologies as a related-work section. The experimental work is careful; the positioning cites almost nothing after 2004 and two "first" claims are false. Reviewers from CIC/JIST/CRA will flag Babaei & Hersch, Chen & Berns and Nussbaum & Hardeberg immediately.

**Five most important missing citations:**
1. Babaei & Hersch 2016 (10.1109/tip.2016.2560526)
2. Shi et al. 2018 (10.1145/3272127.3275057) + Chen & Urban 2021 (10.1364/oe.410796): ML already beyond four colorants
3. He et al. 2023 (10.2352/lim.2023.4.1.18): prior XYZ-vs-CIELAB fitting-penalty measurement
4. Nussbaum & Hardeberg 2012 (10.1002/col.20674)
5. Shojaee et al. LLM-SR (10.48550/arXiv.2404.18400) + Vacareanu et al. 2024 (10.48550/arXiv.2404.07544)
Plus Su 2021, Zhan 2025 (direct CMYK→CIELab ML competitors), Molada-Tebar 2019 (GP precedent), Finlayson 2015 (root-polynomial distinction).

Suggested replacement sentences for each point are in the agent transcript; the ones above are enough to write from.
