# Paper audit (29 Aug 2026) — references

All 23 active entries in `MDPI-Phil-Journal/refs.bib` checked against primary sources: Crossref API per DOI, OpenAlex/Semantic Scholar, publisher pages (SPIE via browser), Zenodo API, the actual AIC 2025 proceedings PDF (downloaded from aic-color.org), and the live URLs. No file was edited.

**Headline: no hallucinated entries. One materially wrong entry (pang2024standardization). Two citation-to-claim misfits (Cheung 2004 used for claims it does not make). One citation-to-claim fit unverified (Bala 2003).**

## Per-entry verdicts

| key | verdict | checked | detail |
|---|---|---|---|
| bala2003device | OK | Crossref 10.1201/9781420041484-5 | "Device characterization", Digital Color Imaging Handbook, pp. 269-384 confirmed. Crossref author "Balasubramanian" (same person). Optional: add DOI. |
| breiman1984cart | OK (standard, no DOI) | — | Wadsworth 1984. Fine. |
| breiman2001randomforests | OK | Crossref 10.1023/A:1010933404324 | Exact match. |
| cheung2004comparative | OK | Crossref + full PDF | Coloration Technology 120(1):19-25. Exact match (but see claim-fit below). |
| cortes1995support | OK | Crossref 10.1007/BF00994018 | Exact match. |
| cover1967nearest | OK | Crossref 10.1109/TIT.1967.1053964 | Exact match. |
| fogra2020 | OK | fogra.org URL loads | FOGRA51-FOGRA52 archive V1.0, 23.09.2020. Matches. |
| friedman2001greedy | OK | Crossref + Semantic Scholar | Ann. Stat. 29(5):1189-1232. Confirmed. |
| geladi1986partial | OK | Crossref 10.1016/0003-2670(86)80028-9 | Exact match. |
| hoerl1970ridge | OK | Crossref 10.1080/00401706.1970.10488634 | Exact match. |
| kang1992neural | OK | Crossref + SPIE page | Kang & Anderson, JEI 1(2):125-135, 1992. Bib correct (Crossref metadata incomplete). |
| massy1965principal | OK | Crossref 10.1080/01621459.1965.10480787 | Exact match. |
| pang2024standardization | **WRONG** | committee.iso.org case-studies page (200); bib URL with trailing slash → **404** | Author is **Brenda Pang** (APTEC), not "Bob Pang". Exact title "How to adopt printing standardisation" (no "ISO/TC 130 Case Study"). Published 2 Sep 2024 (bib gives that as access date). Fix author, title, URL (`.html`), dates. |
| rasmussen2006gaussian | OK (minor) | Crossref 10.7551/mitpress/3206.001.0001 | Online 2005, print 2006. Fine. |
| sharma2005ciede2000 | OK (minor) | Crossref 10.1002/col.20070 | Online-first 2004, print 30(1) Feb 2005. Standard citation 2005. Fine. |
| tibshirani1996regression | OK | Crossref | Exact match. |
| tominaga1993neural | OK | Crossref 10.2352/CIC.1993.1.1.art00043 | CIC 1993 vol 1 pp 173-177. Exact match. |
| zou2005regularization | OK | Crossref | Exact match. |
| zafar2025aic | OK | Proceedings PDF front matter, TOC, page footers; aic2025.org | ISBN 978-0-6484724-7-6 confirmed (ISSN 2617-2410). Organised by Color Association of Taiwan, published by AIC. Pages 59-64 confirmed. Official title "Proceedings of the 16th Congress of the International Colour Association (AIC) 2025, Taipei, Taiwan"; "Color for Future" is the congress theme. Acceptable. |
| deshpande2014ncolour | OK | Crossref 10.1364/OE.22.031786 + Semantic Scholar | Opt. Express 22(26):31786-31800. Exact match. |
| mansencal2024colour | OK | Zenodo API record 13917514 | DOI = Colour 0.4.6, 2024-10-11. First seven creators exactly as listed, then 41 more; "and others" correct. |
| chardata2026 | OK | chardata.colourbill.com (browser; 403 to plain fetchers) | Title matches; "© 2026 William Li"; latest release 1.18.0, 16 Jul 2026, matches v1.18.0 in main.tex. |
| aptec2023pc | OK, **but note** | registry.color.org aptec_coated_cardboard and aptec_ccnb | Both load. PC10 "APTEC_PC10_CardBoard_2023_v1", PC11 "APTEC PC11_CCNB 2023 v1", July 2023. **Registry describes BOTH PC10 and PC11 substrates as clay coated news back 250 g/m² (PS 11)**, while Table 1 labels PC10 "Cardboard" and PC11 "Coated paper (CCNB)". Authors should check. (Relates to the known PC10-header-says-PC11 data question.) |
| icc2025chardata | OK | registry.color.org aptec_cmykogvcoated | "APTEC_CMYKOGVCoated", July 2025, premium coated, M1 white backing. Matches. |

## Citation-to-claim fit

- (a) kang1992neural, tominaga1993neural — "NNs for printer/scanner characterization since early 1990s": **supported**.
- (b) cheung2004comparative, l.77 "comparable or **modestly better** accuracy for the learned model": **NOT supported**. Paper reports median ΔE 2.57 (polynomial) vs 2.89 (NN), calls them "approximately the same", and recommends polynomials. Reword to "comparable accuracy, with the polynomial recommended on practicality".
  l.79 "evaluate, and commonly **train**, in CIELAB": **half supported**. They evaluate in CIELAB ΔE but both models map RGB → **XYZ**. Drop "and commonly train" for this citation.
- (c) bala2003device, l.79 "conventionally posed with target in a perceptually uniform space": **could not verify** (chapter text not accessible to the agent). Needs a check against pp. 269-384, or swap for Hung 1993 / Vrhel & Trussell 1999 (below).
- (d) pang2024standardization, l.83 "broader push toward printing-process standardization": weak fit. One-page APTEC case study on G7/PSO adoption; says nothing about LUTs or extended gamut. Cite ISO 12647-2 / ISO/TC 130 instead, or drop the sentence.
- (e) deshpande2014ncolour: **supported** (SCOP extended to generic n-colour forward model, demonstrated on 7-ink litho).
- (f) sharma2005ciede2000 as "CIEDE2000, current standard": **supported** as implementation reference; formula itself is CIE 142-2001 / ISO/CIE 11664-6 (could add).
- (g) massy1965principal for PCR: **supported**.
- (h) mansencal2024colour: **supported**.

## Missing citations, with DOIs confirmed in Crossref

1. "Fitting in a perceptually uniform space is standard practice" (l.69, 427):
   - Hung, P.-C. "Colorimetric calibration in electronic imaging devices using a look-up-table model and interpolations." J. Electronic Imaging 2(1):53, 1993. DOI 10.1117/12.132391
   - Vrhel, M. J.; Trussell, H. J. "Color device calibration: a mathematical formulation." IEEE Trans. Image Process. 8(12):1796-1806, 1999. DOI 10.1109/83.806624
2. "LUT node count grows combinatorially" (l.65):
   - Bala, R.; Klassen, R. V. "Efficient color transformation implementation." Digital Color Imaging Handbook pp. 687-725. DOI 10.1201/9781420041484-11
3. "The ICC characterization brief defines the classical baseline as degree-3 on XYZ" (l.500, 583): **no public ICC document by that name found.** Cite as unpublished / private communication, or cite the actual document meant.
4. "CMYKOGV used commercially to extend gamut and reduce metamerism" (l.65):
   - Boll, H. "Color-to-colorant transformation for a seven ink process." Proc. SPIE 2170:108-118, 1994. DOI 10.1117/12.173839
   - Deshpande, Green, Pointer. "Gamut evaluation of an n-colour printing process with the minimum number of measurements." Color Res. Appl. 40(4):408-415, 2015. DOI 10.1002/col.21909
   - Deshpande & Green. "A simplified method of predicting the colorimetry of spot color overprints." CIC 18:213-216, 2010. DOI 10.2352/cic.2010.18.1.art00037
5. "GP O(N³)" (l.589): rasmussen2006gaussian Ch. 8 already covers it; optional Quiñonero-Candela, Rasmussen, Williams 2007, DOI 10.7551/mitpress/7496.003.0011.
6. Optional for l.77: Vrhel & Trussell, "Color scanner calibration via a neural network," ICASSP 1999. DOI 10.1109/icassp.1999.757588

## Action items
1. Fix `pang2024standardization` (author, title, URL, dates) and reconsider whether it belongs at l.83 at all.
2. Reword both Cheung 2004 claims.
3. Decide how to cite the "ICC characterization brief".
4. Verify or replace the Bala 2003 claim.
5. Check the PC10/PC11 substrate labels in Table 1 against the registry.
