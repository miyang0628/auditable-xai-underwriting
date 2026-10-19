# Auditable XAI for Insurance Underwriting

Reproducible code for an auditable, counterfactual-explanation pipeline that manages the
**model-risk, interpretability, and regulatory-compliance** exposures of AI-based insurance
underwriting, demonstrated on diabetes risk classification using the Korea National Health and
Nutrition Examination Survey (KNHANES, 2020–2024).

> **Anonymised for peer review.** Author, affiliation, and target-journal identifiers have been
> removed for double-blind submission. Please do not add identifying information to this repository
> until the review process is complete.

---

## What this repository does

Starting from the raw KNHANES survey waves, the notebooks build a sex-stratified diabetes-risk
classifier, calibrate it, identify a high-risk screening cohort, generate individualised
**counterfactual** recourse targets, turn them into applicant-readable consultation guides with an
LLM, and validate guide quality with a reproducible **LLM-as-a-Judge** protocol — then quantify the
actuarial value with a cost–benefit and Monte-Carlo analysis, and map every component to EU and
Korean AI-governance obligations.

## Key findings (reproduced by the notebooks)

- **Sample:** n = 23,104 complete-case adults (9,999 male / 13,105 female); diabetes prevalence
  19.1% / 13.5%.
- **Classifier:** sex-stratified XGBoost, test AUC **0.772 (male)** and **0.821 (female)**; no
  statistically significant AUC difference from other tree ensembles (DeLong), superior to SVM.
  Models are already well calibrated (ECE ≈ 0.01).
- **Screening cohort (cumulative lift ≥ 2):** top 25% (male) / top 40% (female) of the risk
  distribution.
- **Counterfactuals:** the 15 cm waist-circumference cap is enforced *per instance per run*
  (max observed best-CF reduction 14.4 cm). Counterfactuals supply the concrete numeric recourse
  targets that SHAP does not provide in any case (SHAP's top feature is the immutable variable
  *age* in 100% of cases; feature-set overlap Jaccard = 0.115).
- **Honest negative result:** under a **feasibility-aware** rubric, the LLM-judged guides are *not*
  more actionable than feature-attribution baselines (actionability ≈ 1.9/5 for both; paired
  t-tests non-significant). The advantage of counterfactuals here is **structural/normative**
  (they alone furnish recourse targets), not a measured quality gain; target *feasibility* is the
  binding limitation. Guides are strong on association-framing (D2 ≈ 4.6) and disclaimer adequacy
  (D5 ≈ 4.5).
- **Judge design:** the generator (gpt-4o-mini) and the two judge models (gpt-5.4-mini, gpt-5.4)
  are distinct, removing same-model self-evaluation; cross-model agreement is strong on medical
  accuracy (r = 0.809). All three are OpenAI-family, so shared-vendor bias is mitigated but not
  eliminated.
- **Cost–benefit:** base-case loss-ratio improvement ≈ 2.5 pp (averted claims $300K / premium base
  $12.0M over 10 years); Monte-Carlo median 2.8 pp (95% interval 2.1–3.7); net benefit positive in
  100% of 50,000 simulations.

## Pipeline

| # | Notebook | Runs | Output |
|---|----------|------|--------|
| 01 | `01_data_preparation.ipynb` | local/cloud | analytic dataset, prevalence table |
| 02 | `02_eda_sample_characteristics.ipynb` | local/cloud | baseline table, prevalence-by-age figure |
| 03 | `03_model_training_and_selection.ipynb` | local/cloud | models, 7-algorithm comparison, DeLong tests |
| 04 | `04_calibration.ipynb` | local/cloud | calibrated scores, calibration figure |
| 05 | `05_decision_curve_and_lift.ipynb` | local/cloud | DCA/lift figures, high-risk cohort |
| 06 | `06_counterfactual_dice.ipynb` | local/cloud | representative cases, CF distribution |
| 07 | `07_shap_vs_dice.ipynb` | local/cloud | SHAP vs DiCE comparison, eval cases |
| 08 | `08_robustness_inter_model_consistency.ipynb` | local/cloud | inter-model consistency |
| 09 | `09_cost_benefit_and_monte_carlo.ipynb` | local/cloud | scenario table, Monte Carlo, tornado |
| 10 | `10_llm_guide_generation.ipynb` | **local (API key)** | `results/llm/guides.jsonl` |
| 11 | `11_llm_judge_evaluation.ipynb` | **local (API key)** | `results/llm/judge_scores.csv` |
| 12 | `12_quality_analysis.ipynb` | after 10–11 | guide-quality tables & figure |

Notebooks 01–09 require no API key and reproduce the quantitative modelling results. Notebooks
10–11 call the OpenAI API and are run locally with your own key; notebook 12 analyses their output.
Each notebook is a single, self-contained code cell with a fixed seed (`random_state = 42`).

## Repository structure

```
.
├── data/                      # KNHANES .sas7bdat (not redistributed) + derived artifacts
├── notebooks/                 # 01–12, one self-contained cell each
├── results/
│   ├── figures/               # grayscale, dpi 600, PNG + PDF
│   ├── tables/                # CSV tables
│   └── llm/                   # guides + judge scores (generated locally by 10–11)
├── models/                    # trained XGBoost models (generated by 03–04)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Data access

KNHANES microdata are publicly available from the Korea Disease Control and Prevention Agency
(KDCA) at <https://knhanes.kdca.go.kr> under an open-data licence and are **not redistributed**
here. Place the wave files `hn20_all.sas7bdat` … `hn24_all.sas7bdat` in `data/` before running
`01_data_preparation.ipynb`.

**Outcome.** Diabetes = any of: HbA1c ≥ 6.5%, fasting plasma glucose ≥ 126 mg/dL, physician
diagnosis, glucose-lowering medication/insulin, or the KNHANES HbA1c-based diabetes status.
**Predictors (16):** anthropometric (BMI, waist circumference, weight, height), dietary (energy,
carbohydrate, sugar, cholesterol), lifestyle (current smoking, monthly alcohol, aerobic-activity
guideline), and socioeconomic/behavioural (breakfast frequency, education, household income,
self-rated health). The full dictionary with DiCE actionability tiers is written to
`results/tables/predictor_dictionary.csv`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env with your OpenAI API key
jupyter lab                     # run notebooks 01 → 12 in order
```

The API steps use `temperature = 0.3` for guide generation and `temperature = 0.0` for evaluation;
the generator and the two judge models are configured in `.env`. Because the judges differ from the
generator, same-model self-evaluation is avoided; note, however, that all three models are from the
same vendor family, so this mitigates but does not fully eliminate shared-vendor evaluation bias.
Validation with a heterogeneous-architecture judge or a human expert panel remains future work.

## Figures

All figures are generated with seaborn in grayscale at 600 dpi and saved as both `.png` and `.pdf`
without embedded titles or captions, for direct inclusion in the manuscript.

## Citation

> Anonymised for review. A full citation (authors, venue, year, DOI) will be added here after the
> double-blind review process concludes.

## License

Code released under the MIT License (see `LICENSE`). KNHANES data are governed by the KDCA
open-data licence and are not covered by this repository's license.
