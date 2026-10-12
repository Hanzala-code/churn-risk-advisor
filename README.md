# ML-Powered Customer Churn Advisor

> Predicting which telecom customers will churn and turning that prediction 
> into a live web app a retention team can actually use.

---

## Model card: Churn Risk Advisor v1.0

**Intended use:** rank existing telecom customers by churn risk so a
retention team can prioritize calls. Decision support, not automation.

**Not for:** credit scoring, pricing decisions, or any decision that
denies a customer a service.

**Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.

**Model:** XGBClassifier, chosen by 5-fold CV. Features: 30 encoded columns.

**Performance:** CV AUC 0.850 +/- 0.012; test AUC 0.8483 (test set used once).

**Threshold:** 0.20, from a cost analysis (FN = PKR 6,000, FP = PKR 1,000).
t* = 1,000 / 7,000 = 0.14; empirical best = 0.15; operational choice = 0.20.

**Limitations:** one US dataset from one period; no Pakistan telecom data;
associations, not causes; performance may drift as plans and pricing change.
The training-serving skew bug (drop_first on single rows) was found and fixed
with a parity test across all 7,043 customers (largest difference: 0.0).

**Fairness check (Week 5):** compare recall across gender and senior citizens.

**Owner and version:** Hanzala, v1.0, October 2026.

---
## Screenshots:

Individual Result:
<img width="1855" height="981" alt="Screenshot 2026-10-09 185833" src="https://github.com/user-attachments/assets/c956e397-0305-46ae-9e38-341887145563" />

Batch Result:

<img width="1847" height="991" alt="Screenshot 2026-10-09 185959" src="https://github.com/user-attachments/assets/3b57a298-32e7-4e05-b47f-7e17370d71b3" />

---

## Live App

🔗 **[Churn Risk Advisor — Live on Streamlit](https://churn-risk-advisor-hanzala-code.streamlit.app/)**

---
## How to Run Locally

```bash
git clone https://github.com/Hanzala-code/churn-risk-advisor
cd churn-risk-advisor
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```
## Week 1: Exploratory Data Analysis

- Dataset: 7,043 customers, 21 raw features, 26.5% churn rate
- Month-to-month churn rate: 42.7% vs 2.8% on two-year contracts
- Median tenure: churners 10 months vs stayers 38 months
- Key finding: contract type and tenure are the strongest early signals

📓 [Week 1 Notebook — EDA](https://github.com/Hanzala-code/applied-ai-project1-churn/blob/main/week-1-customer-churn-eda.ipynb)

---

## Week 2: Building, Evaluating, and Interpreting ML Models

- Baseline (always "stay"): accuracy 0.735
- Best model: Random Forest, AUC 0.842, recall 0.856 at threshold 0.20
- Top churn drivers (permutation importance): tenure, TotalCharges,
  Contract_Two year
- Threshold chosen: 0.20, because a missed churner costs PKR 6,000 vs
  PKR 1,000 for an unnecessary retention offer — the 6:1 cost ratio makes
  aggressive flagging rational (t* = 0.14 by decision theory)
- Engineered features: n_services, is_new, charge_per_mo, price_jump;
  effect on AUC: 0.8422 → 0.8420 (no improvement — Random Forest already
  captures these interactions internally)
- Biggest lesson: accuracy is a misleading metric on imbalanced data —
  a model that catches zero churners can still score 73.5%

📓 [Week 2 Notebook — ML Models](https://github.com/Hanzala-code/applied-ai-project1-churn/blob/main/week-2-building-ml-models.ipynb)

---

## Week 3: Model Optimization and Unsupervised Learning

- Split-to-split accuracy range across 20 seeds: 0.780 to 0.828
  (std 0.0104, theoretical SE 0.0107, 95% CI ±0.021)

- 5-fold CV AUC:
  - LR  (tuned C=10):    0.846 +/- 0.013
  - RF  (random search): 0.844 +/- 0.011
  - XGBoost (tuned):     0.850 +/- 0.012

- Tuning: best RF params — max_depth=15, min_samples_leaf=15,
  max_features=0.213; random search (24 iterations) in 131s matched
  grid search quality while exploring a wider parameter space

- XGBoost early stopping chose 247 trees out of 2,000;
  best params: learning_rate=0.034, max_depth=2,
  n_estimators=476, reg_lambda=1.974

- Test AUC of final model (XGBoost, used once): 0.8483
  recall 0.521, precision 0.659 — within CV mean ± 2 std ✓

- Customer segments (k=4):
  | Segment | Size | Churn Rate | Action |
  |---|---|---|---|
  | New, High Spenders | 2,157 | 43% | Outreach before month 18 |
  | New, Budget | 1,918 | 32% | Free service bundle for 3 months |
  | Loyal, Heavy Users | 1,938 | 14% | Reward programme, upsell |
  | Loyal, Light Users | 1,030 | 5% | Minimal intervention |

- PCA: 15 of 30 components explain 90% of the variance —
  half the features are redundant, driven by collinear
  "No internet service" dummy columns (all loading at 0.302 on PC1)

- Biggest lesson: a single train/test split can swing ±0.024 accuracy
  on the same model and data — always report CV mean ± std

📓 [Week 3 Notebook — Optimization](https://github.com/Hanzala-code/applied-ai-project1-churn/blob/main/week-3-model-optimization.ipynb)

---

## Check Your Understanding (Week 4)

**Q1. Why does get_dummies(drop_first=True) break predictions on a single row?**

When scoring one customer, only the categories present in that row exist.
drop_first drops the only category seen, making both Contract columns 0 —
which the model reads as Month-to-month. This silently doubled the churn
probability for a Two year contract customer (0.012 → 0.136). The fix is
get_dummies without drop_first at serving, then reindex to training columns.

**Q2. What does the parity test compare, and why every row?**

It compares p_train (training-time predictions using X[cols]) against
p_serve (serving-time predictions using prepare_input). It must cover all
7,043 rows because the encoding bug could affect only specific category
combinations — testing a handful might miss the exact trigger. Our test
confirmed largest difference = 0.0, proving both paths are identical.

**Q3. st.cache_resource or st.cache_data for the model? Why?**

st.cache_resource — it is designed for shared objects like ML models that
should exist as one instance in memory across all sessions. st.cache_data
would create a separate model copy per user session, wasting memory and
slowing startup. The model loads once at server start and is reused.

**Q4. Why derive TotalCharges instead of asking the user to type it?**

TotalCharges = tenure × MonthlyCharges is a derived quantity. Asking users
to type it risks inconsistent inputs (e.g. TotalCharges = 5,000 with only
2 months tenure). Deriving it guarantees internal consistency and matches
how the training data was structured, reducing inputs and preventing errors.

**Q5. Why pin the exact scikit-learn version in requirements.txt?**

Scikit-learn's serialisation format can change between versions. A model
trained on 1.3.2 may load silently on 1.4.0 but produce different
predictions due to internal changes. Pinning the exact version ensures the
deployment server is identical to the training environment — which our
parity test (largest difference: 0.0) confirmed only holds when versions match.

**Q6. What-if shows two-year contract lowers risk 0.935 → 0.765. Would offering one make the customer stay?**

Not necessarily. This is a correlation the model learned from historical
data, not a causal effect. Committed customers accept long contracts —
the contract did not cause loyalty, loyalty caused the contract acceptance.
To find out, run a randomised controlled trial: offer contract upgrades to
a random sample of high-risk customers, withhold from a control group, and
compare 6-month actual churn rates.

---




