# UEFA Shortlist Task Analysis ⚽

Interactive Streamlit app to explore UEFA shortlist data: compute **averages**, inspect **correlations** (by intervals & halves/totals), answer **statistical questions**, and map *Unidentified* matches to reviewed labels via **similarity**.

## ✨ Highlights
- **Global filters**: 7 categories × 4 subsets applied across the app.
- **Averages**: Home vs Away (including xG/PPDA ratio) + per-interval means (Int1..Int6).
- **Correlation**: interval heatmaps, diagonal Int _i_↔Int _i_, totals & halves.
- **Statistical Questions**: paired tests (xG, PPDA), phase leaders, largest H−A gaps, etc.
- **Similarity** (optional): prototype + kNN agreement to label *Unidentified* matches.

---

## 📁 Expected Columns

Minimum useful set (the more you include, the richer the app):

**Intervals**
- `xG_H_Int1..6`, `xG_A_Int1..6`
- `PPDA_H_Int1..6`, `PPDA_A_Int1..6`

**Halves & Totals** (or inferred from intervals)
- `xG_[H/A]_{1stHalf,2ndHalf,Total}`
- `PPDA_[H/A]_{1stHalf,2ndHalf,Total}`

**Labels / Subsets**
- `MatchSuspicionType` ∈ {`both not suspicious`,`home suspicious`,`away suspicious`,`both suspicious`,`unidentified`}
- `Relative quality` ∈ {`home team stronger`,`away team stronger`,`equal strength teams`}

**Dynamic (optional)**
- `xG_H_TotalChange`, `xG_A_TotalChange`, `PPDA_H_TotalChange`, `PPDA_A_TotalChange`

---

## 🧪 Methods (quick notes)

- **Paired t-tests** compare Home vs Away **within the same match**. If means differ but `p ≥ 0.05`, the app shows “No clear difference” (not statistically significant).  
- **Phase spread**: Early (1–2), Mid (3–4), Late (5–6). For xG we use **sums**; for PPDA we use **means**. *Spread* is `max(phase) − min(phase)`.
- **Dynamic “high” flags** (e.g., “Phase spread xG high”): a value is marked **high** if its **absolute value** is **≥ global threshold**. With the default `THRESH_MODE="percentile"` at 75%, that means it’s in the **top 25%** of magnitudes across the whole dataset.

---

## 🚀 Run Locally (macOS/Linux)

1. **Clone** the repo
   ```bash
   git clone https://github.com/<you>/uefa-shortlist-app.git
   cd uefa-shortlist-app
