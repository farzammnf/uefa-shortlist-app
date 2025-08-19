# UEFA Shortlist Task Analysis ⚽

Interactive Streamlit app to explore UEFA shortlist data: compute **averages**, inspect **correlations** (by intervals & halves/totals), answer **statistical questions**, and map *Unidentified* matches to reviewed labels via **similarity**.

## ✨ Highlights
- **Global filters**: 7 categories × 4 subsets applied across the app.
- **Averages**: Home vs Away (including xG/PPDA ratio) + per-interval means (Int1..Int6).
- **Correlation**: interval heatmaps, diagonal Int _i_↔Int _i_, totals & halves.
- **Statistical Questions**: paired tests (xG, PPDA), phase leaders, largest H−A gaps, etc.
- **Similarity** (optional): prototype + kNN agreement to label *Unidentified* matches.


## 🚀 Run Locally (macOS/Linux)

1. **Clone** the repo
   ```bash
   git clone https://github.com/<you>/uefa-shortlist-app.git
   cd uefa-shortlist-app
