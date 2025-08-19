# app.py — UEFA Shortlist Task Analysis (single-file, dynamic subset baseline)
# Run: streamlit run app.py

import os, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- THEME / CONSTANTS ----------
APP_TITLE = "UEFA Shortlist Task Analysis"
APP_ICON = "⚽"
DEFAULT_PATH = "/Users/farzammanafzadeh/Desktop/Uefa Task/Streamlit.csv"

# soft, opposing palette (not harsh) — keep your colors
COL_HOME = "#3B82F6"     # soft blue
COL_AWAY = "#D4A373"     # muted amber
COL_BASE = "#1F2937"     # dark gray (generic)
# Baseline overlay colors (distinct per side)
COL_BASE_HOME = "#0D47A1"  # dark blue (home baseline)
COL_BASE_AWAY = "#E65100"  # dark orange (away baseline)

HEATMAP_SCALE = "Blues"

CATEGORIES = [
    "All Matches",
    "Clean (Both Not Suspicious)",
    "Suspicious (Any of Home/Away/Both)",
    "Home Suspicious",
    "Away Suspicious",
    "Both Suspicious",
    "Unidentified",
]
SUBSETS = ["Overall", "Home Stronger", "Away Stronger", "Equal Strength"]

PPDA_H_INT = [f"PPDA_H_Int{i}" for i in range(1,7)]
PPDA_A_INT = [f"PPDA_A_Int{i}" for i in range(1,7)]
XG_H_INT   = [f"xG_H_Int{i}"   for i in range(1,7)]
XG_A_INT   = [f"xG_A_Int{i}"   for i in range(1,7)]
HALF_TOTAL = [
    "PPDA_H_1stHalf","PPDA_H_2ndHalf","PPDA_H_Total",
    "PPDA_A_1stHalf","PPDA_A_2ndHalf","PPDA_A_Total",
    "xG_H_1stHalf","xG_H_2ndHalf","xG_H_Total",
    "xG_A_1stHalf","xG_A_2ndHalf","xG_A_Total",
]
REQ_AVG = [
    "xG_H_Total","xG_A_Total","PPDA_H_Total","PPDA_A_Total",
    "PPDA_H_1stHalf","PPDA_H_2ndHalf","PPDA_A_1stHalf","PPDA_A_2ndHalf",
]

# ---------- WHITE BACKGROUND + keep your UI accents ----------
CSS = """
<style>
  .stApp { background-color: #FFFFFF; } /* all white */
  .block-container { padding-top: 2rem; } /* move title slightly down */
  .card { padding:1rem; border:1px solid #e5e7eb; border-radius:14px; background:#ffffff; margin:.25rem 0 1rem; }
  :root { --home:#3B82F6; --away:#D4A373; --base:#1F2937; }
  .badge {display:inline-block; padding:.2rem .55rem; border-radius:999px;
          background:#EEF2FF; color:#1E3A8A; font-weight:600; margin:.1rem .35rem .1rem 0; font-size:.85rem;}
  .badge.away { background:#FFF1E6; color:#7C5A34; }
  .badge.base { background:#F3F4F6; color:#111827; }
  h1, h2, h3 { line-height:1.2; }
  .big-title { font-size: 2.6rem; font-weight: 800; margin: 1rem 0 .25rem 0; }
</style>
"""

# ---------- SMALL HELPERS ----------
def safe_num(obj):
    if isinstance(obj, pd.DataFrame): return obj.apply(pd.to_numeric, errors="coerce")
    return pd.to_numeric(obj, errors="coerce")

def exists_all(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

def diag_ylim(arrs, symmetric=True, min_span=0.2, pad=0.02):
    vals = []
    for a in arrs:
        if a is None: continue
        a = np.asarray(a, dtype=float)
        vals.extend([v for v in a if np.isfinite(v)])
    if not vals: return (-1.0, 1.0)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if symmetric:
        m = max(abs(vmin), abs(vmax), min_span/2) + pad
        return (-m, m)
    span = vmax - vmin
    if span < min_span:
        mid = (vmin+vmax)/2; half = max(min_span/2, span/2)+pad
        return (mid-half, mid+half)
    return (vmin-pad, vmax+pad)

# --- helpers for Statistical Questions ---
def _has_cols(df, cols):
    return all(c in df.columns for c in cols)

def _phase_sums(df, xg_prefix, ppda_prefix, side):  # side = "H" or "A"
    """Return per-match Early(1–2), Mid(3–4), Late(5–6) sums for xG and means for PPDA."""
    xg_int = [f"{xg_prefix}_{side}_Int{i}" for i in range(1,7)]
    pp_int = [f"{ppda_prefix}_{side}_Int{i}" for i in range(1,7)]
    if not _has_cols(df, xg_int + pp_int):
        return None
    E_xg = safe_num(df[[xg_int[0], xg_int[1]]]).sum(axis=1)
    M_xg = safe_num(df[[xg_int[2], xg_int[3]]]).sum(axis=1)
    L_xg = safe_num(df[[xg_int[4], xg_int[5]]]).sum(axis=1)
    E_pp = safe_num(df[[pp_int[0], pp_int[1]]]).mean(axis=1)
    M_pp = safe_num(df[[pp_int[2], pp_int[3]]]).mean(axis=1)
    L_pp = safe_num(df[[pp_int[4], pp_int[5]]]).mean(axis=1)
    return {"xg": {"Early":E_xg, "Mid":M_xg, "Late":L_xg},
            "pp": {"Early":E_pp, "Mid":M_pp, "Late":L_pp}}

def _std_betas_two_predictors(y, x1, x2):
    """Standardized multiple regression (lightweight): betas and R²."""
    d = pd.DataFrame({"y":safe_num(y), "x1":safe_num(x1), "x2":safe_num(x2)}).dropna()
    if len(d) < 10:
        return np.nan, np.nan, np.nan
    z = (d - d.mean())/d.std(ddof=1)
    X = z[["x1","x2"]].values
    Y = z["y"].values
    Xc = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    y_hat = Xc @ coef
    ssr = np.sum((y_hat - Y.mean())**2)
    sst = np.sum((Y - Y.mean())**2) + 1e-12
    r2 = ssr/sst
    beta1, beta2 = float(coef[1]), float(coef[2])
    return beta1, beta2, float(r2)

def _ratio_side(dfs, side):
    """xG/PPDA for H or A. Uses precomputed column if present; else computes."""
    col = f"xG_{side}_to_PPDA"
    if col in dfs.columns:
        return safe_num(dfs[col])
    need = [f"xG_{side}_Total", f"PPDA_{side}_1stHalf", f"PPDA_{side}_2ndHalf"]
    if _has_cols(dfs, need):
        return safe_num(dfs[f"xG_{side}_Total"]) / safe_num(dfs[[f"PPDA_{side}_1stHalf", f"PPDA_{side}_2ndHalf"]]).mean(axis=1)
    return pd.Series(dtype=float)

# ---------- DATA LOADING & FILTERING ----------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_or_upload():
    if "df" in st.session_state and isinstance(st.session_state["df"], pd.DataFrame):
        return st.session_state["df"], "✅ Using in-memory DataFrame"
    if os.path.isfile(DEFAULT_PATH):
        return load_csv(DEFAULT_PATH), "✅ Loaded default CSV"
    up = st.sidebar.file_uploader("📥 Upload CSV", type=["csv"])
    if up: return load_csv(up), "✅ Loaded uploaded CSV"
    return None, "❌ No data — upload a CSV"

# ---- label/subset filters ----
def _col_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        s = df[col]
        return s.reindex(df.index) if not s.index.equals(df.index) else s
    return pd.Series("", index=df.index, dtype="object")

def cat_mask(df: pd.DataFrame, category: str) -> pd.Series:
    ms = _col_or_blank(df, "MatchSuspicionType")
    if category == "All Matches": return pd.Series(True, index=df.index)
    if category == "Clean (Both Not Suspicious)": return ms.eq("both not suspicious")
    if category == "Suspicious (Any of Home/Away/Both)": return ms.isin(["home suspicious","away suspicious","both suspicious"])
    if category == "Home Suspicious": return ms.eq("home suspicious")
    if category == "Away Suspicious": return ms.eq("away suspicious")
    if category == "Both Suspicious": return ms.eq("both suspicious")
    if category == "Unidentified": return ms.eq("unidentified")
    return pd.Series(False, index=df.index)

def sub_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    rq = _col_or_blank(df, "Relative quality")
    if subset == "Overall": return pd.Series(True, index=df.index)
    if subset == "Home Stronger": return rq.eq("home team stronger")
    if subset == "Away Stronger": return rq.eq("away team stronger")
    if subset == "Equal Strength": return rq.eq("equal strength teams")
    return pd.Series(False, index=df.index)

def filter_data(df: pd.DataFrame, category: str, subset: str) -> pd.DataFrame:
    m_cat = cat_mask(df, category)
    d = df[m_cat.fillna(False)]
    m_sub = sub_mask(d, subset)
    return d[m_sub.fillna(False)]

# ---------- AVERAGES ----------
def _ratio_series(dfs: pd.DataFrame, side: str) -> pd.Series:
    pre = f"xG_{side}_to_PPDA"
    if pre in dfs.columns: return safe_num(dfs[pre])
    xg_t = safe_num(dfs.get(f"xG_{side}_Total", np.nan))
    p1   = safe_num(dfs.get(f"PPDA_{side}_1stHalf", np.nan))
    p2   = safe_num(dfs.get(f"PPDA_{side}_2ndHalf", np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        return xg_t / ((p1 + p2) / 2.0)

def compute_avg(dfs: pd.DataFrame) -> pd.DataFrame:
    if dfs.empty: return pd.DataFrame()
    xg_h = safe_num(dfs["xG_H_Total"]).mean() if "xG_H_Total" in dfs.columns else np.nan
    xg_a = safe_num(dfs["xG_A_Total"]).mean() if "xG_A_Total" in dfs.columns else np.nan
    pp_h = safe_num(dfs["PPDA_H_Total"]).mean() if "PPDA_H_Total" in dfs.columns else np.nan
    pp_a = safe_num(dfs["PPDA_A_Total"]).mean() if "PPDA_A_Total" in dfs.columns else np.nan
    r_h  = _ratio_series(dfs, "H").mean() if any(c.startswith("PPDA_H_") or c=="xG_H_Total" for c in dfs.columns) else np.nan
    r_a  = _ratio_series(dfs, "A").mean() if any(c.startswith("PPDA_A_") or c=="xG_A_Total" for c in dfs.columns) else np.nan
    out = pd.DataFrame({"Metric":["xG","PPDA","xG/PPDA"], "Home":[xg_h,pp_h,r_h], "Away":[xg_a,pp_a,r_a]})
    out["H−A"] = out["Home"] - out["Away"]
    return out

def interval_means_table(dfs: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy table of per-interval means for xG and PPDA (Home/Away)."""
    have_xg = exists_all(dfs, XG_H_INT + XG_A_INT)
    have_pp = exists_all(dfs, PPDA_H_INT + PPDA_A_INT)
    if (not have_xg) and (not have_pp):
        return pd.DataFrame()

    rows = []
    ints = [f"Int {i}" for i in range(1,7)]
    if have_xg:
        xgH = safe_num(dfs[XG_H_INT]).mean()
        xgA = safe_num(dfs[XG_A_INT]).mean()
        for i, label in enumerate(ints, start=1):
            rows.append({"Metric":"xG", "Interval":label, "Home":float(xgH.iloc[i-1]), "Away":float(xgA.iloc[i-1])})
    if have_pp:
        ppH = safe_num(dfs[PPDA_H_INT]).mean()
        ppA = safe_num(dfs[PPDA_A_INT]).mean()
        for i, label in enumerate(ints, start=1):
            rows.append({"Metric":"PPDA", "Interval":label, "Home":float(ppH.iloc[i-1]), "Away":float(ppA.iloc[i-1])})
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["H−A"] = df_out["Home"] - df_out["Away"]
    return df_out

def bars_home_away(df_metrics: pd.DataFrame, title: str):
    tidy = df_metrics.melt(id_vars="Metric", value_vars=["Home","Away"], var_name="Side", value_name="Value")
    fig = px.bar(tidy, x="Metric", y="Value", color="Side", barmode="group", title=title,
                 color_discrete_map={"Home":COL_HOME, "Away":COL_AWAY}, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------- CORRELATIONS ----------
def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    a, b = safe_num(a), safe_num(b)
    m = a.notna() & b.notna()
    return float(a[m].corr(b[m])) if m.sum() >= 3 else np.nan

def interval_matrix(dfs: pd.DataFrame, home=True) -> pd.DataFrame:
    rows = PPDA_H_INT if home else PPDA_A_INT
    cols = XG_H_INT if home else XG_A_INT
    if not exists_all(dfs, rows+cols): return pd.DataFrame(index=rows, columns=cols, data=np.nan)
    corr = dfs[rows+cols].corr()
    return corr.loc[rows, cols].round(3)

def diag_series(dfs: pd.DataFrame, home=True) -> pd.Series:
    rows = PPDA_H_INT if home else PPDA_A_INT
    cols = XG_H_INT if home else XG_A_INT
    vals = []
    for i in range(6):
        vals.append(_safe_corr(dfs.get(rows[i], pd.Series(dtype=float)),
                               dfs.get(cols[i], pd.Series(dtype=float))))
    return pd.Series(vals, index=[1,2,3,4,5,6]).round(3)

def totals_halves_table(dfs: pd.DataFrame) -> pd.DataFrame:
    if not exists_all(dfs, HALF_TOTAL): return pd.DataFrame()
    data = {
        "Home: Total": _safe_corr(dfs["PPDA_H_Total"], dfs["xG_H_Total"]),
        "Away: Total": _safe_corr(dfs["PPDA_A_Total"], dfs["xG_A_Total"]),
        "Home: 1st Half": _safe_corr(dfs["PPDA_H_1stHalf"], dfs["xG_H_1stHalf"]),
        "Home: 2nd Half": _safe_corr(dfs["PPDA_H_2ndHalf"], dfs["xG_H_2ndHalf"]),
        "Away: 1st Half": _safe_corr(dfs["PPDA_A_1stHalf"], dfs["xG_A_1stHalf"]),
        "Away: 2nd Half": _safe_corr(dfs["PPDA_A_2ndHalf"], dfs["xG_A_2ndHalf"]),
    }
    return pd.DataFrame(data, index=["r"]).T.round(3)

def top_corr_signed(dfs: pd.DataFrame, target: str, topn=10) -> pd.Series:
    num = dfs.select_dtypes(include="number")
    if target not in num.columns or len(num)<3: return pd.Series(dtype=float)
    cols = [c for c in num.columns if (("xG" not in c) or (c == target))]
    s = num[cols].corr()[target].drop(labels=[target], errors="ignore").dropna()
    return s.reindex(s.abs().sort_values(ascending=False).index).head(topn).round(3)

def heatmap(df_mat: pd.DataFrame, title: str, row_prefix="PPDA", col_prefix="xG"):
    if df_mat.empty: st.info("Not enough data for this matrix."); return
    mat = df_mat.copy()
    mat.index = [f"{row_prefix} Int {i}" for i in range(1,7)]
    mat.columns = [f"{col_prefix} Int {i}" for i in range(1,7)]
    fig = px.imshow(mat, text_auto=True, aspect="auto", title=title,
                    color_continuous_scale=HEATMAP_SCALE, template="plotly_white")
    fig.update_xaxes(title_text=f"{col_prefix} intervals")
    fig.update_yaxes(title_text=f"{row_prefix} intervals")
    st.plotly_chart(fig, use_container_width=True)

def plot_diag_vs_baseline(home_sel, home_base, away_sel, away_base,
                          title="Diagonal — Selection vs Baseline",
                          baseline_name="Baseline"):
    x = list(range(1,7)); lab = [f"Int {i}" for i in x]
    y_min, y_max = diag_ylim([home_sel, home_base, away_sel, away_base], symmetric=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Home","Away"), shared_yaxes=True)

    # Baseline lines (distinct colors per side)
    fig.add_trace(go.Scatter(x=x, y=home_base, mode="lines+markers", name=baseline_name,
                             line=dict(color=COL_BASE_HOME, width=2, dash="dash"), marker=dict(size=7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=home_sel,  mode="lines+markers+text", name="Selection",
                             line=dict(color=COL_HOME, width=3), marker=dict(size=8),
                             text=[f"{v:.2f}" if np.isfinite(v) else "" for v in home_sel],
                             textposition="top center"), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=away_base, mode="lines+markers", showlegend=False,
                             line=dict(color=COL_BASE_AWAY, width=2, dash="dash"), marker=dict(size=7)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=away_sel,  mode="lines+markers+text", showlegend=False,
                             line=dict(color=COL_AWAY, width=3), marker=dict(size=8),
                             text=[f"{v:.2f}" if np.isfinite(v) else "" for v in away_sel],
                             textposition="top center"), row=1, col=2)

    for c in (1,2):
        fig.update_xaxes(title_text="Intervals", tickmode="array", tickvals=x, ticktext=lab, row=1, col=c)
    fig.update_yaxes(title_text="Correlation (r)", range=[y_min, y_max], zeroline=True, zerolinecolor="#9CA3AF")
    fig.update_layout(title=title, template="plotly_white", height=420, margin=dict(l=40,r=20,t=60,b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True)

def plot_totals_halves_selection(th_sel: pd.DataFrame, title="Totals & Halves — Selection"):
    if th_sel is None or th_sel.empty:
        st.info("Totals/halves columns not found.")
        return

    def _fetch(df, idx):
        if idx not in df.index:
            return np.nan
        row = df.loc[idx]
        if isinstance(row, pd.Series):
            if "r" in row.index:
                v = row["r"]
            else:
                v = row.iloc[0] if len(row) else np.nan
        else:
            v = row
        try:
            return float(v)
        except Exception:
            return np.nan

    cats = ["Total", "1st Half", "2nd Half"]
    hv = [_fetch(th_sel, "Home: Total"), _fetch(th_sel, "Home: 1st Half"), _fetch(th_sel, "Home: 2nd Half")]
    av = [_fetch(th_sel, "Away: Total"), _fetch(th_sel, "Away: 1st Half"), _fetch(th_sel, "Away: 2nd Half")]

    fig = go.Figure()
    fig.add_bar(x=cats, y=hv, name="Home", marker_color=COL_HOME)
    fig.add_bar(x=cats, y=av, name="Away", marker_color=COL_AWAY)

    y_min, y_max = diag_ylim([hv + av], symmetric=True)
    fig.update_yaxes(title_text="Correlation (r)", range=[y_min, y_max], zeroline=True, zerolinecolor="#9CA3AF")
    fig.update_layout(
        title=title, template="plotly_white", barmode="group", height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- STATS TESTS ----------
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def paired_t(a, b):
    dfp = pd.DataFrame({"a":safe_num(a), "b":safe_num(b)}).dropna(); n=len(dfp)
    if n<3: return float("nan"), float("nan"), float("nan"), n, float("nan"), float("nan")
    diff = dfp["a"]-dfp["b"]; md=diff.mean(); sd=diff.std(ddof=1)
    t = md/(sd/np.sqrt(n)) if sd>0 else float("inf"); dfree=n-1
    p = 2*stats.t.sf(abs(t), dfree) if HAS_SCIPY else 2*(1-0.5*(1+math.erf(abs(t)/np.sqrt(2))))
    d = md/(sd+1e-12)
    return float(t), float(dfree), float(p), n, float(md), float(d)

def welch(a, b):
    x = safe_num(a).dropna()
    y = safe_num(b).dropna()
    n1, n2 = len(x), len(y)
    if min(n1, n2) < 3:
        return float("nan"), float("nan"), float("nan"), n1, n2, float("nan")

    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)

    denom = np.sqrt(v1/n1 + v2/n2)
    t = (m1 - m2) / denom if denom > 0 else float("inf")

    dfree = ((v1/n1 + v2/n2)**2) / (
        (v1**2) / ((n1**2) * (n1 - 1)) +
        (v2**2) / ((n2**2) * (n2 - 1))
    )

    if HAS_SCIPY:
        p = 2 * stats.t.sf(abs(t), dfree)
    else:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / np.sqrt(2))))

    return float(t), float(dfree), float(p), n1, n2, float(m1 - m2)

def pearson(x, y):
    d = pd.DataFrame({"x":safe_num(x), "y":safe_num(y)}).dropna(); n=len(d)
    if n<3: return float("nan"), n, float("nan"), float("nan"), float("nan")
    r = d["x"].corr(d["y"])
    t = r*np.sqrt((n-2)/max(1e-12,1-r**2))
    if HAS_SCIPY:
        p = 2*stats.t.sf(abs(t), n-2); zcrit=stats.norm.ppf(0.975)
    else:
        p = 2*(1-0.5*(1+math.erf(abs(t)/np.sqrt(2)))); zcrit=1.96
    z = 0.5*np.log((1+r)/(1-r)); se = 1/np.sqrt(n-3)
    lo, hi = z - zcrit*se, z + zcrit*se
    r_lo = (np.exp(2*lo)-1)/(np.exp(2*lo)+1); r_hi = (np.exp(2*hi)-1)/(np.exp(2*hi)+1)
    return float(r), n, float(p), float(r_lo), float(r_hi)

# ========================= APP =========================
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

df, status = load_or_upload()
if df is None:
    st.error(status); st.stop()

st.sidebar.markdown("## ⚙️ Filters")
st.sidebar.caption("Apply to all tabs")
st.sidebar.write(status)
cat = st.sidebar.selectbox("Category", CATEGORIES, index=0)
sub = st.sidebar.selectbox("Subset", SUBSETS, index=0)

# Selection and dynamic baseline:
df_sel  = filter_data(df, cat, sub)
df_base = filter_data(df, "All Matches", sub)   # Baseline = All Matches × current Subset
BASELINE_LABEL = f"Baseline (All Matches × {sub})"

st.sidebar.metric("Rows (filter)", len(df_sel))
st.sidebar.metric("Columns", df.shape[1])

# ---- Title: bigger, moved down, and no caption below ----
st.markdown(f'<h1 class="big-title">{APP_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)

# Top badges (removed the "Theme: Blue" badge)
st.markdown(
    f'<span class="badge">Category: {cat}</span>'
    f'<span class="badge away">Subset: {sub}</span>',
    unsafe_allow_html=True,
)

# ---------- TABS (Reference removed) ----------
tab1, tab2, tab3, tab4, tab7, tab6 = st.tabs(
    ["📦 General", "📝 Description", "📈 Average", "🔗 Correlation", "🧪 Stat Tests", "🧬 Similarity"]
)

# --- General ---
with tab1:
    st.markdown("### 👀 Quick view")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Rows (all)", len(df))
    with c2: st.metric("Rows (filter)", len(df_sel))
    with c3: st.metric("Numeric cols", df.select_dtypes(include=[np.number]).shape[1])
    with c4: st.metric("Missing cells", int(df.isna().sum().sum()))
    st.markdown("### 🗂️ Data sample")
    st.dataframe(df_sel.head(50), use_container_width=True, hide_index=True, height=360)

    st.markdown("### 📊 Distribution")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        c1,c2 = st.columns([2,1])
        with c1: col = st.selectbox("Column", options=num_cols, index=0)
        with c2: bins = st.slider("Bins", 10, 100, 40, 5)
        density = st.checkbox("Show density", value=True)
        marginal = st.selectbox("Side plot", ["None","box","violin","rug"], index=0)
        histnorm = "probability density" if density else None
        marginal = None if marginal=="None" else marginal
        fig = px.histogram(df_sel, x=col, nbins=bins, histnorm=histnorm, marginal=marginal,
                           title=f"Distribution — {col}", color_discrete_sequence=[COL_HOME], template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# --- Description ---
with tab2:
    st.markdown("### 📝 What this app shows")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("- **7 categories** × **4 subsets** apply across all analyses.")
    st.markdown("- **Averages**: xG, PPDA, xG/PPDA for Home/Away and H−A; plus per-interval means (Int1–6).")
    st.markdown(f"- **Correlation**: PPDA ↔ xG interval matrices (Home/Away) ")
    st.markdown("- **Statistical tests**: 8 focused questions summarized with clear ‘tends to’ direction and significance.")
    st.markdown("- **Similarity**: maps *Unidentified* matches to reviewed labels using centroid similarity (optionally kNN).")
    # Removed the baseline color badge row here on purpose
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 🏷️ Labels")
    lab = pd.DataFrame([
        {"Label":"🟢 Not Suspicious","Definition":"Reviewed; no deliberate underperformance.","Reviewed?":"✅ Yes","Abnormal Betting?":"❌ No","Certainty":"✅ High","Implication":"Treat as clean (baseline)."},
        {"Label":"🔴 Suspicious","Definition":"Reviewed; abnormal betting suggests deliberate underperformance.","Reviewed?":"✅ Yes","Abnormal Betting?":"✅ Yes","Certainty":"✅ High","Implication":"Flagged for manipulation."},
        {"Label":"⚪ Unidentified","Definition":"Not reviewed; no clear betting signal.","Reviewed?":"❌ No","Abnormal Betting?":"❌ No / weak","Certainty":"❓ Low","Implication":"Possibly clean; use with caution."},
    ])
    st.dataframe(lab, use_container_width=True, hide_index=True)

# --- Average ---
with tab3:
    st.markdown(f"### 📈 Averages — {cat} · {sub}")
    tbl = compute_avg(df_sel)
    if tbl.empty:
        st.info("Required columns for averages not found.")
    else:
        m1,m2,m3 = st.columns(3)
        with m1: st.metric("xG (H−A)", f"{tbl.loc[tbl['Metric']=='xG','H−A'].iloc[0]:.3f}")
        with m2: st.metric("PPDA (H−A)", f"{tbl.loc[tbl['Metric']=='PPDA','H−A'].iloc[0]:.3f}")
        with m3: st.metric("xG/PPDA (H−A)", f"{tbl.loc[tbl['Metric']=='xG/PPDA','H−A'].iloc[0]:.3f}")
        st.dataframe(tbl.round(3), use_container_width=True, hide_index=True)
        st.markdown("#### 🎨 Home vs Away")
        bars_home_away(tbl, "Averages — Home vs Away")

    st.markdown("---")
    st.markdown("### ⏱️ By-interval averages (means)")
    df_int_sel  = interval_means_table(df_sel)
    df_int_base = interval_means_table(df_base)  # Dynamic baseline: All Matches × current Subset
    if df_int_sel.empty or df_int_base.empty:
        st.info("Interval columns (Int1..Int6) not found for xG/PPDA.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**xG — per interval (Selection)**")
            st.dataframe(
                df_int_sel[df_int_sel["Metric"]=="xG"][["Interval","Home","Away","H−A"]].round(3),
                use_container_width=True, hide_index=True
            )
        with c2:
            st.markdown("**PPDA — per interval (Selection)**")
            st.dataframe(
                df_int_sel[df_int_sel["Metric"]=="PPDA"][["Interval","Home","Away","H−A"]].round(3),
                use_container_width=True, hide_index=True
            )

        # ---------- Line charts with dynamic Baseline overlay ----------
        ints = [f"Int {i}" for i in range(1,7)]
        def _series(metric, side, source, tag):
            df_f = source[source["Metric"]==metric][["Interval", side]]
            df_f = df_f.set_index("Interval").reindex(ints).reset_index()
            df_f.columns = ["Interval", "Mean"]
            df_f["Series"] = f"{side} — {tag}"
            return df_f

        # xG
        xg_lines = pd.concat([
            _series("xG", "Home", df_int_sel,  "Selection"),
            _series("xG", "Away", df_int_sel,  "Selection"),
            _series("xG", "Home", df_int_base, "Baseline"),
            _series("xG", "Away", df_int_base, "Baseline"),
        ], ignore_index=True)

        fig_xg = px.line(
            xg_lines, x="Interval", y="Mean", color="Series", markers=True,
            title=f"xG — mean by interval (Selection vs {BASELINE_LABEL})",
            template="plotly_white",
            color_discrete_map={
                "Home — Selection": COL_HOME,
                "Away — Selection": COL_AWAY,
                "Home — Baseline": COL_BASE_HOME,
                "Away — Baseline": COL_BASE_AWAY,
            },
        )
        fig_xg.update_traces(selector=lambda tr: "Baseline" in tr.name, line=dict(dash="dash", width=2))
        st.plotly_chart(fig_xg, use_container_width=True)

        # PPDA
        pp_lines = pd.concat([
            _series("PPDA", "Home", df_int_sel,  "Selection"),
            _series("PPDA", "Away", df_int_sel,  "Selection"),
            _series("PPDA", "Home", df_int_base, "Baseline"),
            _series("PPDA", "Away", df_int_base, "Baseline"),
        ], ignore_index=True)

        fig_pp = px.line(
            pp_lines, x="Interval", y="Mean", color="Series", markers=True,
            title=f"PPDA — mean by interval (lower = more pressing) — Selection vs {BASELINE_LABEL}",
            template="plotly_white",
            color_discrete_map={
                "Home — Selection": COL_HOME,
                "Away — Selection": COL_AWAY,
                "Home — Baseline": COL_BASE_HOME,
                "Away — Baseline": COL_BASE_AWAY,
            },
        )
        fig_pp.update_traces(selector=lambda tr: "Baseline" in tr.name, line=dict(dash="dash", width=2))
        st.plotly_chart(fig_pp, use_container_width=True)

# --- Correlation ---
with tab4:
    st.markdown("### 🔗 Correlation")
    if len(df_sel) < 3:
        st.info("Need at least 3 rows for stable correlations.")
    else:
        # ========== Interval Matrices ==========
        st.markdown("#### 🧩 Intervals: PPDA ↔ xG (1–6)")
        mat_h, mat_a = interval_matrix(df_sel, True), interval_matrix(df_sel, False)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**🏠 Home**")
            heatmap(mat_h, "Home — PPDA vs xG (Intervals)", "PPDA (Home)", "xG (Home)")
            st.dataframe(mat_h.round(3), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**🛫 Away**")
            heatmap(mat_a, "Away — PPDA vs xG (Intervals)", "PPDA (Away)", "xG (Away)")
            st.dataframe(mat_a.round(3), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ========== Diagonal Int i ↔ Int i (Baseline = All Matches × current Subset) ==========
        st.markdown(f"#### ◇ Diagonal: Int i ↔ Int i (baseline = {BASELINE_LABEL})")
        dh_sel, da_sel = diag_series(df_sel, True),  diag_series(df_sel, False)
        dh_base, da_base = diag_series(df_base, True), diag_series(df_base, False)

        labels = [f"Int {i}" for i in range(1,7)]
        home_vals = pd.DataFrame({BASELINE_LABEL: dh_base.values, "Selection": dh_sel.values}, index=labels).round(3)
        away_vals = pd.DataFrame({BASELINE_LABEL: da_base.values, "Selection": da_sel.values}, index=labels).round(3)

        home_sorted = home_vals.sort_values("Selection", ascending=False)
        away_sorted = away_vals.sort_values("Selection", ascending=False)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**🏠 Home — values**")
            st.dataframe(home_vals, use_container_width=True)
            st.markdown("**🏠 Home — sorted by Selection (desc)**")
            st.dataframe(home_sorted, use_container_width=True)
        with cc2:
            st.markdown("**🛫 Away — values**")
            st.dataframe(away_vals, use_container_width=True)
            st.markdown("**🛫 Away — sorted by Selection (desc)**")
            st.dataframe(away_sorted, use_container_width=True)

        plot_diag_vs_baseline(
            dh_sel.values.tolist(), dh_base.values.tolist(),
            da_sel.values.tolist(), da_base.values.tolist(),
            title=f"Diagonal — Selection vs {BASELINE_LABEL}",
            baseline_name=BASELINE_LABEL
        )

        st.markdown("---")

        # ========== Totals & Halves (Selection only) ==========
        st.markdown("#### 🧱 Totals & Halves (PPDA ↔ xG) — Selection only")
        th_sel = totals_halves_table(df_sel)
        st.dataframe(th_sel, use_container_width=True)
        plot_totals_halves_selection(th_sel)

        st.markdown("---")

        # ========== 🔝 Top 10 drivers (signed) ==========
        st.markdown("#### 🔝 Top 10 drivers (signed)")
        d1,d2 = st.columns(2)
        with d1:
            st.markdown("**Target: xG_H_Total**")
            top_h = top_corr_signed(df_sel, "xG_H_Total", 10).rename("r")
            st.dataframe(top_h, use_container_width=True)
        with d2:
            st.markdown("**Target: xG_A_Total**")
            top_a = top_corr_signed(df_sel, "xG_A_Total", 10).rename("r")
            st.dataframe(top_a, use_container_width=True)

# --- Statistical Questions (renumbered 1–8) ---
with tab7:
    st.markdown("### 🧪 Statistical Questions (renumbered)")
    st.caption("All answers respect the current **Category** and **Subset** filter (left sidebar).")

    c1, c2 = st.columns([1,1])
    with c1: st.metric("Rows in selection", len(df_sel))
    with c2:
        nnum = df_sel.select_dtypes(include="number").shape[1]
        st.metric("Numeric columns", nnum)

    st.divider()

    # Q1: Home xG > Away xG ?
    st.markdown("#### 1) 🏠 Is Home xG higher than Away xG?")
    st.write("- **Method**: Paired t-test on total xG (home vs away) within the same match.")
    st.write("- **Meaning**: Positive difference ⇒ home creates more chances.")
    if _has_cols(df_sel, ["xG_H_Total","xG_A_Total"]):
        t,dfree,p,n,md,d = paired_t(df_sel["xG_H_Total"], df_sel["xG_A_Total"])
        m1, m2 = safe_num(df_sel["xG_H_Total"]).mean(), safe_num(df_sel["xG_A_Total"]).mean()
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Matches", n)
        k2.metric("Home mean xG", f"{m1:.3f}" if np.isfinite(m1) else "—")
        k3.metric("Away mean xG", f"{m2:.3f}" if np.isfinite(m2) else "—")
        k4.metric("Mean diff (H−A)", f"{md:.3f}" if np.isfinite(md) else "—")
        k5.metric("p-value", f"{p:.4f}" if np.isfinite(p) else "—")

        if np.isfinite(p) and np.isfinite(md):
            if p < 0.05 and md > 0:
                st.success("**Answer**: Yes — home tends to have higher xG (significant).")
            elif p < 0.05 and md < 0:
                st.warning("**Answer**: Away tends to have higher xG (significant).")
            else:
                # Directional tendency even when not significant
                if md > 0:
                    st.info("**Answer**: Home *tends* to have higher xG (not statistically significant).")
                elif md < 0:
                    st.info("**Answer**: Away *tends* to have higher xG (not statistically significant).")
                else:
                    st.info("**Answer**: Means are essentially equal (not statistically different).")
    else:
        st.info("Need `xG_H_Total`, `xG_A_Total`.")

    st.divider()

    # Q2: Home PPDA < Away PPDA ?
    st.markdown("#### 2) 🧱 Is Home PPDA lower than Away PPDA?")
    st.write("- **Method**: Paired t-test on total PPDA (home vs away).")
    st.write("- **Meaning**: Lower PPDA ⇒ more pressing. Negative (H−A) ⇒ home presses more.")
    if _has_cols(df_sel, ["PPDA_H_Total","PPDA_A_Total"]):
        t,dfree,p,n,md,d = paired_t(df_sel["PPDA_H_Total"], df_sel["PPDA_A_Total"])
        m1, m2 = safe_num(df_sel["PPDA_H_Total"]).mean(), safe_num(df_sel["PPDA_A_Total"]).mean()
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Matches", n)
        k2.metric("Home mean PPDA", f"{m1:.3f}" if np.isfinite(m1) else "—")
        k3.metric("Away mean PPDA", f"{m2:.3f}" if np.isfinite(m2) else "—")
        k4.metric("Mean diff (H−A)", f"{md:.3f}" if np.isfinite(md) else "—")
        k5.metric("p-value", f"{p:.4f}" if np.isfinite(p) else "—")

        if np.isfinite(p) and np.isfinite(md):
            if p < 0.05 and md < 0:
                st.success("**Answer**: Yes — home tends to press more (PPDA lower; significant).")
            elif p < 0.05 and md > 0:
                st.warning("**Answer**: Away tends to press more (PPDA lower; significant).")
            else:
                # Directional tendency even when not significant
                if md < 0:
                    st.info("**Answer**: Home *tends* to press more (PPDA lower; not statistically significant).")
                elif md > 0:
                    st.info("**Answer**: Away *tends* to press more (PPDA lower; not statistically significant).")
                else:
                    st.info("**Answer**: Means are essentially equal (not statistically different).")
    else:
        st.info("Need `PPDA_H_Total`, `PPDA_A_Total`.")

    st.divider()

    # Q3: What matters more for xG?
    st.markdown("#### 3) What matters more for xG: opponent presses less or we press more?")
    st.write("- **Method**: Standardized regression with both factors.")
    st.write("- **Meaning**: Bigger absolute weight ⇒ stronger link to our xG.")
    if _has_cols(df_sel, ["xG_H_Total","PPDA_H_Total","PPDA_A_Total"]):
        b_our, b_opp_raw, r2 = _std_betas_two_predictors(df_sel["xG_H_Total"], df_sel["PPDA_H_Total"], df_sel["PPDA_A_Total"])
        b_opp = -b_opp_raw  # positive means opponent presses less → our xG up
        st.table(pd.DataFrame({
            "Weight: our pressing more (PPDA↓)":[None if np.isnan(b_our) else round(b_our,3)],
            "Weight: opponent presses less":[None if np.isnan(b_opp) else round(b_opp,3)],
            "R² (fit)":[None if np.isnan(r2) else round(r2,3)],
        }))
        if np.isfinite(b_our) and np.isfinite(b_opp):
            if abs(b_our) > abs(b_opp):
                st.success("**Answer**: Our own pressing has the stronger link to xG.")
            elif abs(b_opp) > abs(b_our):
                st.success("**Answer**: The opponent pressing less has the stronger link.")
            else:
                st.info("**Answer**: Both links look similar.")
    else:
        st.info("Need: `xG_H_Total`, `PPDA_H_Total`, `PPDA_A_Total`.")

    st.divider()

    # Q4: Interval with most xG & lowest PPDA
    st.markdown("#### 4) Which interval has the most xG and the lowest PPDA?")
    st.write("- **Method**: Average per interval (Int1..Int6) for Home and Away; report leaders.")
    need = [f"xG_H_Int{i}" for i in range(1,7)] + [f"xG_A_Int{i}" for i in range(1,7)] + \
           [f"PPDA_H_Int{i}" for i in range(1,7)] + [f"PPDA_A_Int{i}" for i in range(1,7)]
    if _has_cols(df_sel, need):
        xgH = safe_num(df_sel[[f"xG_H_Int{i}" for i in range(1,7)]]).mean()
        xgA = safe_num(df_sel[[f"xG_A_Int{i}" for i in range(1,7)]]).mean()
        ppH = safe_num(df_sel[[f"PPDA_H_Int{i}" for i in range(1,7)]]).mean()
        ppA = safe_num(df_sel[[f"PPDA_A_Int{i}" for i in range(1,7)]]).mean()
        idx_to_name = {i-1: f"Int {i}" for i in range(1,7)}
        st.table(pd.DataFrame({
            "Interval":[f"Int {i}" for i in range(1,7)],
            "xG Home":xgH.values.round(3), "xG Away":xgA.values.round(3),
            "PPDA Home":ppH.values.round(3), "PPDA Away":ppA.values.round(3)
        }))
        st.success(f"**Home** — Most xG: {idx_to_name[int(xgH.values.argmax())]} · Lowest PPDA: {idx_to_name[int(ppH.values.argmin())]}")
        st.success(f"**Away** — Most xG: {idx_to_name[int(xgA.values.argmax())]} · Lowest PPDA: {idx_to_name[int(ppA.values.argmin())]}")
    else:
        st.info("Need Int1..6 for xG and PPDA.")

    st.divider()

    # Q5: Largest Home–Away xG gap
    st.markdown("#### 5) Which interval has the largest Home–Away xG gap?")
    st.write("- **Method**: Mean of (Home−Away) xG for each interval; report the largest.")
    need = [f"xG_H_Int{i}" for i in range(1,7)] + [f"xG_A_Int{i}" for i in range(1,7)]
    if _has_cols(df_sel, need):
        gaps = []
        for i in range(1,7):
            h = safe_num(df_sel[f"xG_H_Int{i}"])
            a = safe_num(df_sel[f"xG_A_Int{i}"])
            gaps.append((f"Int {i}", (h-a).mean()))
        df_gap = pd.DataFrame(gaps, columns=["Interval","Mean(H−A)"]).round(4)
        st.table(df_gap)
        best = df_gap.iloc[df_gap["Mean(H−A)"].idxmax()]
        st.success(f"**Answer**: {best['Interval']} (mean H−A = {best['Mean(H−A)']:.3f})")
    else:
        st.info("Need Int1..6 xG for both sides.")

    st.divider()

    # Q6: Phase leaders
    st.markdown("#### 6) Which phase (Early/Mid/Late) leads?")
    st.write("- **Method**: Early=Int1–2, Mid=Int3–4, Late=Int5–6; compare phase means.")
    phH = _phase_sums(df_sel,"xG","PPDA","H")
    phA = _phase_sums(df_sel,"xG","PPDA","A")
    if phH and phA:
        tbl = []
        for phase in ["Early","Mid","Late"]:
            tbl.append([phase,
                        phH["xg"][phase].mean(), phA["xg"][phase].mean(),
                        phH["pp"][phase].mean(), phA["pp"][phase].mean()])
        out = pd.DataFrame(tbl, columns=["Phase","xG Home","xG Away","PPDA Home","PPDA Away"]).round(3)
        st.table(out)
        st.success(f"**Home** — Highest xG: {out.iloc[out['xG Home'].idxmax()]['Phase']} · "
                   f"**Lowest PPDA**: {out.iloc[out['PPDA Home'].idxmin()]['Phase']}")
    else:
        st.info("Need Int1..6 for xG and PPDA.")

    st.divider()

    # Q7: 1st vs 2nd half
    st.markdown("#### 7) 1st half vs 2nd half — changes in xG and PPDA?")
    st.write("- **Method**: Paired t-tests (2nd − 1st) for xG and PPDA, Home and Away.")
    needed = ["xG_H_1stHalf","xG_H_2ndHalf","xG_A_1stHalf","xG_A_2ndHalf",
              "PPDA_H_1stHalf","PPDA_H_2ndHalf","PPDA_A_1stHalf","PPDA_A_2ndHalf"]
    if _has_cols(df_sel, needed):
        tH,dfH,pH,nH,mdH,dH = paired_t(df_sel["xG_H_1stHalf"], df_sel["xG_H_2ndHalf"])
        tA,dfA,pA,nA,mdA,dA = paired_t(df_sel["xG_A_1stHalf"], df_sel["xG_A_2ndHalf"])
        tpH,dfpH,ppH,npH,mdpH,dpH = paired_t(df_sel["PPDA_H_1stHalf"], df_sel["PPDA_H_2ndHalf"])
        tpA,dfpA,ppA,npA,mdpA,dpA = paired_t(df_sel["PPDA_A_1stHalf"], df_sel["PPDA_A_2ndHalf"])
        out = pd.DataFrame({
            "Metric":["xG Home","xG Away","PPDA Home","PPDA Away"],
            "Δ (2nd−1st)":[mdH, mdA, mdpH, mdpA],
            "p-value":[pH, pA, ppH, ppA],
            "n":[nH, nA, npH, npA]
        }).round(4)
        st.table(out)
        st.caption("Note: Lower PPDA = more pressing.")
    else:
        st.info("Need half totals (Home & Away).")

    st.divider()

    # Q8: Opponent very low xG → our PPDA?
    st.markdown("#### 8) When opponent xG is very low, what is our PPDA?")
    st.write("- **Method**: Look at matches where the opponent’s total xG is near zero, summarize our PPDA, and compare to overall PPDA.")
    view = st.radio("View", ["Home perspective (opp = Away xG, our PPDA = PPDA_H_Total)",
                             "Away perspective (opp = Home xG, our PPDA = PPDA_A_Total)"],
                    horizontal=False, index=0, key="q14_view")
    mode = st.radio("How to define “very low”?", ["Threshold", "Bottom percentile"], horizontal=True, index=0, key="q14_mode")

    if view.startswith("Home"):
        need = ["xG_A_Total","PPDA_H_Total"]
        opp_xg = "xG_A_Total"
        our_ppda = "PPDA_H_Total"
    else:
        need = ["xG_H_Total","PPDA_A_Total"]
        opp_xg = "xG_H_Total"
        our_ppda = "PPDA_A_Total"

    if _has_cols(df_sel, need):
        series_opp = safe_num(df_sel[opp_xg])
        series_ppd = safe_num(df_sel[our_ppda])
        overall_mean = float(series_ppd.mean())
        if mode == "Threshold":
            thr = st.slider("Opponent xG ≤", 0.00, 0.30, 0.05, 0.01, key="q14_thr")
            mask = series_opp.fillna(0) <= thr
            label = f"opp xG ≤ {thr:.2f}"
        else:
            pct = st.slider("Bottom percentile of opponent xG", 1, 20, 10, 1, key="q14_pct")
            cut = np.nanpercentile(series_opp, pct)
            mask = series_opp <= cut
            label = f"opp xG in bottom {pct}% (≤ {cut:.2f})"

        sub = series_ppd[mask]
        other = series_ppd[~mask]

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches (subset)", int(mask.sum()))
        c2.metric("Our PPDA (subset mean)", f"{sub.mean():.3f}" if mask.any() else "—")
        c3.metric("Our PPDA (overall mean)", f"{overall_mean:.3f}" if np.isfinite(overall_mean) else "—")

        if mask.sum() >= 3 and (~mask).sum() >= 3:
            t_stat, dfree, pval, n1, n2, diff = welch(sub, other)
            st.caption(f"Subset vs others — Δ={diff:.3f}, p≈{pval:.4f} ({label})")

        if mask.any():
            df_plot = pd.DataFrame({
                "PPDA": pd.concat([sub, series_ppd], ignore_index=True),
                "Group": (["Subset"]*sub.size) + (["All"]*series_ppd.size)
            })
            fig = px.histogram(df_plot, x="PPDA", color="Group", barmode="overlay",
                               nbins=40, opacity=0.55,
                               color_discrete_map={"Subset": COL_HOME, "All": COL_AWAY},
                               title="Our PPDA — subset vs all")
            fig.update_layout(template="plotly_white", height=380, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No matches meet the current ‘very low opponent xG’ condition.")
    else:
        st.info(f"Need `{opp_xg}` and `{our_ppda}`.")

# --- Similarity (Unidentified → predicted label) ---
with tab6:
    st.markdown("### 🧬 Similarity mapping for *Unidentified* matches")
    st.write(
        "Most *Unidentified* games are probably clean (no abnormal betting). "
        "To explore further, each *Unidentified* match is mapped to the closest reviewed label "
        "(**both not suspicious**, **home suspicious**, **away suspicious**, **both suspicious**) using **cosine similarity**.\n\n"
        "**Cosine similarity (simple idea):** turn every match into a vector of numbers (features). "
        "Cosine measures the angle between two vectors. If the angle is small (cosine close to 1), the matches look alike."
    )

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        from sklearn.neighbors import NearestNeighbors
        HAS_SK = True
    except Exception:
        HAS_SK = False

    if not HAS_SK:
        st.warning("Install scikit-learn to run similarity mapping: `pip install scikit-learn`")
    else:
        LABEL_COL = 'MatchSuspicionType'
        CAT_COLS = ['Relative quality']
        DROP_COLS = ['xG_A_ChangePct', 'xG_H_ChangePct', 'StrongerTeam_xG_Diff', 'StrongerTeam_xG' ]

        metric = st.selectbox("Similarity metric", ["cosine","euclidean"], index=0)
        KNN_K = st.slider("kNN agreement check (0 disables)", 0, 25, 11, 1)

        def make_features(df_in: pd.DataFrame):
            X = df_in.copy()
            if CAT_COLS:
                X = pd.get_dummies(X, columns=[c for c in CAT_COLS if c in X.columns], drop_first=False)
            for c in DROP_COLS:
                if c in X.columns: X = X.drop(columns=c)
            X = X.select_dtypes(include=[np.number]).copy()
            y = df_in[LABEL_COL].astype(str) if LABEL_COL in df_in.columns else pd.Series(index=df_in.index, dtype=str)
            return X, y, X.columns

        X_all, y_all, feat_cols = make_features(df)
        mask_unid = y_all.str.lower().str.strip().eq("unidentified")
        mask_lab = ~mask_unid
        X_lab, y_lab = X_all.loc[mask_lab], y_all.loc[mask_lab]
        X_unl = X_all.loc[mask_unid]

        if X_unl.empty or X_lab.empty:
            st.info("Not enough data: need both labeled and *Unidentified* matches.")
        else:
            scaler = StandardScaler()
            X_lab_s = pd.DataFrame(scaler.fit_transform(X_lab), index=X_lab.index, columns=feat_cols)
            X_unl_s = pd.DataFrame(scaler.transform(X_unl), index=X_unl.index, columns=feat_cols)

            classes = [c for c in y_lab.unique() if c != "unidentified"]
            if not classes:
                st.info("No labeled classes to learn from."); st.stop()

            # centroids
            centroids = {c: X_lab_s[y_lab==c].mean(axis=0).values for c in classes}
            centroid_matrix = np.vstack([centroids[c] for c in classes])

            # similarity / distance to prototypes
            def prototype_scores(X_std, centroid_matrix, classes, metric="cosine"):
                if metric == "cosine":
                    sims = cosine_similarity(X_std.values, centroid_matrix)
                else:
                    dists = euclidean_distances(X_std.values, centroid_matrix)
                    sims = 1.0 / (1.0 + dists)  # convert to similarity
                scores = pd.DataFrame(sims, index=X_std.index, columns=[f"sim_{c}" for c in classes])
                exp_s = np.exp(sims - sims.max(axis=1, keepdims=True))
                conf = (exp_s / exp_s.sum(axis=1, keepdims=True)).max(axis=1)
                best_idx = sims.argmax(axis=1)
                best_classes = [classes[i] for i in best_idx]
                return scores, pd.Series(best_classes, index=X_std.index, name='PredClass'), pd.Series(conf, index=X_std.index, name='PredConfidence')

            scores_proto, pred_proto, conf_proto = prototype_scores(X_unl_s, centroid_matrix, classes, metric)

            # Optional kNN sanity check
            def knn_predict_labels(X_train, y_train, X_query, k=11, metric='cosine'):
                if k <= 0: return pd.Series(index=X_query.index, dtype=object), pd.Series(index=X_query.index, dtype=float)
                nn = NearestNeighbors(n_neighbors=min(k, len(X_train)), metric=('cosine' if metric=='cosine' else 'minkowski'))
                nn.fit(X_train.values)
                dists, idxs = nn.kneighbors(X_query.values)
                if metric == 'cosine':
                    sims = 1.0 - dists
                    sims[sims < 0] = 0
                    weights = sims
                else:
                    weights = 1.0 / (1.0 + dists)
                preds, confs = [], []
                y_train_arr = y_train.values
                for i in range(len(X_query)):
                    neigh_idx = idxs[i]
                    neigh_labels = y_train_arr[neigh_idx]
                    neigh_w = weights[i]
                    class_w = {}
                    for lab, w in zip(neigh_labels, neigh_w):
                        class_w[lab] = class_w.get(lab, 0.0) + w
                    best_lab = max(class_w, key=class_w.get)
                    total = sum(class_w.values())
                    conf = (class_w[best_lab] / total) if total > 0 else 0.0
                    preds.append(best_lab); confs.append(conf)
                return pd.Series(preds, index=X_query.index, name='kNN_Pred'), pd.Series(confs, index=X_query.index, name='kNN_Confidence')

            knn_pred, knn_conf = knn_predict_labels(X_lab_s, y_lab, X_unl_s, k=KNN_K, metric=metric)
            if KNN_K > 0 and len(knn_pred) == len(pred_proto):
                fused_pred = pred_proto.copy(); fused_conf = conf_proto.copy()
                agree_mask = (pred_proto == knn_pred)
                fused_conf[agree_mask] = (conf_proto[agree_mask] + knn_conf[agree_mask]) / 2.0
                for i in fused_pred.index[~agree_mask]:
                    if knn_conf.loc[i] > conf_proto.loc[i]:
                        fused_pred.loc[i] = knn_pred.loc[i]; fused_conf.loc[i] = knn_conf.loc[i]
            else:
                fused_pred, fused_conf = pred_proto, conf_proto

            df_out = df.copy()
            df_out.loc[mask_unid, 'MatchSuspicionType_Pred'] = fused_pred
            df_out.loc[mask_unid, 'Pred_Confidence'] = fused_conf.round(3)
            for col in scores_proto.columns:
                df_out.loc[mask_unid, col] = scores_proto[col].round(4)

            counts = df_out.loc[mask_unid, 'MatchSuspicionType_Pred'].value_counts().rename("count")
            perc = (100*counts/counts.sum()).round(2).astype(str) + "%"
            st.markdown("#### 📊 Prediction summary (Unidentified only)")
            st.dataframe(pd.concat([counts, perc.rename("percent")], axis=1), use_container_width=True)

            st.markdown("#### 🔎 Preview (top 300)")
            st.dataframe(
                df_out.loc[mask_unid, ['MatchSuspicionType_Pred','Pred_Confidence'] + [c for c in df_out.columns if c.startswith('sim_')]]
                      .sort_values(['Pred_Confidence'], ascending=False)
                      .head(300),
                use_container_width=True, height=360, hide_index=True
            )
# streamlit run "/Users/farzammanafzadeh/Desktop/Uefa Task/uefa_shortlist_app/App/Main.py"