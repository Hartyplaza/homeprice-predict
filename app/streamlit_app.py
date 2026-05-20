import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomePrice_Predict",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme colours (amber/gold dark) ──────────────────────────────────────────
AMBER   = '#F59E0B'
GREEN   = '#34D399'
BLUE    = '#60A5FA'
RED     = '#F87171'
BG      = '#0F0F0F'
SURFACE = '#1A1A1A'
TEXT    = '#F5F5F5'

LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,26,26,0.6)',
    font=dict(color=TEXT, size=12),
    title_font=dict(color=AMBER, size=15),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0F0F0F;
    color: #F5F5F5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #2A2A2A;
}
section[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="metric-container"] label { color: #888 !important; font-size: 0.78rem !important; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F59E0B !important; font-size: 1.6rem !important; font-weight: 700; }

/* Tabs */
[data-testid="stTabs"] button { font-family: 'Syne', sans-serif; font-size: 0.85rem; letter-spacing: 0.05em; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #F59E0B; border-bottom: 2px solid #F59E0B; }

/* Sliders */
[data-testid="stSlider"] > div > div > div { background: #F59E0B !important; }

/* Selectbox & number input */
[data-testid="stSelectbox"] select,
[data-testid="stNumberInput"] input { background: #1A1A1A !important; border-color: #2A2A2A !important; color: #F5F5F5 !important; }

/* Predict button */
div.stButton > button {
    background: #F59E0B;
    color: #0F0F0F;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2.5rem;
    width: 100%;
    letter-spacing: 0.06em;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

/* Dividers */
hr { border-color: #2A2A2A; }

/* Header badge */
.badge {
    display: inline-block;
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.72rem;
    color: #F59E0B;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
    margin-right: 6px;
}

/* Price result box */
.price-box {
    background: linear-gradient(135deg, #1A1500 0%, #1A1A1A 100%);
    border: 1px solid #F59E0B44;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.price-label { font-size: 0.8rem; color: #888; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.price-value { font-size: 3rem; font-weight: 700; color: #F59E0B; line-height: 1; }
.price-range { font-size: 0.85rem; color: #888; margin-top: 0.5rem; }

/* Section label */
.section-label {
    font-size: 0.72rem;
    color: #F59E0B;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.5rem;
}

/* Info cards */
.info-card {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    model_dir = os.path.join(base, '..', 'models')
    data_dir  = os.path.join(base, '..', 'data', 'processed')

    ridge  = joblib.load(os.path.join(model_dir, 'ridge_model.pkl'))
    lasso  = joblib.load(os.path.join(model_dir, 'lasso_model.pkl'))
    xgb    = joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))
    w      = joblib.load(os.path.join(model_dir, 'ensemble_weights.pkl'))
    feats  = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))

    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()

    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_train)

    return ridge, lasso, xgb, w, feats, X_train, y_train, explainer, shap_values

ridge_pipe, lasso_pipe, xgb_model, weights, feat_cols, X_train, y_train, explainer, shap_vals = load_artifacts()


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_price(input_dict):
    row = pd.DataFrame([input_dict])
    # Align to training feature columns
    for col in feat_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feat_cols]

    r = ridge_pipe.predict(row)[0]
    l = lasso_pipe.predict(row)[0]
    x = xgb_model.predict(row)[0]
    ensemble = weights[0]*r + weights[1]*l + weights[2]*x
    price = np.expm1(ensemble)

    # ±10% confidence interval (heuristic)
    low  = price * 0.90
    high = price * 1.10
    return price, low, high, row


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 1rem;">
    <div style="margin-bottom: 0.75rem;">
        <span class="badge">REGRESSION</span>
        <span class="badge">AMES HOUSING</span>
        <span class="badge">ENSEMBLE</span>
    </div>
    <h1 style="font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;">
        HomePrice<span style="color: #F59E0B;">_</span>Predict
    </h1>
    <p style="color: #888; margin-top: 0.4rem; font-size: 0.95rem;">
        Ridge · Lasso · XGBoost ensemble — Ames Housing dataset
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Model Performance", "SHAP Analysis", "About"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">House Parameters</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Size & Structure**")
        gr_liv_area    = st.slider("Above Grade Living Area (sq ft)", 300, 5000, 1500)
        total_bsmt_sf  = st.slider("Total Basement SF", 0, 3000, 800)
        first_flr_sf   = st.slider("1st Floor SF", 300, 4000, 1000)
        garage_area    = st.slider("Garage Area (sq ft)", 0, 1500, 400)
        garage_cars    = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)

        st.markdown("**Age & Condition**")
        year_built     = st.slider("Year Built", 1872, 2010, 1990)
        year_remod     = st.slider("Year Remodelled", 1950, 2010, 1995)
        overall_qual   = st.slider("Overall Quality (1–10)", 1, 10, 6)
        overall_cond   = st.slider("Overall Condition (1–10)", 1, 10, 5)

    with col_right:
        st.markdown("**Rooms & Features**")
        full_bath      = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
        half_bath      = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
        bedroom        = st.selectbox("Bedrooms Above Grade", [0, 1, 2, 3, 4, 5, 6], index=3)
        tot_rms        = st.selectbox("Total Rooms Above Grade", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=5)
        fireplaces     = st.selectbox("Fireplaces", [0, 1, 2, 3], index=0)

        st.markdown("**Quality Ratings**")
        exter_qual     = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
        kitchen_qual   = st.selectbox("Kitchen Quality",  ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
        bsmt_qual      = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "Po", "None"], index=2)

        st.markdown("**Location**")
        neighborhood   = st.selectbox("Neighborhood", [
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
            "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
            "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
            "IDOTRR", "ClearCr", "StoneBr", "SWISU", "Blmngtn",
            "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste"
        ], index=0)

    st.markdown("---")
    predict_btn = st.button("Predict Sale Price")

    if predict_btn:
        qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}

        yr_sold = 2010   # Ames dataset year range
        house_age        = yr_sold - year_built
        yrs_since_remod  = yr_sold - year_remod
        is_remodeled     = int(year_built != year_remod)
        is_new           = int(year_built == yr_sold)
        total_sf         = total_bsmt_sf + first_flr_sf
        total_bathrooms  = full_bath + 0.5*half_bath
        qual_x_area      = overall_qual * gr_liv_area
        qual_x_total_sf  = overall_qual * total_sf

        # Build input dict — numeric features
        input_dict = {
            'GrLivArea'         : np.log1p(gr_liv_area),
            'TotalBsmtSF'       : np.log1p(total_bsmt_sf),
            '1stFlrSF'          : np.log1p(first_flr_sf),
            'GarageArea'        : np.log1p(garage_area),
            'GarageCars'        : garage_cars,
            'FullBath'          : full_bath,
            'HalfBath'          : half_bath,
            'BedroomAbvGr'      : bedroom,
            'TotRmsAbvGrd'      : tot_rms,
            'Fireplaces'        : fireplaces,
            'OverallQual'       : overall_qual,
            'OverallCond'       : overall_cond,
            'ExterQual'         : qual_map[exter_qual],
            'KitchenQual'       : qual_map[kitchen_qual],
            'BsmtQual'          : qual_map[bsmt_qual],
            'TotalSF'           : np.log1p(total_sf),
            'TotalBathrooms'    : total_bathrooms,
            'HouseAge'          : np.log1p(max(house_age, 0)),
            'YearsSinceRemodel' : np.log1p(max(yrs_since_remod, 0)),
            'IsRemodeled'       : is_remodeled,
            'IsNew'             : is_new,
            'QualxArea'         : np.log1p(qual_x_area),
            'QualxTotalSF'      : np.log1p(qual_x_total_sf),
            'HasGarage'         : int(garage_area > 0),
            'HasBsmt'           : int(total_bsmt_sf > 0),
            'HasFireplace'      : int(fireplaces > 0),
            'YrSold'            : yr_sold,
            'YearBuilt'         : year_built,
            'YearRemodAdd'      : year_remod,
        }

        # One-hot: Neighborhood
        for nbhd in ["NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt",
                     "Gilbert","Sawyer","NWAmes","SawyerW","BrkSide","Crawfor",
                     "Mitchel","NoRidge","Timber","IDOTRR","ClearCr","StoneBr",
                     "SWISU","Blmngtn","MeadowV","BrDale","Veenker","NPkVill","Blueste"]:
            input_dict[f'Neighborhood_{nbhd}'] = int(neighborhood == nbhd)

        price, low, high, input_row = predict_price(input_dict)

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        with res_col1:
            st.markdown(f"""
            <div class="price-box">
                <div class="price-label">PREDICTED SALE PRICE</div>
                <div class="price-value">${price:,.0f}</div>
                <div class="price-range">95% range: ${low:,.0f} – ${high:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            r_pred = np.expm1(ridge_pipe.predict(input_row)[0])
            l_pred = np.expm1(lasso_pipe.predict(input_row)[0])
            x_pred = np.expm1(xgb_model.predict(input_row)[0])

            st.metric("Ridge",   f"${r_pred:,.0f}")
            st.metric("Lasso",   f"${l_pred:,.0f}")
            st.metric("XGBoost", f"${x_pred:,.0f}")

        with res_col3:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=price,
                number={'prefix': '$', 'valueformat': ',.0f', 'font': {'color': AMBER, 'size': 18}},
                gauge={
                    'axis': {'range': [0, 800000], 'tickcolor': '#444'},
                    'bar':  {'color': AMBER},
                    'bgcolor': SURFACE,
                    'steps': [
                        {'range': [0, 150000],    'color': '#111'},
                        {'range': [150000, 300000],'color': '#161616'},
                        {'range': [300000, 500000],'color': '#1A1A1A'},
                        {'range': [500000, 800000],'color': '#1E1E1E'},
                    ],
                    'threshold': {'line': {'color': GREEN, 'width': 2}, 'value': price}
                },
                title={'text': "Price Gauge", 'font': {'color': '#888', 'size': 12}}
            ))
            fig_gauge.update_layout(
                height=200,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT),
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Model agreement bar
        st.markdown('<div class="section-label" style="margin-top:1.5rem">Model Breakdown</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=['Ridge', 'Lasso', 'XGBoost', 'Ensemble'],
            y=[r_pred, l_pred, x_pred, price],
            marker_color=[BLUE, GREEN, RED, AMBER],
            text=[f'${v:,.0f}' for v in [r_pred, l_pred, x_pred, price]],
            textposition='outside',
            textfont=dict(color=TEXT, size=11)
        ))
        fig_bar.update_layout(
            title='Individual Model Predictions vs Ensemble',
            yaxis_title='SalePrice ($)',
            height=320,
            showlegend=False,
            **LAYOUT
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    ridge_pred = ridge_pipe.predict(X_train)
    lasso_pred = lasso_pipe.predict(X_train)
    xgb_pred   = xgb_model.predict(X_train)
    ens_pred   = weights[0]*ridge_pred + weights[1]*lasso_pred + weights[2]*xgb_pred

    def metrics(y, p):
        return {
            'RMSE': np.sqrt(mean_squared_error(y, p)),
            'MAE' : mean_absolute_error(y, p),
            'R²'  : r2_score(y, p)
        }

    m = {
        'Ridge'   : metrics(y_train, ridge_pred),
        'Lasso'   : metrics(y_train, lasso_pred),
        'XGBoost' : metrics(y_train, xgb_pred),
        'Ensemble': metrics(y_train, ens_pred),
    }

    # Top metric cards
    best = m['Ensemble']
    c1, c2, c3 = st.columns(3)
    c1.metric("Ensemble RMSE", f"{best['RMSE']:.5f}")
    c2.metric("Ensemble MAE",  f"{best['MAE']:.5f}")
    c3.metric("Ensemble R²",   f"{best['R²']:.4f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        # RMSE comparison
        fig_rmse = px.bar(
            x=list(m.keys()), y=[v['RMSE'] for v in m.values()],
            title='RMSE by Model (log1p scale)',
            labels={'x': 'Model', 'y': 'RMSE'},
            color=list(m.keys()),
            color_discrete_sequence=[BLUE, GREEN, RED, AMBER]
        )
        fig_rmse.update_layout(**LAYOUT, showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_b:
        # R² comparison
        fig_r2 = px.bar(
            x=list(m.keys()), y=[v['R²'] for v in m.values()],
            title='R² Score by Model',
            labels={'x': 'Model', 'y': 'R²'},
            color=list(m.keys()),
            color_discrete_sequence=[BLUE, GREEN, RED, AMBER]
        )
        fig_r2.update_layout(**LAYOUT, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Actual vs Predicted
    st.markdown('<div class="section-label">Actual vs Predicted — Ensemble</div>', unsafe_allow_html=True)
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=y_train.values, y=ens_pred,
        mode='markers',
        marker=dict(color=AMBER, size=4, opacity=0.4),
        name='Predictions'
    ))
    fig_avp.add_trace(go.Scatter(
        x=[y_train.min(), y_train.max()],
        y=[y_train.min(), y_train.max()],
        mode='lines',
        line=dict(color=GREEN, dash='dash', width=1.5),
        name='Perfect fit'
    ))
    fig_avp.update_layout(
        title='Actual vs Predicted (log1p scale)',
        xaxis_title='Actual log1p(SalePrice)',
        yaxis_title='Predicted log1p(SalePrice)',
        height=420, **LAYOUT
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # Residuals
    residuals = y_train.values - ens_pred
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=ens_pred, y=residuals,
        mode='markers',
        marker=dict(color=BLUE, size=4, opacity=0.4),
        name='Residuals'
    ))
    fig_resid.add_hline(y=0, line_dash='dash', line_color=AMBER, opacity=0.6)
    fig_resid.update_layout(
        title='Residual Plot — Ensemble (Predicted vs Residual)',
        xaxis_title='Predicted log1p(SalePrice)',
        yaxis_title='Residual',
        height=350, **LAYOUT
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    # Ensemble weights
    st.markdown('<div class="section-label">Ensemble Weights</div>', unsafe_allow_html=True)
    wt_col1, wt_col2, wt_col3 = st.columns(3)
    wt_col1.metric("Ridge weight",   f"{weights[0]:.4f}")
    wt_col2.metric("Lasso weight",   f"{weights[1]:.4f}")
    wt_col3.metric("XGBoost weight", f"{weights[2]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Global Feature Importance</div>', unsafe_allow_html=True)

    mean_shap = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_train.columns
    ).sort_values(ascending=False)

    top_n = st.slider("Number of top features to display", 10, 40, 20)

    top_shap = mean_shap.head(top_n)
    fig_shap = px.bar(
        x=top_shap.values, y=top_shap.index,
        orientation='h',
        title=f'Top {top_n} Features by Mean |SHAP| Value (XGBoost)',
        labels={'x': 'Mean |SHAP|', 'y': 'Feature'},
        color=top_shap.values,
        color_continuous_scale=[[0, SURFACE], [1, AMBER]]
    )
    fig_shap.update_layout(**LAYOUT, height=max(400, top_n * 22))
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # SHAP dependence
    st.markdown('<div class="section-label">SHAP Dependence Plot</div>', unsafe_allow_html=True)
    sel_feat = st.selectbox("Select feature", options=mean_shap.head(30).index.tolist())

    shap_series = pd.Series(shap_vals[:, X_train.columns.get_loc(sel_feat)])
    feat_vals   = X_train[sel_feat]

    color_by = 'OverallQual' if 'OverallQual' in X_train.columns else None
    fig_dep = px.scatter(
        x=feat_vals, y=shap_series,
        color=X_train[color_by] if color_by else None,
        title=f'SHAP Dependence — {sel_feat}',
        labels={'x': sel_feat, 'y': f'SHAP({sel_feat})', 'color': color_by},
        color_continuous_scale=[[0, '#1E3A5F'], [1, AMBER]],
        opacity=0.6
    )
    fig_dep.add_hline(y=0, line_dash='dash', line_color='#444', opacity=0.5)
    fig_dep.update_layout(**LAYOUT, height=380)
    st.plotly_chart(fig_dep, use_container_width=True)

    st.markdown("---")

    # Waterfall — single prediction
    st.markdown('<div class="section-label">Single Prediction Explanation</div>', unsafe_allow_html=True)
    sample_idx = st.slider("Training sample index", 0, len(X_train)-1, 0)

    shap_single = shap_vals[sample_idx]
    top_idx = np.argsort(np.abs(shap_single))[::-1][:15]

    wf_df = pd.DataFrame({
        'Feature': X_train.columns[top_idx],
        'SHAP'   : shap_single[top_idx]
    }).sort_values('SHAP')

    fig_wf = px.bar(
        wf_df, x='SHAP', y='Feature',
        orientation='h',
        title=f'SHAP Waterfall — Sample index {sample_idx}',
        color='SHAP',
        color_continuous_scale=[[0, RED], [0.5, SURFACE], [1, AMBER]]
    )
    fig_wf.add_vline(x=0, line_color='#444', line_dash='dash', opacity=0.6)
    fig_wf.update_layout(**LAYOUT, height=480)
    st.plotly_chart(fig_wf, use_container_width=True)

    pred_price_sample = np.expm1(
        weights[0]*ridge_pipe.predict(X_train.iloc[[sample_idx]])[0] +
        weights[1]*lasso_pipe.predict(X_train.iloc[[sample_idx]])[0] +
        weights[2]*xgb_model.predict(X_train.iloc[[sample_idx]])[0]
    )
    st.markdown(f"**Ensemble predicted price for this sample:** `${pred_price_sample:,.0f}`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
        <div class="section-label">Competition</div>
        <h2 style="margin-top:0.25rem; font-size:1.6rem; font-weight:700">HomePrice_Predict</h2>
        <p style="color:#888; line-height:1.7;">
            This project is a submission to the Kaggle competition
            <strong style="color:#F5F5F5;">House Prices: Advanced Regression Techniques</strong> —
            a knowledge competition designed to challenge data scientists with creative feature
            engineering and advanced regression methods.
        </p>
        <p style="color:#888; line-height:1.7;">
            The goal is to predict the final sale price of residential homes in Ames, Iowa
            using 79 explanatory variables covering almost every aspect of the property —
            from the size of the basement to the proximity of the house to a railroad.
            Submissions are evaluated on the Root Mean Squared Logarithmic Error (RMSLE)
            between the predicted and actual sale prices.
        </p>
        <p style="color:#888; line-height:1.7;">
            This solution combines Ridge, Lasso, and XGBoost in a weighted ensemble, with
            extensive feature engineering (14 new features, ordinal encoding, log transforms)
            and SHAP explainability for every prediction.
        </p>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">Kaggle Submission</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card" style="border-color:#F59E0B44;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:#888; font-size:0.75rem; letter-spacing:0.08em;">COMPETITION SCORE (RMSLE)</span><br>
                    <span style="font-size:2rem; font-weight:700; color:#F59E0B;">0.12427</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#888; font-size:0.75rem;">submission.csv</span><br>
                    <span style="font-size:0.85rem; color:#34D399;">Accepted</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">Links</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:0.5rem;">
            <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview"
               target="_blank"
               style="display:inline-block; padding:6px 16px; background:#1A1A1A; border:1px solid #2A2A2A;
                      border-radius:20px; font-size:0.8rem; color:#F5F5F5; text-decoration:none;">
               Kaggle Competition →
            </a>
            <a href="https://github.com/Hartyplaza/homeprice-predict" target="_blank"
               style="display:inline-block; padding:6px 16px; background:#1A1A1A; border:1px solid #2A2A2A;
                      border-radius:20px; font-size:0.8rem; color:#F5F5F5; text-decoration:none;">
               GitHub →
            </a>
            <a href="https://linkedin.com/in/hart-ofigwe" target="_blank"
               style="display:inline-block; padding:6px 16px; background:#1A1A1A; border:1px solid #2A2A2A;
                      border-radius:20px; font-size:0.8rem; color:#F5F5F5; text-decoration:none;">
               LinkedIn →
            </a>
            <a href="https://hartyplaza.github.io" target="_blank"
               style="display:inline-block; padding:6px 16px; background:#1A1A1A; border:1px solid #2A2A2A;
                      border-radius:20px; font-size:0.8rem; color:#F5F5F5; text-decoration:none;">
               Portfolio →
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Stack</div>', unsafe_allow_html=True)
        stack = {
            "Models"      : "Ridge · Lasso · XGBoost",
            "Explainability": "SHAP (TreeExplainer)",
            "Feature Eng" : "RFM · Ordinal encoding · Log1p",
            "Serving"     : "Streamlit Cloud",
            "Language"    : "Python 3.11",
            "Dataset"     : "Ames Housing — Kaggle",
        }
        for k, v in stack.items():
            st.markdown(f"""
            <div class="info-card" style="padding:0.65rem 1rem; margin-bottom:0.5rem;">
                <span style="color:#F59E0B; font-size:0.75rem; font-family:'DM Mono',monospace;
                             letter-spacing:0.08em;">{k}</span><br>
                <span style="font-size:0.9rem;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">Performance</div>', unsafe_allow_html=True)
        from sklearn.metrics import mean_squared_error, r2_score
        ens_p = weights[0]*ridge_pipe.predict(X_train) + weights[1]*lasso_pipe.predict(X_train) + weights[2]*xgb_model.predict(X_train)
        rmse  = np.sqrt(mean_squared_error(y_train, ens_p))
        r2    = r2_score(y_train, ens_p)

        st.markdown(f"""
        <div class="info-card">
            <span style="color:#888; font-size:0.75rem;">ENSEMBLE (train, log1p scale)</span><br>
            <span style="font-size:1.1rem; font-weight:600; color:#F59E0B;">RMSE {rmse:.5f}</span><br>
            <span style="font-size:1.1rem; font-weight:600; color:#34D399;">R² {r2:.4f}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem;">
        <p style="font-size:1.1rem; font-weight:700; margin:0">
            HomePrice<span style="color:#F59E0B">_</span>Predict
        </p>
        <p style="color:#666; font-size:0.78rem; margin:4px 0 0; font-family:'DM Mono',monospace;">
            v1.0 · Ofigwe Hart
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-label">Quick Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card">
        <span style="color:#888; font-size:0.72rem;">TRAINING SAMPLES</span><br>
        <span style="font-size:1.1rem; font-weight:600;">{len(X_train):,}</span>
    </div>
    <div class="info-card">
        <span style="color:#888; font-size:0.72rem;">FEATURES</span><br>
        <span style="font-size:1.1rem; font-weight:600;">{len(feat_cols):,}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Ensemble Weights</div>', unsafe_allow_html=True)
    for name, w in zip(['Ridge', 'Lasso', 'XGBoost'], weights):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:4px 0;
                    border-bottom:1px solid #1E1E1E; font-size:0.85rem;">
            <span style="color:#888">{name}</span>
            <span style="color:#F59E0B; font-family:'DM Mono',monospace">{w:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style="color:#444; font-size:0.72rem; text-align:center; margin-top:1rem;">
        Ames Housing · Kaggle<br>
        github.com/Hartyplaza
    </p>
    """, unsafe_allow_html=True)
