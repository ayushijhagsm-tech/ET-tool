import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import os

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETo Prediction System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        padding: 20px 30px; border-radius: 12px; color: white !important;
        margin-bottom: 24px;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; color: white !important; }
    .main-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.95rem; color: white !important; }
    .result-card {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 12px; padding: 24px; text-align: center;
        margin-top: 16px;
    }
    .result-value { font-size: 3rem; font-weight: 800; color: #ffffff !important; }
    .result-unit  { font-size: 1rem; color: #a5d6a7 !important; }
    .info-box {
        background: rgba(26, 115, 232, 0.15); border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #1a73e8;
        margin-bottom: 12px; font-size: 0.9rem; color: inherit !important;
    }
    .warning-box {
        background: rgba(230, 81, 0, 0.15); border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #ff6d00;
        margin-bottom: 12px; font-size: 0.9rem; color: inherit !important;
    }
    .error-box {
        background: rgba(198, 40, 40, 0.15); border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #ef5350;
        margin-bottom: 12px; font-size: 0.9rem; color: inherit !important;
    }
    .success-box {
        background: rgba(46, 125, 50, 0.15); border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #66bb6a;
        margin-bottom: 12px; font-size: 0.9rem; color: inherit !important;
    }
    .param-badge {
        display: inline-block; background: rgba(26, 115, 232, 0.2);
        color: #90caf9 !important; border: 1px solid #1a73e8;
        border-radius: 20px; padding: 3px 10px;
        font-size: 0.8rem; font-weight: 600; margin: 2px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1) !important;
        color: white !important; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 700; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9 !important; }
    .stButton > button:disabled { background: #555 !important; color: #aaa !important; }
</style>
""", unsafe_allow_html=True)

# ─── ANN CLASS ────────────────────────────────────────────────────────────────
class ANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate, activation='elu'):
        super(ANN, self).__init__()
        acts = {'elu': nn.ELU(), 'relu': nn.ReLU(), 'selu': nn.SELU()}
        layer_list = []
        in_dim = input_dim
        for i, units in enumerate(hidden_layers):
            layer_list.append(nn.Linear(in_dim, units))
            layer_list.append(nn.BatchNorm1d(units))
            layer_list.append(acts[activation])
            if dropout_rate > 0 and i < len(hidden_layers) - 1:
                layer_list.append(nn.Dropout(dropout_rate))
            in_dim = units
        layer_list.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL_DIR    = 'ann_models_pt'
SCALER_DIR   = 'ann_scalers_pt'
SCORES_CSV   = 'model_scores_pt.csv'
R2_THRESHOLD = 0.90
DEVICE       = torch.device('cpu')

FEATURE_META = {
    'n (Sunshine hrs)' : {
        'label': '☀️ Sunshine Hours (n)', 'unit': 'hrs/day',
        'min': 0.0,   'max': 14.0,  'step': 0.1,  'default': 8.0,
        'help': 'Daily sunshine duration in hours (0–14 hrs)',
        'valid_min': 0.0, 'valid_max': 14.0,
        'description': 'Number of sunshine hours per day'
    },
    'Tmax (°C)'        : {
        'label': '🌡️ Max Temperature (Tmax)', 'unit': '°C',
        'min': -10.0, 'max': 50.0,  'step': 0.1,  'default': 30.0,
        'help': 'Maximum daily temperature in °C (-10 to 50)',
        'valid_min': -10.0, 'valid_max': 50.0,
        'description': 'Daily maximum air temperature'
    },
    'Tmin (°C)'        : {
        'label': '🌡️ Min Temperature (Tmin)', 'unit': '°C',
        'min': -20.0, 'max': 40.0,  'step': 0.1,  'default': 15.0,
        'help': 'Minimum daily temperature in °C (-20 to 40)',
        'valid_min': -20.0, 'valid_max': 40.0,
        'description': 'Daily minimum air temperature'
    },
    'RHmax'            : {
        'label': '💧 Max Relative Humidity (RHmax)', 'unit': '%',
        'min': 10.0,  'max': 100.0, 'step': 1.0,  'default': 80.0,
        'help': 'Maximum daily relative humidity (10–100%)',
        'valid_min': 10.0, 'valid_max': 100.0,
        'description': 'Daily maximum relative humidity percentage'
    },
    'RHmin'            : {
        'label': '💧 Min Relative Humidity (RHmin)', 'unit': '%',
        'min': 5.0,   'max': 100.0, 'step': 1.0,  'default': 30.0,
        'help': 'Minimum daily relative humidity (5–100%)',
        'valid_min': 5.0, 'valid_max': 100.0,
        'description': 'Daily minimum relative humidity percentage'
    },
    'u (Windspeed m/s)': {
        'label': '🌬️ Wind Speed (u)', 'unit': 'm/s',
        'min': 0.0,   'max': 10.0,  'step': 0.1,  'default': 1.5,
        'help': 'Average daily wind speed in m/s (0–10)',
        'valid_min': 0.0, 'valid_max': 10.0,
        'description': 'Average daily wind speed at 2m height'
    }
}

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    try:
        scores_df = pd.read_csv(SCORES_CSV)
    except FileNotFoundError:
        st.error(f"❌ '{SCORES_CSV}' not found! Place it in the same folder as app.py")
        st.stop()

    qualified    = scores_df[scores_df['R2_Test'] >= R2_THRESHOLD].copy()
    models_dict  = {}
    scalers_dict = {}
    model_info   = []
    load_errors  = []

    for _, row in qualified.iterrows():
        mid       = row['Model_ID']
        full_path = os.path.join(MODEL_DIR,  f"{mid}_full.pt")
        sx_path   = os.path.join(SCALER_DIR, f"{mid}_scaler_X.pkl")
        sy_path   = os.path.join(SCALER_DIR, f"{mid}_scaler_y.pkl")

        missing = []
        if not os.path.exists(full_path): missing.append(f"{mid}_full.pt")
        if not os.path.exists(sx_path):   missing.append(f"{mid}_scaler_X.pkl")
        if not os.path.exists(sy_path):   missing.append(f"{mid}_scaler_y.pkl")
        if missing:
            load_errors.append(f"⚠️ {mid}: Missing — {', '.join(missing)}")
            continue

        try:
            ckpt  = torch.load(full_path, map_location=DEVICE, weights_only=False)
            model = ANN(ckpt['input_dim'], ckpt['hidden'],
                        ckpt['dropout'],   ckpt['activation'])
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            models_dict[mid] = model

            with open(sx_path, 'rb') as f: sx = pickle.load(f)
            with open(sy_path, 'rb') as f: sy = pickle.load(f)
            scalers_dict[mid] = {
                'scaler_X': sx, 'scaler_y': sy,
                'input_cols': ckpt['input_cols']
            }
            model_info.append({
                'id'        : mid,
                'features'  : row['Input_Features'],
                'n_inputs'  : int(row['Num_Inputs']),
                'r2_test'   : round(float(row['R2_Test']),   4),
                'rmse'      : round(float(row['RMSE_Test']), 4),
                'mae'       : round(float(row['MAE_Test']),  4),
                'nse'       : round(float(row['NSE_Test']),  4),
                'r2_train'  : round(float(row['R2_Train']),  4),
                'rank'      : int(row['Rank']),
                'input_cols': ckpt['input_cols'],
                'epochs_run': int(row['Epochs_Run'])
            })
        except Exception as e:
            load_errors.append(f"❌ {mid}: Load error — {str(e)}")

    model_info.sort(key=lambda x: x['rank'])
    return models_dict, scalers_dict, model_info, load_errors

# ─── VALIDATION ───────────────────────────────────────────────────────────────
def validate_inputs(input_vals):
    errors = []
    for feat, val in input_vals.items():
        meta = FEATURE_META[feat]
        if val < meta['valid_min'] or val > meta['valid_max']:
            errors.append(
                f"• **{meta['label']}**: {val} is out of range "
                f"({meta['valid_min']} – {meta['valid_max']} {meta['unit']})"
            )
    if 'Tmax (°C)' in input_vals and 'Tmin (°C)' in input_vals:
        if input_vals['Tmax (°C)'] <= input_vals['Tmin (°C)']:
            errors.append("• **Tmax must be greater than Tmin**")
    if 'RHmax' in input_vals and 'RHmin' in input_vals:
        if input_vals['RHmax'] <= input_vals['RHmin']:
            errors.append("• **RHmax must be greater than RHmin**")
    return errors

# ─── COLOR FUNCTIONS ──────────────────────────────────────────────────────────
def color_r2(val):
    if val >= 0.97:   return 'background-color: #1b5e20; color: white'
    elif val >= 0.95: return 'background-color: #2e7d32; color: white'
    elif val >= 0.93: return 'background-color: #388e3c; color: white'
    elif val >= 0.91: return 'background-color: #43a047; color: white'
    else:             return 'background-color: #66bb6a; color: white'

def color_rmse(val):
    if val <= 0.35:   return 'background-color: #1b5e20; color: white'
    elif val <= 0.45: return 'background-color: #388e3c; color: white'
    elif val <= 0.55: return 'background-color: #f9a825; color: black'
    else:             return 'background-color: #e53935; color: white'

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌿 ETo Prediction System</h1>
        <p>ANN-based Reference Evapotranspiration Estimation (1970–2020) &nbsp;|&nbsp;
           Models with R² ≥ 0.90 &nbsp;|&nbsp; 5-Layer Deep Neural Network</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        MODELS, SCALERS, MODEL_INFO, load_errors = load_all_models()

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} model(s) could not be loaded"):
            for e in load_errors:
                st.warning(e)

    if not MODELS:
        st.error("❌ No models loaded. Check your model files and folder paths.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["⚡ Predict ETo", "📋 Model Rankings", "ℹ️ Parameter Guide"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — PREDICTION
    # ════════════════════════════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            # 👇 YE NAYA CODE
    st.header("🌾 Irrigation Input Section")

    mode = st.radio("Select Input Mode", ["Manual", "GPS"])

    if mode == "Manual":
        district = st.selectbox("Select District", ["Ludhiana"])
        block = st.selectbox("Select Block", ["Ludhiana-1"])
    else:
        st.info("📍 GPS location will be detected")

    crop = st.selectbox("Select Crop", ["Wheat"])

    variety = st.selectbox("Select Variety", ["PBW-621", "PBW-550", "HD-2967"])

    sowing_date = st.date_input("Select Sowing Date")

    rain_option = st.radio("Rainfall occurred?", ["Yes", "No"])

    if rain_option == "Yes":
        rain = st.number_input("Enter rainfall (mm)", min_value=0.0)
    else:
        rain = 0.0

    method = st.selectbox("Select Irrigation Method", ["Border", "Sprinkler", "Drip"])

    if method == "Border":
        st.number_input("Field Area (ha)")
        st.number_input("Depth (mm)")

    elif method == "Sprinkler":
        st.number_input("Discharge (lph)")
        st.number_input("Spacing (m)")

    elif method == "Drip":
        st.number_input("Emitter spacing (m)")
        st.number_input("Flow rate (lph)")

    st.divider()
            st.subheader("1️⃣ Select Parameters You Have")
            st.markdown("""
            <div class="info-box">
                ℹ️ <strong>Check only the parameters you have data for.</strong><br>
                The app will automatically find the best matching model (R² ≥ 0.90 only).
                Minimum 2 parameters required.
            </div>
            """, unsafe_allow_html=True)

            ALL_FEATURES = list(FEATURE_META.keys())
            chk_cols = st.columns(2)
            selected_features = []
            for i, feat in enumerate(ALL_FEATURES):
                meta = FEATURE_META[feat]
                checked = chk_cols[i % 2].checkbox(
                    f"{meta['label']} ({meta['unit']})",
                    value=True,
                    key=f"chk_{feat}"
                )
                if checked:
                    selected_features.append(feat)

            st.divider()

            # ── AUTO MODEL MATCHING (EXACT ONLY) ──────────────────
            matched_model = None

            if len(selected_features) < 2:
                st.markdown("""
                <div class="error-box">
                ❌ <strong>Minimum 2 parameters required.</strong><br>
                Please select at least 2 parameters.
                </div>""", unsafe_allow_html=True)

            else:
                selected_set  = set(selected_features)

                # EXACT match only — no fallback to lower R² models
                exact_matches = [
                    m for m in MODEL_INFO
                    if set(m['input_cols']) == selected_set
                ]

                if exact_matches:
                    matched_model = min(exact_matches, key=lambda x: x['rank'])
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ <strong>Model found!</strong><br>
                    <strong>{matched_model['id']}</strong> — Rank #{matched_model['rank']} |
                    R² = {matched_model['r2_test']} | RMSE = {matched_model['rmse']}
                    </div>""", unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div class="error-box">
                    ❌ <strong>No model with R² ≥ 0.90 exists for this exact combination.</strong><br>
                    Please adjust your selected parameters.
                    </div>""", unsafe_allow_html=True)

                    # Suggest: add 1 parameter to unlock a model
                    one_add = [
                        m for m in MODEL_INFO
                        if len(set(m['input_cols']) - selected_set) == 1
                        and selected_set.issubset(set(m['input_cols']))
                    ]
                    if one_add:
                        suggestions = []
                        for sm in sorted(one_add, key=lambda x: x['rank'])[:3]:
                            extra       = set(sm['input_cols']) - selected_set
                            extra_label = FEATURE_META[list(extra)[0]]['label'].split('(')[0].strip()
                            suggestions.append(
                                f"• Add **{extra_label}** → unlocks **{sm['id']}** (R²={sm['r2_test']})"
                            )
                        st.markdown(
                            "<div class='info-box'>💡 <strong>Add 1 parameter to unlock a model:</strong><br>"
                            + "<br>".join(suggestions) + "</div>",
                            unsafe_allow_html=True
                        )

                    # Suggest: remove 1 parameter to unlock a model
                    one_remove = [
                        m for m in MODEL_INFO
                        if set(m['input_cols']).issubset(selected_set)
                        and len(selected_set) - len(set(m['input_cols'])) == 1
                    ]
                    if one_remove:
                        suggestions = []
                        for sm in sorted(one_remove, key=lambda x: x['rank'])[:3]:
                            extra       = selected_set - set(sm['input_cols'])
                            extra_label = FEATURE_META[list(extra)[0]]['label'].split('(')[0].strip()
                            suggestions.append(
                                f"• Remove **{extra_label}** → unlocks **{sm['id']}** (R²={sm['r2_test']})"
                            )
                        st.markdown(
                            "<div class='info-box'>💡 <strong>Or remove 1 parameter to unlock a model:</strong><br>"
                            + "<br>".join(suggestions) + "</div>",
                            unsafe_allow_html=True
                        )

                    # No suggestions at all
                    if not one_add and not one_remove:
                        st.markdown("""
                        <div class="info-box">
                        💡 <strong>Try a different combination.</strong><br>
                        Check the <strong>Model Rankings</strong> tab to see all
                        24 valid combinations with R² ≥ 0.90.
                        </div>""", unsafe_allow_html=True)

            # ── STEP 2: INPUT VALUES ───────────────────────────────
            if matched_model:
                st.subheader("2️⃣ Enter Values")
                required_cols = matched_model['input_cols']

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R² Test", matched_model['r2_test'])
                c2.metric("RMSE",    matched_model['rmse'])
                c3.metric("MAE",     matched_model['mae'])
                c4.metric("NSE",     matched_model['nse'])

                badges = " ".join([
                    f'<span class="param-badge">✓ {FEATURE_META[f]["label"].split("(")[0].strip()}</span>'
                    for f in required_cols
                ])
                st.markdown(f"**Model uses:** {badges}", unsafe_allow_html=True)

                input_vals = {}
                n_cols     = 2 if len(required_cols) > 3 else len(required_cols)
                inp_cols   = st.columns(n_cols)

                for i, feat in enumerate(required_cols):
                    meta = FEATURE_META[feat]
                    with inp_cols[i % n_cols]:
                        val = st.number_input(
                            label=f"{meta['label']} ({meta['unit']})",
                            min_value=float(meta['min']),
                            max_value=float(meta['max']),
                            value=float(meta['default']),
                            step=float(meta['step']),
                            help=meta['help'],
                            key=f"val_{matched_model['id']}_{feat}"
                        )
                        input_vals[feat] = val

                val_errors = validate_inputs(input_vals)
                if val_errors:
                    st.markdown(
                        "<div class='error-box'>⚠️ <strong>Fix these errors:</strong><br>"
                        + "<br>".join(val_errors) + "</div>",
                        unsafe_allow_html=True
                    )

                st.divider()
                predict_btn = st.button(
                    "⚡ Predict ETo",
                    disabled=bool(val_errors),
                    use_container_width=True
                )

        # ── RIGHT COLUMN — RESULT ──────────────────────────────────
        with col_right:
            st.subheader("3️⃣ Prediction Result")

            if matched_model and 'predict_btn' in dir() and predict_btn and not val_errors:
                try:
                    mid      = matched_model['id']
                    scaler_X = SCALERS[mid]['scaler_X']
                    scaler_y = SCALERS[mid]['scaler_y']
                    cols_    = SCALERS[mid]['input_cols']

                    vals = np.array([[input_vals[c] for c in cols_]], dtype=np.float32)
                    X_s  = scaler_X.transform(vals).astype(np.float32)
                    X_t  = torch.tensor(X_s)
                    with torch.no_grad():
                        y_s = MODELS[mid](X_t).numpy()
                    eto = float(scaler_y.inverse_transform(y_s).flatten()[0])

                    if eto < 0:
                        st.markdown(f"""
                        <div class='warning-box'>
                        ⚠️ <strong>Predicted ETo is negative ({eto:.4f} mm/day).</strong><br>
                        ETo must be positive. Please check your input values.
                        </div>""", unsafe_allow_html=True)
                    elif eto > 15:
                        st.markdown(f"""
                        <div class='warning-box'>
                        ⚠️ <strong>Predicted ETo is unusually high ({eto:.4f} mm/day).</strong><br>
                        Typical range is 0–15 mm/day. Please verify your inputs.
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="font-size:1rem;font-weight:600;color:#a5d6a7;">
                                Predicted Reference ETo
                            </div>
                            <div class="result-value">{eto:.4f}</div>
                            <div class="result-unit">mm / day</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("**📥 Input Summary**")
                    st.dataframe(pd.DataFrame({
                        'Parameter'  : [FEATURE_META[f]['label'] for f in cols_],
                        'Value'      : [input_vals[f] for f in cols_],
                        'Unit'       : [FEATURE_META[f]['unit']  for f in cols_],
                        'Valid Range': [
                            f"{FEATURE_META[f]['valid_min']}–{FEATURE_META[f]['valid_max']}"
                            for f in cols_
                        ]
                    }), use_container_width=True, hide_index=True)

                    st.info(
                        f"**{mid}** | Rank #{matched_model['rank']} | "
                        f"{len(cols_)} inputs | R²={matched_model['r2_test']} | "
                        f"Trained {matched_model['epochs_run']} epochs"
                    )

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

            elif not matched_model:
                st.markdown("""
                <div class="info-box" style="text-align:center; padding:40px 16px;">
                    👈 Select your available parameters.<br>
                    The best matching model will be found automatically.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box" style="text-align:center; padding:40px 16px;">
                    👈 Enter your values and click <strong>⚡ Predict ETo</strong>
                </div>""", unsafe_allow_html=True)
                st.markdown("**💡 Quick Tips**")
                st.markdown("""
- More parameters = higher R² model used
- All 6-input models give **R² > 0.97**
- Models with **sunshine hours (n)** perform best
                """)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — MODEL RANKINGS
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("📋 All Models with R² ≥ 0.90")

        n_filter = st.multiselect(
            "Filter by number of inputs",
            options=[2, 3, 4, 5, 6],
            default=[2, 3, 4, 5, 6]
        )

        table_data = []
        for info in MODEL_INFO:
            if info['n_inputs'] not in n_filter:
                continue
            table_data.append({
                'Rank'          : info['rank'],
                'Model ID'      : info['id'],
                'Input Features': info['features'],
                'No. Inputs'    : info['n_inputs'],
                'R² Test'       : info['r2_test'],
                'RMSE'          : info['rmse'],
                'MAE'           : info['mae'],
                'NSE'           : info['nse'],
                'R² Train'      : info['r2_train'],
                'Epochs'        : info['epochs_run'],
            })

        if table_data:
            tdf = pd.DataFrame(table_data)
            st.dataframe(
                tdf.style
                   .map(color_r2,   subset=['R² Test'])
                   .map(color_rmse, subset=['RMSE']),
                use_container_width=True,
                hide_index=True,
                height=600
            )
            st.caption(f"Showing {len(table_data)} of {len(MODEL_INFO)} qualifying models")
        else:
            st.warning("No models match the selected filter.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — PARAMETER GUIDE
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("ℹ️ Parameter Guide & Valid Ranges")

        st.markdown("""
        <div class="info-box">
        ℹ️ <strong>Minimum 2 parameters required.</strong>
        Models cover all combinations of 2–6 parameters (57 total, 24 with R² ≥ 0.90).
        </div>
        """, unsafe_allow_html=True)

        for feat, meta in FEATURE_META.items():
            with st.expander(f"{meta['label']}  —  `{meta['unit']}`", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Description:** {meta['description']}")
                    st.markdown(f"**Unit:** `{meta['unit']}`")
                    st.markdown(f"**Valid Range:** `{meta['valid_min']}` to `{meta['valid_max']} {meta['unit']}`")
                with c2:
                    st.markdown(f"**Typical Value:** `{meta['default']} {meta['unit']}`")
                    st.markdown(f"**Tip:** {meta['help']}")
                    count = sum(1 for mi in MODEL_INFO if feat in mi['input_cols'])
                    st.markdown(f"**Used in:** `{count}` qualifying models")

        st.divider()
        st.subheader("⚠️ Common Input Errors")
        st.markdown("""
| Error | Fix |
|---|---|
| Tmax ≤ Tmin | Increase Tmax or decrease Tmin |
| RHmax ≤ RHmin | Increase RHmax or decrease RHmin |
| Sunshine > 14 hrs | Max = 14 hrs/day |
| Wind speed < 0 | Enter value ≥ 0 |
| RH > 100% | Enter value between 5–100% |
| ETo < 0 mm/day | Check all input values are realistic |
| ETo > 15 mm/day | Verify Tmax and wind speed values |
        """)

if __name__ == '__main__':
    main()

