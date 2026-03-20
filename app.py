import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import glob
import os
import zipfile
import tempfile
import io
import re

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Experiment Dashboard", page_icon="🧪")

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* App background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    /* Main content area */
    .main .block-container {
        padding: 2rem 2.5rem 2rem 2.5rem;
        max-width: 1400px;
    }

    /* Title */
    h1 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.6rem !important;
        color: #e6edf3 !important;
        letter-spacing: -0.02em !important;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem !important;
    }

    /* Subheaders */
    h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        color: #e6edf3 !important;
        letter-spacing: -0.01em !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #21262d;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8b949e !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 600 !important;
    }

    /* Sidebar section headers */
    [data-testid="stSidebar"] header {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
    }

    /* Dividers */
    hr {
        border-color: #21262d !important;
        margin: 1rem 0 !important;
    }

    /* Radio buttons */
    .stRadio label {
        color: #c9d1d9 !important;
        font-size: 0.875rem !important;
    }
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem !important;
        color: #c9d1d9 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        font-weight: 400 !important;
    }

    /* Checkboxes */
    .stCheckbox label {
        color: #c9d1d9 !important;
        font-size: 0.875rem !important;
    }

    /* Multiselect / Selectbox */
    .stMultiSelect label, .stSelectbox label {
        color: #8b949e !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1f6feb !important;
        border-radius: 4px !important;
    }
    .stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"] {
        background-color: #21262d !important;
        border-color: #30363d !important;
    }

    /* File uploader */
    .stFileUploader label {
        color: #8b949e !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #161b22 !important;
        border: 1px dashed #30363d !important;
        border-radius: 8px !important;
    }

    /* Info / Warning / Error boxes */
    .stAlert {
        border-radius: 6px !important;
        border: 1px solid #30363d !important;
    }
    [data-testid="stAlert"] {
        background-color: #161b22 !important;
    }

    /* Info box */
    div[data-testid="stAlert"][kind="info"],
    .stInfo {
        background-color: #0d2137 !important;
        border-color: #1f6feb !important;
        color: #79c0ff !important;
    }

    /* Warning box */
    div[data-testid="stAlert"][kind="warning"],
    .stWarning {
        background-color: #1c1700 !important;
        border-color: #9e6a03 !important;
        color: #e3b341 !important;
    }

    /* Error box */
    div[data-testid="stAlert"][kind="error"],
    .stError {
        background-color: #2d0f0f !important;
        border-color: #da3633 !important;
        color: #ff7b72 !important;
    }

    /* Download / Normal Buttons */
    .stDownloadButton button, .stButton button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        transition: all 0.15s ease !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        background-color: #30363d !important;
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }

    /* Plotly chart background */
    .js-plotly-plot .plotly .bg {
        fill: #161b22 !important;
    }

    /* Captions */
    .stCaption {
        color: #8b949e !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* Expander */
    .stExpander {
        border: 1px solid #21262d !important;
        border-radius: 6px !important;
        background-color: #161b22 !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

    /* Scrollable Tabs */
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
        overflow-x: auto !important;
        overflow-y: hidden !important;
        padding-bottom: 8px !important;
        scrollbar-width: thin;
        scrollbar-color: #30363d transparent;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 6px;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: transparent;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background-color: #30363d;
        border-radius: 3px;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab"] {
        white-space: nowrap !important;
    }
    [data-testid="stTabs"] button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        color: #8b949e !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #e6edf3 !important;
        border-bottom-color: #1f6feb !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Consistent Color Palette for Sensors ---
SENSOR_COLORS = {
    'Sys BP (C)': '#1f77b4',  # Muted Blue
    'ITP (D)': '#ff7f0e',  # Safety Orange
    'IAP (E)': '#2ca02c',  # Cooked Asparagus Green
    'PCWP (F)': '#9467bd',  # Muted Purple
    'mRA (G)': '#8c564b'  # Chestnut Brown
}

MULTI_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F'
]


# --- Helper Functions ---

def update_sel(key, vals):
    """Callback helper to safely update session state for quick select buttons."""
    st.session_state[key] = vals


def clean_condition_name(raw_name):
    """
    Cleans messy strings like 'expirium -40 = expirium- 40 = ,,,,,expirium-40 = ,expirium-40 second time ,,,,,,,'
    Extracts strictly to 'Expirium 40' or 'Inspirium 20'.
    """
    name = str(raw_name).lower()
    phase = ""

    if 'insp' in name:
        phase = "Inspirium"
    elif 'exp' in name or 'exs' in name:
        phase = "Expirium"
    else:
        return str(raw_name).strip()  # Return as is if no clear phase detected

    # Search for the first number sequence
    num_match = re.search(r'(\d+)', name)
    if num_match:
        pressure = num_match.group(1)
        return f"{phase} {pressure}"

    return phase


def load_and_process_data_from_dir(directory, agg_mode="30s", subject_id="Unknown"):
    """
    Reads files, processes bins (30s or by respiratory rate), and assigns a subject ID.
    """
    files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    files.sort()

    if not files:
        return pd.DataFrame(), []

    file_metadata = []
    seen_timestamps = set()

    # 1. Scan and Deduplicate
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                for _ in range(3): f.readline()
                f.readline()
                first_data_line = f.readline()

            if first_data_line:
                timestamp_str = first_data_line.split(',')[0]
                if timestamp_str in seen_timestamps:
                    continue
                seen_timestamps.add(timestamp_str)
                file_metadata.append({
                    'path': file_path,
                    'condition_raw': first_line,
                    'timestamp': timestamp_str,
                    'subject_id': subject_id
                })
        except Exception:
            continue

    file_metadata.sort(key=lambda x: x['timestamp'])

    # 2. Process Files
    all_aggregated_data = []
    cumulative_time_offset = 0
    processed_files_info = []

    for meta in file_metadata:
        file_path = meta['path']
        condition_str = meta['condition_raw'].replace("NOTE:", "").strip().replace('"', '')

        try:
            df = pd.read_csv(file_path, skiprows=4)
            if df.empty: continue
        except Exception:
            continue

        # --- Logic: Experiment vs Break ---
        if 'motor_pwm' in df.columns:
            reversed_pwm = df['motor_pwm'][::-1]
            rolling_max = reversed_pwm.rolling(window=101, min_periods=1).max()
            df['is_experiment'] = rolling_max[::-1] > 0
        else:
            df['is_experiment'] = False

        # --- Binning Logic ---
        if 'elapsed [sec]' not in df.columns:
            continue

        resp_col_candidates = [c for c in df.columns if 'resp' in c.lower() and 'rate' in c.lower()]
        has_resp_rate = len(resp_col_candidates) > 0
        if has_resp_rate:
            resp_col = resp_col_candidates[0]
            df['standard_resp_rate'] = pd.to_numeric(df[resp_col], errors='coerce')

        if agg_mode == "resp_rate":
            if has_resp_rate:
                valid_rate = df['standard_resp_rate'].replace(0, pd.NA)
                cycle_sec = 60.0 / valid_rate
                df['time_bin'] = (df['elapsed [sec]'] // cycle_sec).fillna(df['elapsed [sec]'] // 30).astype(int)
            else:
                df['time_bin'] = (df['elapsed [sec]'] // 30).astype(int)
        else:
            df['time_bin'] = (df['elapsed [sec]'] // 30).astype(int)

        # Aggregation
        sensors = ['hx711-C [mmHg]', 'hx711-D [mmHg]', 'hx711-E [mmHg]', 'hx711-F [mmHg]', 'hx711-G [mmHg]']
        available_sensors = [s for s in sensors if s in df.columns]
        agg_rules = {s: ['mean', 'std'] for s in available_sensors}

        if 'applied pressure [mmHg]' in df.columns:
            agg_rules['applied pressure [mmHg]'] = ['max', 'mean']

        if has_resp_rate:
            agg_rules['standard_resp_rate'] = 'mean'

        agg_rules['elapsed [sec]'] = 'mean'
        agg_rules['is_experiment'] = lambda x: x.mode()[0] if not x.mode().empty else False

        df_agg = df.groupby('time_bin').agg(agg_rules).reset_index()

        df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]

        rename_map = {
            'time_bin_': 'time_bin',
            'applied pressure [mmHg]_max': 'pressure_max',
            'applied pressure [mmHg]_mean': 'pressure_mean',
            'elapsed [sec]_mean': 'local_time',
            'is_experiment_<lambda>': 'is_experiment',
            'standard_resp_rate_mean': 'resp_rate'
        }
        df_agg = df_agg.rename(columns=rename_map)

        df_agg['global_time'] = df_agg['local_time'] + cumulative_time_offset
        df_agg['condition'] = condition_str
        df_agg['original_file'] = os.path.basename(file_path)
        df_agg['source_path'] = file_path
        df_agg['subject_id'] = subject_id

        all_aggregated_data.append(df_agg)
        processed_files_info.append(meta)

        if not df['elapsed [sec]'].empty:
            cumulative_time_offset += df['elapsed [sec]'].max() + 30
        else:
            cumulative_time_offset += 30

    if not all_aggregated_data:
        return pd.DataFrame(), []

    final_df = pd.concat(all_aggregated_data, ignore_index=True)
    return final_df, processed_files_info


def get_raw_data_slice(file_path, time_bin, agg_mode="30s"):
    """
    Reloads a specific file and extracts rows for a specific bin.
    """
    try:
        df = pd.read_csv(file_path, skiprows=4)

        if agg_mode == "resp_rate":
            resp_col_candidates = [c for c in df.columns if 'resp' in c.lower() and 'rate' in c.lower()]
            if resp_col_candidates:
                resp_col = resp_col_candidates[0]
                valid_rate = pd.to_numeric(df[resp_col], errors='coerce').replace(0, pd.NA)
                cycle_sec = 60.0 / valid_rate
                df['bin'] = (df['elapsed [sec]'] // cycle_sec).fillna(df['elapsed [sec]'] // 30).astype(int)
            else:
                df['bin'] = (df['elapsed [sec]'] // 30).astype(int)
        else:
            df['bin'] = (df['elapsed [sec]'] // 30).astype(int)

        subset = df[df['bin'] == time_bin].copy()
        return subset
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()


def extract_pre_post_stats(files_df, sensors_to_calc, available_map, window_sec=30, metric="Mean"):
    """
    Extracts exact pre/post statistics based on window_sec.
    Pre = Last window_sec of REST in the PREVIOUS file.
    Post = First window_sec of ACTIVE in the CURRENT file.
    Calculates P-Values per run dynamically using scipy.stats.
    """
    results = []
    subjects = files_df['subject_id'].unique()

    for subj in subjects:
        subj_files = files_df[files_df['subject_id'] == subj]
        paths = subj_files['source_path'].tolist()
        conds = subj_files['condition'].tolist()

        for i in range(1, len(paths)):
            prev_path = paths[i - 1]
            curr_path = paths[i]
            curr_cond = conds[i]
            prev_cond = conds[i - 1]

            try:
                # --- 1. POST (Current File Active Part) ---
                curr_df = pd.read_csv(curr_path, skiprows=4)
                if 'elapsed [sec]' not in curr_df.columns or 'motor_pwm' not in curr_df.columns:
                    continue

                exp_starts = curr_df[curr_df['motor_pwm'] > 0]
                if exp_starts.empty:
                    continue

                t_start_active = exp_starts['elapsed [sec]'].iloc[0]
                post_df = curr_df[(curr_df['elapsed [sec]'] >= t_start_active) &
                                  (curr_df['elapsed [sec]'] <= t_start_active + window_sec)]

                # --- 2. PRE (Previous File Rest Part) ---
                prev_df = pd.read_csv(prev_path, skiprows=4)
                if 'elapsed [sec]' not in prev_df.columns or 'motor_pwm' not in prev_df.columns:
                    pre_df = pd.DataFrame()
                else:
                    rest_part = prev_df[prev_df['motor_pwm'] == 0]
                    if rest_part.empty:
                        rest_part = prev_df  # Fallback
                    t_end_rest = rest_part['elapsed [sec]'].max()
                    pre_df = rest_part[(rest_part['elapsed [sec]'] >= t_end_rest - window_sec) &
                                       (rest_part['elapsed [sec]'] <= t_end_rest)]

                row_data = {
                    'Subject ID': subj,
                    'Condition Raw': curr_cond,
                    'Baseline Source Raw': prev_cond,
                    'File': os.path.basename(curr_path)
                }

                for label in sensors_to_calc:
                    raw_col = available_map[label].replace("_mean", "")
                    if raw_col in curr_df.columns:

                        pre_data = pre_df[
                            raw_col].dropna() if not pre_df.empty and raw_col in pre_df.columns else pd.Series(
                            dtype=float)
                        post_data = post_df[raw_col].dropna() if not post_df.empty else pd.Series(dtype=float)

                        # Core Values (Always keep SD/Min/Max for variance context)
                        if not pre_data.empty:
                            row_data[f'{label} Pre {metric}'] = round(
                                pre_data.mean() if metric == "Mean" else pre_data.median(), 2)
                            row_data[f'{label} Pre SD'] = round(pre_data.std(), 2)
                            row_data[f'{label} Pre Min'] = round(pre_data.min(), 2)
                            row_data[f'{label} Pre Max'] = round(pre_data.max(), 2)
                        else:
                            row_data[f'{label} Pre {metric}'] = pd.NA
                            row_data[f'{label} Pre SD'] = pd.NA
                            row_data[f'{label} Pre Min'] = pd.NA
                            row_data[f'{label} Pre Max'] = pd.NA

                        if not post_data.empty:
                            row_data[f'{label} Post {metric}'] = round(
                                post_data.mean() if metric == "Mean" else post_data.median(), 2)
                            row_data[f'{label} Post SD'] = round(post_data.std(), 2)
                            row_data[f'{label} Post Min'] = round(post_data.min(), 2)
                            row_data[f'{label} Post Max'] = round(post_data.max(), 2)
                        else:
                            row_data[f'{label} Post {metric}'] = pd.NA
                            row_data[f'{label} Post SD'] = pd.NA
                            row_data[f'{label} Post Min'] = pd.NA
                            row_data[f'{label} Post Max'] = pd.NA

                        # Delta & P-Value logic
                        if not pre_data.empty and not post_data.empty:
                            if metric == 'Mean':
                                delta_val = post_data.mean() - pre_data.mean()
                                row_data[f'{label} Δ'] = round(delta_val, 2)
                                row_data[f'{label} Δ SD'] = round((pre_data.std() ** 2 + post_data.std() ** 2) ** 0.5,
                                                                  2)  # Propagation of Error

                                if len(pre_data) > 1 and len(post_data) > 1:
                                    _, p_val = stats.ttest_ind(post_data, pre_data, equal_var=False)
                                else:
                                    p_val = pd.NA
                            else:  # Median
                                delta_val = post_data.median() - pre_data.median()
                                row_data[f'{label} Δ'] = round(delta_val, 2)
                                row_data[f'{label} Δ SD'] = pd.NA

                                if len(pre_data) > 1 and len(post_data) > 1:
                                    _, p_val = stats.mannwhitneyu(post_data, pre_data, alternative='two-sided')
                                else:
                                    p_val = pd.NA

                            if pd.isna(p_val):
                                row_data[f'{label} P-Value'] = "N/A"
                            elif p_val < 0.001:
                                row_data[f'{label} P-Value'] = "<0.001"
                            else:
                                row_data[f'{label} P-Value'] = f"{p_val:.3f}"

                        else:
                            row_data[f'{label} Δ'] = pd.NA
                            row_data[f'{label} Δ SD'] = pd.NA
                            row_data[f'{label} P-Value'] = pd.NA

                results.append(row_data)
            except Exception:
                continue

    return pd.DataFrame(results)


# --- Main App ---

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
    <span style="font-size:1.5rem;">🧪</span>
    <div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1.5rem; font-weight:600; color:#e6edf3; letter-spacing:-0.02em;">Experiment Dashboard</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8b949e; margin-top:2px; letter-spacing:0.04em;">INTERACTIVE PRESSURE SENSOR ANALYSIS</div>
    </div>
</div>
<hr style="border-color:#21262d; margin:0.75rem 0 1.5rem 0;">
""", unsafe_allow_html=True)

# --- Sidebar: Upload ---
with st.sidebar:
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#8b949e;font-weight:600;margin-bottom:0.5rem;">① Upload & Processing</p>',
        unsafe_allow_html=True)

    agg_mode_ui = st.radio(
        "Aggregation Method:",
        ["Fixed 30s Bins", "By Respiratory Cycle (if available)"],
        help="Binning by respiratory cycle helps eliminate respiratory swing artifacts from your pressure readings."
    )
    agg_mode = "resp_rate" if "Respiratory Cycle" in agg_mode_ui else "30s"

    uploaded_zips = st.file_uploader("Upload ZIPs (1 ZIP = 1 Pig)", type="zip", accept_multiple_files=True)

    st.markdown("---")
    with st.expander("ℹ️ Multi-Subject Upload Help"):
        st.write(
            "You can now upload **multiple ZIP files at once**. The dashboard assumes 1 ZIP file = 1 Pig (Subject). "
            "The name of the ZIP file will automatically become the Subject ID in your extracted data."
        )

if not uploaded_zips:
    st.markdown("""
<div style="margin-top:3rem; padding:2.5rem; background:#161b22; border:1px solid #21262d; border-radius:10px; text-align:center; max-width:560px; margin-left:auto; margin-right:auto;">
    <div style="font-size:2.5rem; margin-bottom:1rem;">📂</div>
    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1.1rem; font-weight:500; color:#e6edf3; margin-bottom:0.5rem;">No data loaded</div>
    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.875rem; color:#8b949e;">Upload one or multiple ZIP files containing your experiment CSV files using the panel on the left to begin.</div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# Use a temporary directory to extract files
all_dfs = []
all_files_info = []

with tempfile.TemporaryDirectory() as temp_dir:
    for zip_file in uploaded_zips:
        pig_id = os.path.splitext(zip_file.name)[0]
        pig_dir = os.path.join(temp_dir, pig_id)
        os.makedirs(pig_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(pig_dir)
        except Exception as e:
            st.error(f"Error extracting {zip_file.name}: {e}")
            continue

        df_pig, info_pig = load_and_process_data_from_dir(pig_dir, agg_mode, subject_id=pig_id)
        if not df_pig.empty:
            all_dfs.append(df_pig)
            all_files_info.extend(info_pig)

    if not all_dfs:
        st.error("No valid CSV files found in any of the uploaded ZIPs.")
        st.stop()

    df_all_raw = pd.concat(all_dfs, ignore_index=True)
    files_info = all_files_info

    # --- Sidebar: Controls ---
    with st.sidebar:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#8b949e;font-weight:600;margin-bottom:0.5rem;">② Filters & Settings</p>',
            unsafe_allow_html=True)

        # Global Respiratory Phase Filter (Applies to everything, including Stats table)
        phase_filter = st.radio(
            "Filter by Respiratory Phase:",
            ["All Phases", "Inspirium Only", "Expirium Only"],
            help="Filters globally across all subjects."
        )

        if phase_filter == "Inspirium Only":
            df_all = df_all_raw[df_all_raw['condition'].str.lower().str.contains('insp', na=False)].copy()
        elif phase_filter == "Expirium Only":
            df_all = df_all_raw[df_all_raw['condition'].str.lower().str.contains('exp|exs', na=False)].copy()
        else:
            df_all = df_all_raw.copy()

        if df_all.empty:
            st.warning("No data matches the selected phase filter.")
            st.stop()

        st.markdown("---")

        available_subjects = list(df_all['subject_id'].unique())

        # --- Quick Select Buttons for Subjects ---
        if "subject_multi" not in st.session_state:
            st.session_state.subject_multi = [available_subjects[0]] if available_subjects else []
        else:
            st.session_state.subject_multi = [s for s in st.session_state.subject_multi if s in available_subjects]
            if not st.session_state.subject_multi and available_subjects:
                st.session_state.subject_multi = [available_subjects[0]]

        c1, c2 = st.columns(2)
        c1.button("All Subjects", on_click=update_sel, args=("subject_multi", available_subjects), key="btn_all_subj",
                  use_container_width=True)
        c2.button("Clear Subjects", on_click=update_sel, args=("subject_multi", []), key="btn_clr_subj",
                  use_container_width=True)

        selected_subjects = st.multiselect(
            "Select Subject(s) for Visualization:",
            options=available_subjects,
            key="subject_multi",
            help="Select one or more Pigs to view in the Time-Series chart. (The Stats table below will always calculate for ALL loaded pigs)."
        )

        if not selected_subjects:
            st.warning("Please select at least one subject to visualize.")
            st.stop()

        # Extract subset for visualization
        df_vis = df_all[df_all['subject_id'].isin(selected_subjects)].copy()

        # Batch Selection (Applies only to visualization)
        batch_option = st.radio("Select Visualization Batch:",
                                ["Full Experiment", "Batch 1 (First 7 Files)", "Batch 2 (Last 7 Files)"])

        # Determine target files across ALL selected pigs
        target_files_all_pigs = []
        for subj in selected_subjects:
            subj_files = [os.path.basename(f['path']) for f in files_info if f['subject_id'] == subj]
            if batch_option == "Batch 1 (First 7 Files)":
                target_files_all_pigs.extend(subj_files[:7])
            elif batch_option == "Batch 2 (Last 7 Files)":
                target_files_all_pigs.extend(subj_files[7:])
            else:
                target_files_all_pigs.extend(subj_files)

        df_vis = df_vis[df_vis['original_file'].isin(target_files_all_pigs)].copy()
        df_vis = df_vis.reset_index(drop=True)

        st.markdown("---")

        # Sensor Selection
        sensor_map = {
            'Sys BP (C)': 'hx711-C [mmHg]',
            'ITP (D)': 'hx711-D [mmHg]',
            'IAP (E)': 'hx711-E [mmHg]',
            'PCWP (F)': 'hx711-F [mmHg]',
            'mRA (G)': 'hx711-G [mmHg]'
        }
        available_map = {k: v for k, v in sensor_map.items() if f"{v}_mean" in df_all.columns}

        if not available_map:
            st.error("No expected sensors found.")
            st.stop()

        available_sensor_keys = list(available_map.keys())

        # --- Quick Select Buttons for Sensors ---
        if "sensor_multi" not in st.session_state:
            st.session_state.sensor_multi = [available_sensor_keys[0]] if available_sensor_keys else []
        else:
            st.session_state.sensor_multi = [s for s in st.session_state.sensor_multi if s in available_sensor_keys]
            if not st.session_state.sensor_multi and available_sensor_keys:
                st.session_state.sensor_multi = [available_sensor_keys[0]]

        c1, c2 = st.columns(2)
        c1.button("All Sensors", on_click=update_sel, args=("sensor_multi", available_sensor_keys), key="btn_all_sens",
                  use_container_width=True)
        c2.button("Clear Sensors", on_click=update_sel, args=("sensor_multi", []), key="btn_clr_sens",
                  use_container_width=True)

        selected_sensor_labels = st.multiselect(
            "Select Sensors:",
            options=available_sensor_keys,
            key="sensor_multi"
        )

        if not selected_sensor_labels:
            st.warning("Please select at least one sensor.")
            st.stop()

        show_std = st.checkbox("Show Standard Deviation Bands", value=True)
        plot_style = st.radio("Plot Style:", ["Lines & Markers", "Markers Only"])

        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.8rem; color:#58a6ff; background:#0d2137; border:1px solid #1f6feb; border-radius:5px; padding:0.4rem 0.75rem; margin-top:0.5rem;">Plotting <b>{len(df_vis)}</b> intervals</div>',
            unsafe_allow_html=True)

    # --- Main Graph ---

    if df_vis.empty:
        st.warning(f"No data available to plot under these filters.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Dash styles for distinguishing different pigs
        dash_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']

        # Dynamic Color Allocation Logic
        is_multi_subj = len(selected_subjects) > 1
        num_sensors = len(selected_sensor_labels)


        def get_trace_color(subj_idx, sensor_idx, label, is_pressure=False):
            if not is_multi_subj:
                return '#d62728' if is_pressure else SENSOR_COLORS.get(label, '#1f77b4')
            else:
                # Deterministic unique color offset per (Subject + Sensor) combo
                c_idx = subj_idx * (num_sensors + 1) + (sensor_idx if not is_pressure else num_sensors)
                return MULTI_PALETTE[c_idx % len(MULTI_PALETTE)]


        # Group info logic (background shading) only works well for a single subject.
        # Overlapping backgrounds for multiple subjects gets visually messy.
        if len(selected_subjects) == 1:
            subj = selected_subjects[0]
            df_subj = df_vis[df_vis['subject_id'] == subj]
            df_subj['group_id'] = ((df_subj['condition'] != df_subj['condition'].shift()) | (
                    df_subj['is_experiment'] != df_subj['is_experiment'].shift())).cumsum()
            grouped = df_subj.groupby('group_id')

            for _, group in grouped:
                start_time = group['global_time'].min() - 15
                end_time = group['global_time'].max() + 15
                is_exp = group['is_experiment'].iloc[0]
                condition = group['condition'].iloc[0]

                color = "rgba(0, 128, 0, 0.1)" if is_exp else "rgba(128, 128, 128, 0.1)"
                fig.add_vrect(x0=start_time, x1=end_time, fillcolor=color, layer="below", line_width=0)
                fig.add_annotation(x=(start_time + end_time) / 2, y=1.05, yref="y domain", text=f"{condition}",
                                   showarrow=False, font=dict(size=10))

        # Iterate through selected subjects AND selected sensors
        for s_idx, subj in enumerate(selected_subjects):
            df_subj = df_vis[df_vis['subject_id'] == subj]
            dash_style = dash_styles[s_idx % len(dash_styles)]

            for l_idx, label in enumerate(selected_sensor_labels):
                col_name = available_map[label]
                mean_col = f"{col_name}_mean"
                std_col = f"{col_name}_std"
                hex_color = get_trace_color(s_idx, l_idx, label)

                rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                rgba_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)"

                if show_std and std_col in df_subj.columns:
                    fig.add_trace(
                        go.Scatter(x=df_subj['global_time'], y=df_subj[mean_col] + df_subj[std_col], mode='lines',
                                   line=dict(width=0),
                                   showlegend=False, hoverinfo='skip'), secondary_y=False)
                    fig.add_trace(
                        go.Scatter(x=df_subj['global_time'], y=df_subj[mean_col] - df_subj[std_col], mode='lines',
                                   line=dict(width=0), fill='tonexty',
                                   fillcolor=rgba_color, name=f'{subj} {label} SD', hoverinfo='skip'),
                        secondary_y=False)

                trace_mode = 'lines+markers' if plot_style == "Lines & Markers" else 'markers'
                fig.add_trace(go.Scatter(
                    x=df_subj['global_time'], y=df_subj[mean_col],
                    mode=trace_mode, line=dict(color=hex_color, width=2, dash=dash_style),
                    marker=dict(size=6, color=hex_color),
                    name=f'{subj} - {label} Mean'
                ), secondary_y=False)

            if 'pressure_max' in df_subj.columns:
                p_color = get_trace_color(s_idx, 0, None, is_pressure=True)
                fig.add_trace(go.Scatter(x=df_subj['global_time'], y=df_subj['pressure_max'], mode=trace_mode,
                                         line=dict(color=p_color, width=2, dash=dash_style),
                                         marker=dict(size=4, symbol='diamond'),
                                         name=f'{subj} - Max Pressure'), secondary_y=True)

            if 'resp_rate' in df_subj.columns and df_subj['resp_rate'].notna().any():
                fig.add_trace(go.Scatter(x=df_subj['global_time'], y=df_subj['resp_rate'], mode=trace_mode,
                                         line=dict(color='#8b949e', width=2, dash=dash_style),
                                         marker=dict(size=4, symbol='triangle-up'),
                                         name=f'{subj} - Resp Rate (bpm)'), secondary_y=True)

        title_prefix = "30s" if agg_mode == "30s" else "Respiratory Cycle"
        subj_title = "Multiple Subjects" if len(selected_subjects) > 1 else selected_subjects[0]

        fig.update_layout(
            title=dict(
                text=f"<b>Visualization: {subj_title}</b> — {title_prefix} Averages",
                font=dict(size=14, color="#e6edf3", family="IBM Plex Sans"),
                x=0
            ),
            height=520,
            hovermode="x unified",
            legend=dict(
                orientation="h", y=1.12, x=0,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="#c9d1d9", family="IBM Plex Sans"),
                bordercolor="#30363d", borderwidth=0
            ),
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font=dict(family="IBM Plex Sans", color="#8b949e"),
            margin=dict(l=10, r=10, t=80, b=10),
            xaxis=dict(
                gridcolor="#21262d", gridwidth=1,
                linecolor="#30363d",
                tickfont=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
                title_font=dict(color="#8b949e")
            ),
            yaxis=dict(
                gridcolor="#21262d", gridwidth=1,
                linecolor="#30363d",
                tickfont=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
                title_font=dict(color="#8b949e")
            ),
            hoverlabel=dict(
                bgcolor="#21262d", font_size=12,
                font_family="IBM Plex Mono", font_color="#c9d1d9",
                bordercolor="#30363d"
            )
        )
        fig.update_yaxes(title_text="Sensor Value (mmHg)", secondary_y=False, title_font=dict(color="#8b949e", size=11))
        fig.update_yaxes(title_text="Pressure / Resp Rate", secondary_y=True, title_font=dict(color="#8b949e", size=11))

        if len(selected_subjects) > 1:
            st.info(
                "ℹ️ Background shading for Experiment/Break phases is disabled when viewing multiple subjects to prevent confusing overlaps.")

        # --- Interactive Chart with Selection ---
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.7rem; color:#8b949e; letter-spacing:0.05em; margin-bottom:0.25rem;">CLICK A POINT TO DRILL DOWN ↓</div>',
            unsafe_allow_html=True)
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

        # --- Drill Down Logic ---
        st.divider()

        if event and event['selection']['points']:
            try:
                point_index = event['selection']['points'][0]['point_index']
                # The point index maps exactly to df_vis which contains all currently visualized points
                selected_row = df_vis.iloc[point_index]

                if 'time_bin' not in selected_row:
                    st.error("Error: 'time_bin' data missing.")
                else:
                    target_bin = int(selected_row['time_bin'])
                    target_file = selected_row['source_path']
                    target_condition = selected_row['condition']
                    target_subj = selected_row['subject_id']

                    # Re-calculate index so detail colors match the main graph
                    s_idx = selected_subjects.index(target_subj) if target_subj in selected_subjects else 0

                    st.markdown(f"""
    <div style="margin:1.5rem 0 0.5rem 0; padding:0.75rem 1rem; background:#161b22; border:1px solid #21262d; border-left:3px solid #1f6feb; border-radius:6px;">
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1rem; font-weight:500; color:#e6edf3;">🔍 Detail View — {target_condition}</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8b949e; margin-top:4px;">Bin #{target_bin} | Subject: {target_subj}</div>
    </div>
    """, unsafe_allow_html=True)

                    caption_text = f"Source File: {os.path.basename(target_file)} | Time Bin: {target_bin} (approx {target_bin * 30}-{(target_bin + 1) * 30} sec)" if agg_mode == "30s" else f"Source File: {os.path.basename(target_file)} | Respiratory Cycle: {target_bin}"
                    st.caption(caption_text)

                    raw_slice = get_raw_data_slice(target_file, target_bin, agg_mode)

                    if not raw_slice.empty:
                        fig_detail = make_subplots(specs=[[{"secondary_y": True}]])
                        raw_trace_mode = 'lines' if plot_style == "Lines & Markers" else 'markers'

                        for l_idx, label in enumerate(selected_sensor_labels):
                            raw_sensor_col = available_map[label].replace("_mean", "")
                            hex_color = get_trace_color(s_idx, l_idx, label)

                            if raw_sensor_col in raw_slice.columns:
                                fig_detail.add_trace(go.Scatter(
                                    x=raw_slice['elapsed [sec]'], y=raw_slice[raw_sensor_col],
                                    mode=raw_trace_mode, line=dict(color=hex_color, width=2),
                                    marker=dict(size=4, color=hex_color),
                                    name=f'Raw {label}'
                                ), secondary_y=False)

                        if 'applied pressure [mmHg]' in raw_slice.columns:
                            p_color = get_trace_color(s_idx, 0, None, is_pressure=True)
                            fig_detail.add_trace(go.Scatter(
                                x=raw_slice['elapsed [sec]'], y=raw_slice['applied pressure [mmHg]'],
                                mode=raw_trace_mode, line=dict(color=p_color, width=2),
                                marker=dict(size=4, color=p_color),
                                name='Raw Applied Pressure'
                            ), secondary_y=True)

                        resp_col_candidates = [c for c in raw_slice.columns if
                                               'resp' in c.lower() and 'rate' in c.lower()]
                        if resp_col_candidates:
                            resp_col_name = resp_col_candidates[0]
                            fig_detail.add_trace(go.Scatter(
                                x=raw_slice['elapsed [sec]'],
                                y=pd.to_numeric(raw_slice[resp_col_name], errors='coerce'),
                                mode=raw_trace_mode, line=dict(color='#8b949e', width=2, dash='dot'),
                                marker=dict(size=4, color='#8b949e', symbol='triangle-up'),
                                name='Raw Resp Rate'
                            ), secondary_y=True)

                        fig_detail.update_layout(
                            height=400, hovermode="x unified", showlegend=True,
                            paper_bgcolor="#161b22",
                            plot_bgcolor="#0d1117",
                            font=dict(family="IBM Plex Sans", color="#8b949e"),
                            legend=dict(
                                bgcolor="rgba(0,0,0,0)",
                                font=dict(size=11, color="#c9d1d9", family="IBM Plex Sans"),
                                bordercolor="#30363d", borderwidth=0
                            ),
                            margin=dict(l=10, r=10, t=20, b=10),
                            xaxis=dict(
                                gridcolor="#21262d", linecolor="#30363d",
                                tickfont=dict(family="IBM Plex Mono", size=11, color="#8b949e")
                            ),
                            yaxis=dict(
                                gridcolor="#21262d", linecolor="#30363d",
                                tickfont=dict(family="IBM Plex Mono", size=11, color="#8b949e")
                            ),
                            hoverlabel=dict(
                                bgcolor="#21262d", font_size=12,
                                font_family="IBM Plex Mono", font_color="#c9d1d9",
                                bordercolor="#30363d"
                            )
                        )
                        fig_detail.update_xaxes(title_text="Time within File (sec)",
                                                title_font=dict(color="#8b949e", size=11))
                        fig_detail.update_yaxes(title_text="Sensor Raw (mmHg)", secondary_y=False,
                                                title_font=dict(color="#8b949e", size=11))
                        fig_detail.update_yaxes(title_text="Pressure / Resp Rate", secondary_y=True,
                                                title_font=dict(color="#d62728", size=11))

                        st.plotly_chart(fig_detail, use_container_width=True)

                    else:
                        st.warning("Could not retrieve raw data for this timestamp.")

            except Exception as e:
                st.error(f"Error displaying detail view: {e}")

        else:
            st.markdown("""
    <div style="margin-top:1rem; padding:1rem 1.25rem; background:#161b22; border:1px dashed #30363d; border-radius:6px; text-align:center;">
        <span style="font-size:1.2rem;">👆</span>
        <span style="font-family:'IBM Plex Sans',sans-serif; font-size:0.875rem; color:#8b949e; margin-left:8px;">Click on a data point in the chart above to drill into raw second-by-second data for that interval.</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Pre/Post Statistical Summary ---
    st.divider()
    st.markdown("""
<div style="margin:1.5rem 0 0.5rem 0; padding:0.75rem 1rem; background:#161b22; border:1px solid #21262d; border-left:3px solid #2ca02c; border-radius:6px;">
    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1rem; font-weight:500; color:#e6edf3;">📊 Master Pre/Post Statistical Summary (All Subjects)</div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8b949e; margin-top:4px;">ADJACENT EXPERIMENT COMPARISON</div>
</div>
""", unsafe_allow_html=True)

    st.warning(
        "⚠️ **Clinical Statistics Note:** The individual *P-Values* calculated below compare the high-frequency data array of the active window against the baseline window for a *single* run. Due to the immense number of data points ($n \\approx 3000$ per 30s) and time-series autocorrelation, these individual p-values are highly susceptible to significance inflation. For publication claims, rely on cohort-level repeated-measures tests across multiple pigs rather than individual intra-run p-values.")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        stat_window_sel = st.selectbox("Analysis Window (Duration before/after trigger):",
                                       ["30 seconds", "60 seconds (1 minute)"])
        window_sec = 60 if "60" in stat_window_sel else 30
    with col_ctrl2:
        stat_metric = st.selectbox("Central Tendency Metric:", ["Mean", "Median"])

    st.caption(
        f"Calculates stats by comparing the **last {window_sec}s of REST in the PREVIOUS file** against the **first {window_sec}s of ACTIVE intervention in the CURRENT file**. Results are filtered by your current respiratory phase selection.")

    # We pull files from df_all_raw (which contains ALL files, unfiltered, to preserve chronological order for baselines)
    stats_input_df = df_all_raw[['source_path', 'subject_id', 'condition']].drop_duplicates()

    if not stats_input_df.empty:
        with st.spinner(
                f"Calculating rigorous Pre/Post statistics across all subjects ({stat_metric}, {window_sec}s)..."):
            stats_df = extract_pre_post_stats(stats_input_df, selected_sensor_labels, available_map, window_sec,
                                              stat_metric)

        if not stats_df.empty:

            # Apply regex text cleaning to standardise 'Expirium 40', 'Inspirium 80', etc.
            stats_df['Clean Condition'] = stats_df['Condition Raw'].apply(clean_condition_name)

            # --- Visualizations Settings UI ---
            st.markdown("""
            <div style="margin:2.5rem 0 1rem 0; padding:0.75rem 1rem; background:#161b22; border:1px solid #21262d; border-left:3px solid #9467bd; border-radius:6px;">
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1rem; font-weight:500; color:#e6edf3;">📈 Statistical Visualizations (Bar Graphs & Box Plots)</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8b949e; margin-top:4px;">MULTI-RUN PAIRED COMPARISONS & COHORT VARIABILITY</div>
            </div>
            """, unsafe_allow_html=True)

            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                stat_sensor = st.selectbox("Select Sensor to Visualize:", selected_sensor_labels,
                                           key="stat_sensor_select")
            with col_v2:
                st.write("")  # Spacing to align with selectbox
                use_raw_names = st.checkbox("Use Raw (Uncleaned) Condition Names",
                                            help="Toggle this to use the original messy text from the CSV files instead of the auto-cleaned versions.")

            cond_col = 'Condition Raw' if use_raw_names else 'Clean Condition'

            # To handle multiple runs of the same condition for the same pig
            stats_df['Run ID'] = stats_df.groupby(['Subject ID', cond_col]).cumcount() + 1
            stats_df['Intervention Label'] = stats_df[cond_col] + " (Run " + stats_df['Run ID'].astype(str) + ")"

            # Apply UI phase filter to the output so we only see the relevant rows
            if phase_filter == "Inspirium Only":
                stats_df = stats_df[stats_df['Condition Raw'].str.lower().str.contains('insp', na=False)]
            elif phase_filter == "Expirium Only":
                stats_df = stats_df[stats_df['Condition Raw'].str.lower().str.contains('exp|exs', na=False)]

            st.dataframe(stats_df, use_container_width=True)

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                f"📊 1. Overall Delta ({stat_metric} Per Pig)",
                f"📊 2. Before vs After ({stat_metric} Per Pig)",
                f"📊 3. Intervention Comparison ({stat_metric} Across Cohort)",
                f"📊 4. Cohort Before vs After (Multi-Intervention)",
                f"📊 5. Cohort Deltas (Multi-Intervention)"
            ])

            # TAB 1: Delta per Pig
            with tab1:
                sel_pig_1 = st.selectbox("Select Subject:", available_subjects, key="t1_pig")
                pig_df_1 = stats_df[stats_df['Subject ID'] == sel_pig_1].dropna(subset=[f'{stat_sensor} Δ'])

                if not pig_df_1.empty:
                    s_idx = available_subjects.index(sel_pig_1) if sel_pig_1 in available_subjects else 0
                    p_color = MULTI_PALETTE[s_idx % len(MULTI_PALETTE)]

                    hover_texts = pig_df_1.apply(lambda r: (
                        f"<b>{r['Intervention Label']}</b><br>"
                        f"Δ {stat_sensor}: {r[f'{stat_sensor} Δ']}<br>"
                        f"P-Value (intra-run): {r[f'{stat_sensor} P-Value']}<br>"
                        f"---<br>"
                        f"Pre (Rest): {r[f'{stat_sensor} Pre {stat_metric}']} (Min: {r[f'{stat_sensor} Pre Min']}, Max: {r[f'{stat_sensor} Pre Max']})<br>"
                        f"Post (Active): {r[f'{stat_sensor} Post {stat_metric}']} (Min: {r[f'{stat_sensor} Post Min']}, Max: {r[f'{stat_sensor} Post Max']})"
                    ), axis=1)

                    fig1 = go.Figure()

                    fig1.add_trace(go.Bar(
                        x=pig_df_1['Intervention Label'],
                        y=pig_df_1[f'{stat_sensor} Δ'],
                        marker_color=p_color,
                        text=pig_df_1[f'{stat_sensor} Δ'].apply(lambda x: f"{x:.1f}"),
                        textposition='auto',
                        hovertext=hover_texts,
                        hoverinfo='text'
                    ))

                    fig1.update_layout(
                        title=dict(
                            text=f"<b>Delta (Δ) for {sel_pig_1}</b><br><sup>Shows all individual intervention runs ({stat_metric})</sup>",
                            font=dict(size=14, color="#e6edf3", family="IBM Plex Sans")),
                        yaxis_title=f"Δ {stat_sensor} ({stat_metric} mmHg)",
                        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                        font=dict(family="IBM Plex Sans", color="#8b949e"),
                        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=False),
                        yaxis=dict(gridcolor="#21262d", linecolor="#30363d")
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No delta data available for this subject.")

            # TAB 2: Before vs After per Pig
            with tab2:
                sel_pig_2 = st.selectbox("Select Subject:", available_subjects, key="t2_pig")
                pig_df_2 = stats_df[stats_df['Subject ID'] == sel_pig_2].dropna(
                    subset=[f'{stat_sensor} Pre {stat_metric}', f'{stat_sensor} Post {stat_metric}'])

                if not pig_df_2.empty:
                    s_idx = available_subjects.index(sel_pig_2) if sel_pig_2 in available_subjects else 0
                    p_color = MULTI_PALETTE[s_idx % len(MULTI_PALETTE)]

                    hover_pre = pig_df_2.apply(lambda
                                                   r: f"Pre {stat_metric}: {r[f'{stat_sensor} Pre {stat_metric}']}<br>SD: {r[f'{stat_sensor} Pre SD']}<br>Min: {r[f'{stat_sensor} Pre Min']}<br>Max: {r[f'{stat_sensor} Pre Max']}",
                                               axis=1)
                    hover_post = pig_df_2.apply(lambda
                                                    r: f"Post {stat_metric}: {r[f'{stat_sensor} Post {stat_metric}']}<br>SD: {r[f'{stat_sensor} Post SD']}<br>P-Value (vs Baseline): {r[f'{stat_sensor} P-Value']}<br>Min: {r[f'{stat_sensor} Post Min']}<br>Max: {r[f'{stat_sensor} Post Max']}",
                                                axis=1)

                    fig2 = go.Figure()

                    kwargs_err_pre = dict(type='data', array=pig_df_2[f'{stat_sensor} Pre SD'], visible=True,
                                          color="#c9d1d9", thickness=1.5) if stat_metric == 'Mean' else None
                    kwargs_err_post = dict(type='data', array=pig_df_2[f'{stat_sensor} Post SD'], visible=True,
                                           color="#c9d1d9", thickness=1.5) if stat_metric == 'Mean' else None

                    fig2.add_trace(go.Bar(
                        x=pig_df_2['Intervention Label'],
                        y=pig_df_2[f'{stat_sensor} Pre {stat_metric}'],
                        name='Baseline (Pre)',
                        marker_color='#8b949e',
                        error_y=kwargs_err_pre,
                        hovertext=hover_pre,
                        hoverinfo='text+x+name'
                    ))
                    fig2.add_trace(go.Bar(
                        x=pig_df_2['Intervention Label'],
                        y=pig_df_2[f'{stat_sensor} Post {stat_metric}'],
                        name='Intervention (Post)',
                        marker_color=p_color,
                        error_y=kwargs_err_post,
                        hovertext=hover_post,
                        hoverinfo='text+x+name'
                    ))

                    sd_note = " with Standard Deviation" if stat_metric == 'Mean' else ""
                    fig2.update_layout(
                        barmode='group',
                        title=dict(
                            text=f"<b>Before vs After for {sel_pig_2}</b><br><sup>Shows individual intervention runs{sd_note}</sup>",
                            font=dict(size=14, color="#e6edf3", family="IBM Plex Sans")),
                        yaxis_title=f"{stat_sensor} {stat_metric} (mmHg)",
                        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                        font=dict(family="IBM Plex Sans", color="#8b949e"),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
                        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=False),
                        yaxis=dict(gridcolor="#21262d", linecolor="#30363d")
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No Pre/Post data available for this subject.")

            # TAB 3: Per Intervention across Pigs
            with tab3:
                available_conds = sorted(stats_df[cond_col].astype(str).unique())
                if available_conds:
                    sel_cond = st.selectbox("Select Intervention:", available_conds, key="t3_cond")
                    cond_df = stats_df[stats_df[cond_col] == sel_cond].dropna(subset=[f'{stat_sensor} Δ'])

                    if not cond_df.empty:
                        cond_df['Pig_Run'] = cond_df['Subject ID'] + " (Run " + cond_df['Run ID'].astype(str) + ")"

                        fig3 = go.Figure()

                        for subj in cond_df['Subject ID'].unique():
                            s_df = cond_df[cond_df['Subject ID'] == subj]
                            s_idx = available_subjects.index(subj) if subj in available_subjects else 0
                            p_color = MULTI_PALETTE[s_idx % len(MULTI_PALETTE)]

                            hover_texts = s_df.apply(lambda r: (
                                f"<b>{r['Subject ID']} - Run {r['Run ID']}</b><br>"
                                f"Δ {stat_sensor}: {r[f'{stat_sensor} Δ']}<br>"
                                f"P-Value (intra-run): {r[f'{stat_sensor} P-Value']}<br>"
                                f"---<br>"
                                f"Pre: {r[f'{stat_sensor} Pre {stat_metric}']} (Min: {r[f'{stat_sensor} Pre Min']}, Max: {r[f'{stat_sensor} Pre Max']})<br>"
                                f"Post: {r[f'{stat_sensor} Post {stat_metric}']} (Min: {r[f'{stat_sensor} Post Min']}, Max: {r[f'{stat_sensor} Post Max']})"
                            ), axis=1)

                            fig3.add_trace(go.Bar(
                                name=subj,
                                x=s_df['Pig_Run'],
                                y=s_df[f'{stat_sensor} Δ'],
                                marker_color=p_color,
                                text=s_df[f'{stat_sensor} Δ'].apply(lambda x: f"{x:.1f}"),
                                textposition='auto',
                                hovertext=hover_texts,
                                hoverinfo='text'
                            ))

                        fig3.update_layout(
                            barmode='group',
                            title=dict(
                                text=f"<b>Delta (Δ) for '{sel_cond}' Across Cohort</b><br><sup>Shows all individual runs across all selected subjects ({stat_metric})</sup>",
                                font=dict(size=14, color="#e6edf3", family="IBM Plex Sans")),
                            yaxis_title=f"Δ {stat_sensor} (mmHg)",
                            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                            font=dict(family="IBM Plex Sans", color="#8b949e"),
                            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
                            xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=False),
                            yaxis=dict(gridcolor="#21262d", linecolor="#30363d")
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("No delta data available for this intervention.")
                else:
                    st.info("No conditions available.")

            # TAB 4: Cohort Before vs After (Multi-Intervention)
            with tab4:
                available_conds_4 = sorted(stats_df[cond_col].astype(str).unique())
                if available_conds_4:

                    if "t4_conds" not in st.session_state:
                        st.session_state.t4_conds = available_conds_4[:2] if len(
                            available_conds_4) >= 2 else available_conds_4
                    else:
                        st.session_state.t4_conds = [c for c in st.session_state.t4_conds if c in available_conds_4]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.button("All", on_click=update_sel, args=("t4_conds", available_conds_4), key="btn_all_t4",
                              use_container_width=True)
                    c2.button("Insp Only", on_click=update_sel,
                              args=("t4_conds", [c for c in available_conds_4 if "insp" in c.lower()]),
                              key="btn_insp_t4", use_container_width=True)
                    c3.button("Exp Only", on_click=update_sel, args=("t4_conds", [c for c in available_conds_4 if
                                                                                  "exp" in c.lower() or "exs" in c.lower()]),
                              key="btn_exp_t4", use_container_width=True)
                    c4.button("Clear", on_click=update_sel, args=("t4_conds", []), key="btn_clr_t4",
                              use_container_width=True)

                    sel_conds_4 = st.multiselect(
                        "Select Interventions to Compare side-by-side:",
                        available_conds_4,
                        key="t4_conds"
                    )

                    if sel_conds_4:
                        cond_df_4 = stats_df[stats_df[cond_col].isin(sel_conds_4)].dropna(
                            subset=[f'{stat_sensor} Pre {stat_metric}', f'{stat_sensor} Post {stat_metric}'])

                        if not cond_df_4.empty:
                            fig4 = go.Figure()
                            added_legends = set()

                            for cond in sel_conds_4:
                                cdf = cond_df_4[cond_df_4[cond_col] == cond]
                                if cdf.empty:
                                    continue

                                x_pre = f"{cond}<br>Pre"
                                x_post = f"{cond}<br>Post"

                                # 1. Add Box Plots for overall cohort statistics (Median, Q1, Q3, Min, Max)
                                fig4.add_trace(go.Box(
                                    y=cdf[f'{stat_sensor} Pre {stat_metric}'],
                                    x=[x_pre] * len(cdf),
                                    name=f'Cohort Pre ({cond})',
                                    boxmean=True,  # Shows mean as a dashed line inside the box
                                    fillcolor='rgba(139, 148, 158, 0.2)',
                                    line=dict(color='#8b949e'),
                                    showlegend=False,
                                    hoverinfo='y'
                                ))
                                fig4.add_trace(go.Box(
                                    y=cdf[f'{stat_sensor} Post {stat_metric}'],
                                    x=[x_post] * len(cdf),
                                    name=f'Cohort Post ({cond})',
                                    boxmean=True,
                                    fillcolor='rgba(139, 148, 158, 0.2)',
                                    line=dict(color='#8b949e'),
                                    showlegend=False,
                                    hoverinfo='y'
                                ))

                                # 2. Add individual paired lines for every run, colored by Pig
                                for idx, row in cdf.iterrows():
                                    subj = row['Subject ID']
                                    s_idx = available_subjects.index(subj) if subj in available_subjects else 0
                                    p_color = MULTI_PALETTE[s_idx % len(MULTI_PALETTE)]

                                    show_leg = subj not in added_legends
                                    if show_leg:
                                        added_legends.add(subj)

                                    hover_text = (f"<b>{subj} - Run {row['Run ID']}</b><br>"
                                                  f"Condition: {cond}<br>"
                                                  f"Pre: {row[f'{stat_sensor} Pre {stat_metric}']}<br>"
                                                  f"Post: {row[f'{stat_sensor} Post {stat_metric}']}<br>"
                                                  f"Δ: {row[f'{stat_sensor} Δ']}")

                                    fig4.add_trace(go.Scatter(
                                        x=[x_pre, x_post],
                                        y=[row[f'{stat_sensor} Pre {stat_metric}'],
                                           row[f'{stat_sensor} Post {stat_metric}']],
                                        mode='lines+markers',
                                        line=dict(color=p_color, width=2),
                                        marker=dict(size=8, symbol='circle', line=dict(width=1, color="#161b22")),
                                        name=subj,
                                        showlegend=show_leg,
                                        hovertext=hover_text,
                                        hoverinfo='text'
                                    ))

                            fig4.update_layout(
                                title=dict(
                                    text=f"<b>Cohort Before vs After (Multi-Intervention)</b><br><sup>Shows individual runs + overall cohort Box Plot (Median, IQR, Mean, Min/Max)</sup>",
                                    font=dict(size=14, color="#e6edf3", family="IBM Plex Sans")),
                                yaxis_title=f"{stat_sensor} {stat_metric} (mmHg)",
                                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                font=dict(family="IBM Plex Sans", color="#8b949e"),
                                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
                                xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=False),
                                yaxis=dict(gridcolor="#21262d", linecolor="#30363d")
                            )
                            st.plotly_chart(fig4, use_container_width=True)
                        else:
                            st.info("No paired data available for the selected interventions.")
                    else:
                        st.info("Please select at least one intervention.")
                else:
                    st.info("No conditions available.")

            # TAB 5: Cohort Deltas (Multi-Intervention)
            with tab5:
                available_conds_5 = sorted(stats_df[cond_col].astype(str).unique())
                if available_conds_5:

                    if "t5_conds" not in st.session_state:
                        st.session_state.t5_conds = available_conds_5[:2] if len(
                            available_conds_5) >= 2 else available_conds_5
                    else:
                        st.session_state.t5_conds = [c for c in st.session_state.t5_conds if c in available_conds_5]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.button("All", on_click=update_sel, args=("t5_conds", available_conds_5), key="btn_all_t5",
                              use_container_width=True)
                    c2.button("Insp Only", on_click=update_sel,
                              args=("t5_conds", [c for c in available_conds_5 if "insp" in c.lower()]),
                              key="btn_insp_t5", use_container_width=True)
                    c3.button("Exp Only", on_click=update_sel, args=("t5_conds", [c for c in available_conds_5 if
                                                                                  "exp" in c.lower() or "exs" in c.lower()]),
                              key="btn_exp_t5", use_container_width=True)
                    c4.button("Clear", on_click=update_sel, args=("t5_conds", []), key="btn_clr_t5",
                              use_container_width=True)

                    sel_conds_5 = st.multiselect(
                        "Select Interventions to Compare side-by-side:",
                        available_conds_5,
                        key="t5_conds"
                    )

                    if sel_conds_5:
                        cond_df_5 = stats_df[stats_df[cond_col].isin(sel_conds_5)].dropna(subset=[f'{stat_sensor} Δ'])

                        if not cond_df_5.empty:
                            fig5 = go.Figure()
                            added_legends = set()

                            # 1. Add Box Plots for overall cohort delta statistics
                            for cond in sel_conds_5:
                                cdf = cond_df_5[cond_df_5[cond_col] == cond]
                                if cdf.empty:
                                    continue

                                fig5.add_trace(go.Box(
                                    y=cdf[f'{stat_sensor} Δ'],
                                    x=[cond] * len(cdf),
                                    name=f'Cohort Δ ({cond})',
                                    boxmean=True,
                                    fillcolor='rgba(139, 148, 158, 0.2)',
                                    line=dict(color='#8b949e'),
                                    showlegend=False,
                                    hoverinfo='y'
                                ))

                            # 2. Add individual points for every run, colored by Pig
                            for idx, row in cond_df_5.iterrows():
                                cond = row[cond_col]
                                subj = row['Subject ID']
                                s_idx = available_subjects.index(subj) if subj in available_subjects else 0
                                p_color = MULTI_PALETTE[s_idx % len(MULTI_PALETTE)]

                                show_leg = subj not in added_legends
                                if show_leg:
                                    added_legends.add(subj)

                                hover_text = (f"<b>{subj} - Run {row['Run ID']}</b><br>"
                                              f"Condition: {cond}<br>"
                                              f"Δ: {row[f'{stat_sensor} Δ']}<br>"
                                              f"Pre: {row[f'{stat_sensor} Pre {stat_metric}']}<br>"
                                              f"Post: {row[f'{stat_sensor} Post {stat_metric}']}")

                                fig5.add_trace(go.Scatter(
                                    x=[cond],
                                    y=[row[f'{stat_sensor} Δ']],
                                    mode='markers',
                                    marker=dict(color=p_color, size=8, symbol='circle',
                                                line=dict(width=1, color="#161b22")),
                                    name=subj,
                                    showlegend=show_leg,
                                    hovertext=hover_text,
                                    hoverinfo='text'
                                ))

                            fig5.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)

                            fig5.update_layout(
                                title=dict(
                                    text=f"<b>Cohort Deltas (Multi-Intervention)</b><br><sup>Shows individual runs + overall cohort Box Plot for Δ (Median, IQR, Mean, Min/Max)</sup>",
                                    font=dict(size=14, color="#e6edf3", family="IBM Plex Sans")),
                                yaxis_title=f"Δ {stat_sensor} ({stat_metric} mmHg)",
                                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                font=dict(family="IBM Plex Sans", color="#8b949e"),
                                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
                                xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=False),
                                yaxis=dict(gridcolor="#21262d", linecolor="#30363d")
                            )
                            st.plotly_chart(fig5, use_container_width=True)
                        else:
                            st.info("No delta data available for the selected interventions.")
                    else:
                        st.info("Please select at least one intervention.")
                else:
                    st.info("No conditions available.")

            # --- Downloads for Stats ---
            csv_stats = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button("📈 Download Multi-Subject Stats for SPSS/R (CSV)", csv_stats,
                               "pre_post_stats_all_subjects.csv", "text/csv")
        else:
            st.info("No experiment triggers (`motor_pwm > 0`) were found in the currently loaded files.")
    else:
        st.info("No data available to calculate statistics.")

    # --- Downloads (Sidebar) ---
    with st.sidebar:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#8b949e;font-weight:600;margin-bottom:0.5rem;">③ Downloads</p>',
            unsafe_allow_html=True)
        # Download all filtered aggregated data
        csv_buffer = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Master Data (All Pigs)", csv_buffer, "processed_data_master.csv", "text/csv")
