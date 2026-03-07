import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
import zipfile
import tempfile
import io

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

    /* Download button */
    .stDownloadButton button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        transition: all 0.15s ease !important;
    }
    .stDownloadButton button:hover {
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

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
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


# --- Helper Functions ---

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

        if agg_mode == "resp_rate":
            resp_col_candidates = [c for c in df.columns if 'resp' in c.lower() and 'rate' in c.lower()]
            if resp_col_candidates:
                resp_col = resp_col_candidates[0]
                valid_rate = pd.to_numeric(df[resp_col], errors='coerce').replace(0, pd.NA)
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

        agg_rules['elapsed [sec]'] = 'mean'
        agg_rules['is_experiment'] = lambda x: x.mode()[0] if not x.mode().empty else False

        df_agg = df.groupby('time_bin').agg(agg_rules).reset_index()

        df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]

        rename_map = {
            'time_bin_': 'time_bin',
            'applied pressure [mmHg]_max': 'pressure_max',
            'applied pressure [mmHg]_mean': 'pressure_mean',
            'elapsed [sec]_mean': 'local_time',
            'is_experiment_<lambda>': 'is_experiment'
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


def extract_pre_post_stats(files_df, sensors_to_calc, available_map):
    """
    Extracts exact 30-sec pre/post statistics based on a DataFrame of target files.
    """
    results = []
    for _, row in files_df.iterrows():
        fpath = row['source_path']
        subj = row['subject_id']
        cond = row['condition']

        try:
            raw_df = pd.read_csv(fpath, skiprows=4)
            if 'elapsed [sec]' not in raw_df.columns or 'motor_pwm' not in raw_df.columns:
                continue

            # Exact millisecond the experiment begins
            exp_starts = raw_df[raw_df['motor_pwm'] > 0]
            if exp_starts.empty:
                continue

            t_start = exp_starts['elapsed [sec]'].iloc[0]

            # 30 seconds strictly before, and 30 seconds strictly after trigger
            pre_df = raw_df[(raw_df['elapsed [sec]'] >= t_start - 30) & (raw_df['elapsed [sec]'] < t_start)]
            post_df = raw_df[(raw_df['elapsed [sec]'] >= t_start) & (raw_df['elapsed [sec]'] <= t_start + 30)]

            row_data = {
                'Subject ID': subj,
                'Condition': cond,
                'File': os.path.basename(fpath),
                'Start Trigger [sec]': round(t_start, 1)
            }

            for label in sensors_to_calc:
                raw_col = available_map[label].replace("_mean", "")
                if raw_col in raw_df.columns:
                    # Baseline (Pre)
                    if not pre_df.empty:
                        row_data[f'{label} Pre Mean'] = round(pre_df[raw_col].mean(), 2)
                        row_data[f'{label} Pre SD'] = round(pre_df[raw_col].std(), 2)
                        row_data[f'{label} Pre Min'] = round(pre_df[raw_col].min(), 2)
                        row_data[f'{label} Pre Max'] = round(pre_df[raw_col].max(), 2)
                    else:
                        row_data[f'{label} Pre Mean'] = pd.NA
                        row_data[f'{label} Pre SD'] = pd.NA
                        row_data[f'{label} Pre Min'] = pd.NA
                        row_data[f'{label} Pre Max'] = pd.NA

                    # Experiment (Post)
                    if not post_df.empty:
                        row_data[f'{label} Post Mean'] = round(post_df[raw_col].mean(), 2)
                        row_data[f'{label} Post SD'] = round(post_df[raw_col].std(), 2)
                        row_data[f'{label} Post Min'] = round(post_df[raw_col].min(), 2)
                        row_data[f'{label} Post Max'] = round(post_df[raw_col].max(), 2)
                    else:
                        row_data[f'{label} Post Mean'] = pd.NA
                        row_data[f'{label} Post SD'] = pd.NA
                        row_data[f'{label} Post Min'] = pd.NA
                        row_data[f'{label} Post Max'] = pd.NA

                    # Delta
                    if not pre_df.empty and not post_df.empty:
                        row_data[f'{label} Δ (Post - Pre)'] = round(post_df[raw_col].mean() - pre_df[raw_col].mean(), 2)
                    else:
                        row_data[f'{label} Δ (Post - Pre)'] = pd.NA

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
            df_all = df_all_raw[df_all_raw['condition'].str.lower().str.contains('exp', na=False)].copy()
        else:
            df_all = df_all_raw.copy()

        if df_all.empty:
            st.warning("No data matches the selected phase filter.")
            st.stop()

        st.markdown("---")

        available_subjects = list(df_all['subject_id'].unique())

        # Multi-select to allow viewing multiple pigs
        selected_subjects = st.multiselect(
            "Select Subject(s) for Visualization:",
            options=available_subjects,
            default=[available_subjects[0]],
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

        selected_sensor_labels = st.multiselect(
            "Select Sensors:",
            options=list(available_map.keys()),
            default=[list(available_map.keys())[0]]
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
        MULTI_PALETTE = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F'
        ]
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
        fig.update_yaxes(title_text="Pressure (mmHg)", secondary_y=True, title_font=dict(color="#8b949e", size=11))

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
                        fig_detail.update_yaxes(title_text="Pressure Raw", secondary_y=True,
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
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8b949e; margin-top:4px;">EXACT 30S BASELINE VS 30S EXPERIMENT</div>
</div>
""", unsafe_allow_html=True)
    st.caption(
        "Calculates stats for **ALL loaded subjects** across your currently active phase filter. Automatically detects the exact millisecond the intervention triggers and strictly extracts the 30 seconds before and 30 seconds after.")

    # We pull files from df_all (which respects Phase Filter, but includes ALL pigs and ALL batches)
    stats_input_df = df_all[['source_path', 'subject_id', 'condition']].drop_duplicates()

    if not stats_input_df.empty:
        with st.spinner("Calculating rigorous Pre/Post statistics across all subjects..."):
            stats_df = extract_pre_post_stats(stats_input_df, selected_sensor_labels, available_map)

        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)

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
