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
st.set_page_config(layout="wide", page_title="Experiment Dashboard")

# --- Helper Functions ---

def load_and_process_data_from_dir(directory):
    """
    Reads files, processes 30s bins, and keeps track of file paths for drill-down.
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
                    'timestamp': timestamp_str
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

        # 30-second bins
        if 'elapsed [sec]' not in df.columns:
            continue
            
        df['time_bin'] = (df['elapsed [sec]'] // 30).astype(int)

        # Aggregation
        sensors = ['hx711-C [mmHg]', 'hx711-D [mmHg]', 'hx711-E [mmHg]', 'hx711-F [mmHg]', 'hx711-G [mmHg]']
        available_sensors = [s for s in sensors if s in df.columns]
        agg_rules = {s: ['mean', 'std'] for s in available_sensors}
        
        if 'applied pressure [mmHg]' in df.columns:
            agg_rules['applied pressure [mmHg]'] = ['max', 'mean']
        
        agg_rules['elapsed [sec]'] = 'mean'
        agg_rules['is_experiment'] = lambda x: x.mode()[0] if not x.mode().empty else False

        # *** FIX HERE: Add .reset_index() to keep time_bin as a column ***
        df_agg = df.groupby('time_bin').agg(agg_rules).reset_index()

        # Flatten columns
        df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]

        rename_map = {
            'time_bin_': 'time_bin',  # Handle potential suffix from flattening
            'applied pressure [mmHg]_max': 'pressure_max',
            'applied pressure [mmHg]_mean': 'pressure_mean',
            'elapsed [sec]_mean': 'local_time',
            'is_experiment_<lambda>': 'is_experiment'
        }
        df_agg = df_agg.rename(columns=rename_map)

        # Stitch Global Time
        df_agg['global_time'] = df_agg['local_time'] + cumulative_time_offset
        df_agg['condition'] = condition_str
        df_agg['original_file'] = os.path.basename(file_path)
        df_agg['source_path'] = file_path 

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

def get_raw_data_slice(file_path, time_bin):
    """
    Reloads a specific file and extracts rows for a specific 30s bin.
    """
    try:
        df = pd.read_csv(file_path, skiprows=4)
        df['bin'] = (df['elapsed [sec]'] // 30).astype(int)
        # Filter for the specific bin
        subset = df[df['bin'] == time_bin].copy()
        return subset
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

# --- Main App ---

st.title("🧪 Experiment Interactive Dashboard")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_zip = st.file_uploader("Upload ZIP containing CSVs", type="zip")
    
    st.markdown("---")
    with st.expander("ℹ️ Help"):
        st.write("Upload a ZIP file. The dashboard will visualize aggregated data. **Click on any point** in the main graph to see the raw data for that 30-second period below.")

if uploaded_zip is None:
    st.info("👋 Please upload a ZIP file containing your experiment CSV files to begin.")
    st.stop()

# Use a temporary directory to extract files
with tempfile.TemporaryDirectory() as temp_dir:
    
    # Extract ZIP
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        st.error(f"Error extracting ZIP file: {e}")
        st.stop()

    # Load Data
    df_all, files_info = load_and_process_data_from_dir(temp_dir)

    if df_all.empty:
        st.error("No valid CSV files found.")
        st.stop()

    # --- Sidebar: Controls ---
    with st.sidebar:
        st.header("2. Settings")

        # Batch Selection
        batch_option = st.radio("Select Data Batch:", ["Full Experiment", "Batch 1 (First 7 Files)", "Batch 2 (Last 7 Files)"])
        
        split_idx = 7
        file_names = [os.path.basename(f['path']) for f in files_info]

        if batch_option == "Batch 1 (First 7 Files)":
            target_files = file_names[:split_idx]
            df = df_all[df_all['original_file'].isin(target_files)].copy()
        elif batch_option == "Batch 2 (Last 7 Files)":
            target_files = file_names[split_idx:]
            df = df_all[df_all['original_file'].isin(target_files)].copy()
        else:
            df = df_all.copy()
            
        # Reset index to ensure selection indices match
        df = df.reset_index(drop=True)

        st.markdown("---")

        # Sensor Selection
        sensor_map = {
            'Sensor C': 'hx711-C [mmHg]', 'Sensor D': 'hx711-D [mmHg]',
            'Sensor E': 'hx711-E [mmHg]', 'Sensor F': 'hx711-F [mmHg]',
            'Sensor G': 'hx711-G [mmHg]'
        }
        available_map = {k: v for k, v in sensor_map.items() if f"{v}_mean" in df.columns}
        
        if not available_map:
            st.error("No expected sensors found.")
            st.stop()
            
        selected_sensor_label = st.selectbox("Select Sensor:", list(available_map.keys()))
        selected_sensor_col = available_map[selected_sensor_label]

        st.info(f"Displaying **{len(df)}** intervals.")

    # --- Main Graph ---
    
    df['group_id'] = ((df['condition'] != df['condition'].shift()) | (df['is_experiment'] != df['is_experiment'].shift())).cumsum()
    grouped = df.groupby('group_id')
    group_info = []
    
    for _, group in grouped:
        group_info.append({
            'start': group['global_time'].min() - 15, 'end': group['global_time'].max() + 15,
            'is_exp': group['is_experiment'].iloc[0], 'condition': group['condition'].iloc[0]
        })

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    mean_col = f"{selected_sensor_col}_mean"
    std_col = f"{selected_sensor_col}_std"

    # Bounds
    fig.add_trace(go.Scatter(x=df['global_time'], y=df[mean_col] + df[std_col], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['global_time'], y=df[mean_col] - df[std_col], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)', name='Std Dev', hoverinfo='skip'), secondary_y=False)

    # Main Sensor Line
    fig.add_trace(go.Scatter(
        x=df['global_time'], y=df[mean_col],
        mode='lines+markers', line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, color='#1f77b4'),
        name=f'{selected_sensor_label} Mean'
    ), secondary_y=False)

    # Pressure
    if 'pressure_max' in df.columns:
        fig.add_trace(go.Scatter(x=df['global_time'], y=df['pressure_max'], mode='lines+markers', line=dict(color='#d62728', width=2, dash='dot'), marker=dict(size=4, symbol='diamond'), name='Max Pressure'), secondary_y=True)

    # Backgrounds
    for info in group_info:
        color = "rgba(0, 128, 0, 0.1)" if info['is_exp'] else "rgba(128, 128, 128, 0.1)"
        fig.add_vrect(x0=info['start'], x1=info['end'], fillcolor=color, layer="below", line_width=0)
        fig.add_annotation(x=(info['start']+info['end'])/2, y=1.05, yref="y domain", text=f"{info['condition']}", showarrow=False, font=dict(size=10))

    fig.update_layout(
        title="<b>Overview:</b> 30s Averages (Click a point to see raw data below)",
        height=500, hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=1)
    )
    fig.update_yaxes(title_text="Sensor (mmHg)", secondary_y=False)
    fig.update_yaxes(title_text="Pressure (mmHg)", secondary_y=True)

    # --- Interactive Chart with Selection ---
    # on_select="rerun" ensures the app re-runs when a point is clicked
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    # --- Drill Down Logic ---
    st.divider()
    
    # Check if a point was selected
    if event and event['selection']['points']:
        try:
            # Get the index of the selected point
            point_index = event['selection']['points'][0]['point_index']
            
            # Retrieve metadata for that point
            selected_row = df.iloc[point_index]
            
            # Ensure time_bin exists before accessing
            if 'time_bin' not in selected_row:
                 st.error("Error: 'time_bin' data missing. Please check file format.")
                 st.stop()
                 
            target_bin = int(selected_row['time_bin'])
            target_file = selected_row['source_path']
            target_condition = selected_row['condition']
            
            st.subheader(f"🔍 Detail View: {target_condition} (Bin #{target_bin})")
            st.caption(f"Source File: {os.path.basename(target_file)} | Time Bin: {target_bin} (approx {target_bin*30}-{(target_bin+1)*30} sec)")
            
            # Load Raw Data for that slice
            raw_slice = get_raw_data_slice(target_file, target_bin)
            
            if not raw_slice.empty:
                fig_detail = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Raw Sensor Data
                raw_sensor_col = selected_sensor_col.replace("_mean", "") 
                if raw_sensor_col in raw_slice.columns:
                     fig_detail.add_trace(go.Scatter(
                        x=raw_slice['elapsed [sec]'], y=raw_slice[raw_sensor_col],
                        mode='lines', line=dict(color='#1f77b4', width=2),
                        name=f'Raw {selected_sensor_label}'
                    ), secondary_y=False)
                
                # Raw Pressure Data
                if 'applied pressure [mmHg]' in raw_slice.columns:
                     fig_detail.add_trace(go.Scatter(
                        x=raw_slice['elapsed [sec]'], y=raw_slice['applied pressure [mmHg]'],
                        mode='lines', line=dict(color='#d62728', width=2),
                        name='Raw Applied Pressure'
                    ), secondary_y=True)
                
                fig_detail.update_layout(height=400, hovermode="x unified", showlegend=True)
                fig_detail.update_xaxes(title_text="Time within File (sec)")
                fig_detail.update_yaxes(title_text="Sensor Raw", secondary_y=False, title_font=dict(color="#1f77b4"))
                fig_detail.update_yaxes(title_text="Pressure Raw", secondary_y=True, title_font=dict(color="#d62728"))
                
                st.plotly_chart(fig_detail, use_container_width=True)
                
            else:
                st.warning("Could not retrieve raw data for this timestamp.")
                
        except Exception as e:
            st.error(f"Error displaying detail view: {e}")
            
    else:
        st.info("👆 **Click on a blue dot (Sensor Mean)** in the graph above to see the raw second-by-second data for that specific interval here.")

    # --- Downloads (Sidebar) ---
    with st.sidebar:
        st.header("3. Downloads")
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Aggregated Data", csv_buffer, "processed_data.csv", "text/csv")
