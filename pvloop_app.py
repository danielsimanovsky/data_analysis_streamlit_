import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PV Loop Hemodynamics Dashboard", layout="wide")
st.title("Hemodynamic PV Loop Analysis Dashboard")
st.markdown("""
*Use this exploratory tool to visualize intra-individual responses (Before vs. After) and pooled inter-individual changes ($\Delta$).*
""")


# --- DATA PROCESSING (CACHED FOR PERFORMANCE) ---
@st.cache_data
def load_and_process_data(uploaded_files):
    """Reads uploaded XLSX files, calculates means, and pairs Before/After data."""
    aggregated_data = []

    for file in uploaded_files:
        filename = file.name
        if not filename.endswith(".xlsx") or filename.startswith("~$"):
            continue

        swine_id = "Swine 1" if "per min" in filename.lower() else "Swine 2"

        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                if df_raw.empty: continue

                # Find header
                header_idx = 0
                for idx, row in df_raw.head(20).iterrows():
                    if 'SW' in ' '.join([str(x) for x in row.values]):
                        header_idx = idx
                        break

                df = df_raw.iloc[header_idx + 1:].copy()
                df.columns = df_raw.iloc[header_idx].values

                # Clean columns
                df = df.loc[:, df.columns.notna()]
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.dropna(axis=1, how='all')

                # Parse metadata
                intervention = "Inspirium" if "insp" in sheet_name.lower() else (
                    "Experium" if "exp" in sheet_name.lower() else "Baseline")
                level_match = re.search(r'(20|40|60|80)', sheet_name)
                level = int(level_match.group(1)) if level_match else 0

                trial = 1
                if re.search(r'(2nd|\(2\))', sheet_name.lower()):
                    trial = 2
                elif re.search(r'(3rd|\(3\))', sheet_name.lower()):
                    trial = 3
                elif re.search(r'(4th|\(4\))', sheet_name.lower()):
                    trial = 4

                # FIX: Renamed State to universally use 'Before' and 'After' to prevent confusing Intervention names with baseline states
                state = "Before" if any(x in sheet_name.lower() for x in ["off", "before", "baseline"]) else "After"

                # Strip units and average
                clean_cols = {col: re.sub(r'\s*\([^)]*\)', '', str(col)).strip() for col in df.columns if pd.notna(col)}

                # FIX: Rename columns first, then drop duplicates using internal alignment to prevent unalignable boolean error
                df = df.rename(columns=clean_cols)
                df = df.loc[:, ~df.columns.duplicated()]

                for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0: continue

                row_data = {
                    'Swine_ID': swine_id, 'Intervention': intervention,
                    'Level': level, 'Trial': trial, 'State': state
                }
                row_data.update(df[numeric_cols].mean().to_dict())
                aggregated_data.append(row_data)
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    master_df = pd.DataFrame(aggregated_data)
    if master_df.empty: return master_df, pd.DataFrame()

    # Calculate Deltas
    baseline = master_df[master_df['State'] == 'Before'].drop(columns=['State'])
    intervention = master_df[master_df['State'] == 'After'].drop(columns=['State'])
    merge_cols = ['Swine_ID', 'Intervention', 'Level', 'Trial']
    merged = pd.merge(baseline, intervention, on=merge_cols, suffixes=('_Before', '_After'))

    delta_df = merged[merge_cols].copy()
    num_vars = [c for c in master_df.columns if c not in merge_cols + ['State']]
    for col in num_vars:
        if f'{col}_Before' in merged.columns and f'{col}_After' in merged.columns:
            delta_df[col] = merged[f'{col}_After'] - merged[f'{col}_Before']

    return master_df, delta_df


# --- SIDEBAR CONTROLS & FILE UPLOAD ---
st.sidebar.header("📁 Data Input")
uploaded_files = st.sidebar.file_uploader("Upload PV Loop Excel Files (.xlsx)", type=['xlsx'],
                                          accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload your .xlsx files using the sidebar to begin the analysis.")
    st.stop()

# Load Data
with st.spinner("Parsing Excel files..."):
    master_df, delta_df = load_and_process_data(uploaded_files)

if master_df.empty:
    st.error("No valid data found in the uploaded files. Please check the format.")
    st.stop()

st.sidebar.header("⚙️ Dashboard Filters")

# Apply filters
selected_interventions = st.sidebar.multiselect("Filter Interventions", ['Inspirium', 'Experium'],
                                                default=['Inspirium', 'Experium'])
selected_pigs = st.sidebar.multiselect("Filter Swine", ['Swine 1', 'Swine 2'], default=['Swine 1', 'Swine 2'])
# NEW FEATURE: Filter specific levels to isolate a specific experiment
available_levels = sorted(master_df['Level'].unique())
selected_levels = st.sidebar.multiselect("Filter Levels", available_levels, default=available_levels)

# --- MAIN UI ---
st.divider()

# Dynamically get all numerical variables and place the selector prominently in the main UI
exclude_cols = ['Swine_ID', 'Intervention', 'Level', 'Trial', 'State']
available_vars = sorted([col for col in master_df.columns if col not in exclude_cols])

st.markdown("### 🎯 Primary Analysis Target")
selected_var = st.selectbox(
    "Choose which hemodynamic variable to visualize across the dashboard:",
    available_vars,
    index=available_vars.index('SW') if 'SW' in available_vars else 0
)

# Apply filters dynamically based on sidebar
filtered_master = master_df[(master_df['Intervention'].isin(selected_interventions)) &
                            (master_df['Swine_ID'].isin(selected_pigs)) &
                            (master_df['Level'].isin(selected_levels))]

filtered_delta = delta_df[(delta_df['Intervention'].isin(selected_interventions)) &
                          (delta_df['Swine_ID'].isin(selected_pigs)) &
                          (delta_df['Level'].isin(selected_levels))]

# --- UI TABS ---
tab1, tab2, tab3 = st.tabs(["🧬 Per Pig (Before vs. After)", "📊 Pooled Experiment (Delta Change)", "🗄️ Raw Data Tables"])

with tab1:
    st.subheader(f"Intra-Individual Response: {selected_var}")
    st.write(
        "Displays the absolute values. **Note:** Swine 1 and Swine 2 are plotted on separate axes because they represent uncalibrated (Volts) vs. calibrated (mmHg) data.")

    if not filtered_master.empty:

        # 1. UPDATED FEATURE: Grouped Bar Chart with Toggle for Individual Runs
        st.markdown("#### Bar Chart (Before vs. After)")
        bar_mode = st.radio("Display Mode:",
                            ["Average across trials (Mean ± SEM)", "Show multiple runs (Individual Trials)"],
                            horizontal=True)

        if bar_mode == "Average across trials (Mean ± SEM)":
            # Calculate mean and standard error for the bars
            agg_df = filtered_master.groupby(['Swine_ID', 'Intervention', 'Level', 'State'])[selected_var].agg(
                mean='mean', sem='sem').reset_index()

            fig_bar = px.bar(
                agg_df, x="Level", y="mean", color="State", facet_col="Intervention", facet_row="Swine_ID",
                error_y="sem",
                barmode="group",
                category_orders={"State": ["Before", "After"], "Level": [20, 40, 60, 80]},
                labels={"mean": f"Mean {selected_var}"},
                title=f"Comparison of {selected_var} Before & After (Averaged across trials)"
            )
        else:
            # Show individual trials side-by-side instead of averaging
            filtered_master_plot = filtered_master.copy()
            filtered_master_plot['Run'] = "Run " + filtered_master_plot['Trial'].astype(str)

            fig_bar = px.bar(
                filtered_master_plot.sort_values(['Level', 'Trial']),
                x="Run", y=selected_var, color="State", facet_col="Level", facet_row="Swine_ID",
                barmode="group",
                category_orders={"State": ["Before", "After"], "Level": [20, 40, 60, 80],
                                 "Run": ["Run 1", "Run 2", "Run 3", "Run 4"]},
                labels={selected_var: f"Absolute {selected_var}"},
                title=f"Comparison of {selected_var} Before & After (Showing Individual Runs per Level)"
            )

        fig_bar.update_layout(height=450 if len(selected_pigs) == 1 else 700)
        fig_bar.update_yaxes(matches=None)  # Allow V vs mmHg difference
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # 2. Box plot showing distribution
        st.markdown("#### Distribution Overview (Boxplot)")
        fig_box = px.box(
            filtered_master,
            x="Level", y=selected_var, color="State", facet_col="Intervention", facet_row="Swine_ID",
            points="all",  # Shows all individual trials
            category_orders={"State": ["Before", "After"], "Level": [20, 40, 60, 80]},
            title=f"Distribution of {selected_var} Before vs. After per Swine",
            hover_data=["Trial", "Intervention"]
        )
        fig_box.update_layout(boxmode='group', height=500 if len(selected_pigs) == 1 else 700)
        fig_box.update_yaxes(matches=None)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("No data matches the selected filters.")

with tab2:
    st.subheader(f"Pooled Intervention Efficacy: $\Delta$ {selected_var}")
    st.write(
        "Displays the absolute change ($\Delta$ = After - Before) per trial. Because this normalizes the starting baseline, pooled visualization is statistically viable.")

    if not filtered_delta.empty and selected_var in filtered_delta.columns:
        fig_delta = px.box(
            filtered_delta,
            x="Level", y=selected_var, color="Intervention", facet_col="Intervention",
            points="all",  # Shows all individual trials
            category_orders={"Level": [20, 40, 60, 80]},
            title=f"Absolute Change ($\Delta$) in {selected_var} Across Levels",
            hover_data=["Trial", "Swine_ID"]
        )
        # Add a zero-line to clearly indicate no change
        fig_delta.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="No Change")
        fig_delta.update_layout(boxmode='group', height=600)
        st.plotly_chart(fig_delta, use_container_width=True)
    else:
        st.warning("No paired Delta data available for these filters.")

with tab3:
    st.subheader("Tabular Data Export")
    st.write("Use these tables to export data directly to statistical software (e.g., SPSS, GraphPad PRISM).")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Raw Absolute Means**")
        st.dataframe(filtered_master)
    with col2:
        st.write("**Paired $\Delta$ Changes**")
        st.dataframe(filtered_delta)
