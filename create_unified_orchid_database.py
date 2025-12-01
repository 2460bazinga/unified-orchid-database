#!/usr/bin/env python3
"""
Unified ORCHID Database Builder - PROPERLY OPTIMIZED
=====================================================

Uses exact feature extraction logic from original but with pre-grouped events.
This avoids filtering millions of rows for each patient.

Speed improvement: ~10-15x faster feature extraction (90min → 6-10min)

Author: Noah Parrish (noah@2460.life)
Date: December 1, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('/home/noah/physionet.org/files/orchid/2.1.1')
OUTPUT_DIR = Path('/home/noah/orchid_enhanced_suite/unified_database')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

ORGANS = [
    'kidney_left', 'kidney_right', 'liver', 'heart',
    'lung_left', 'lung_right', 'pancreas', 'intestine'
]

CHEM_MAPPINGS = {
    'Creatinine': 'creatinine', 'BUN': 'bun', 'CreatinineClearance': 'creatinine_clearance',
    'TotalBili': 'bilirubin_total', 'DirectBili': 'bilirubin_direct', 'IndirectBili': 'bilirubin_indirect',
    'SGOTAST': 'ast', 'SGPTALT': 'alt', 'AlkPhos': 'alp',
    'Albumin': 'albumin', 'TotalProtein': 'total_protein',
    'INR': 'inr', 'PT': 'pt', 'PTT': 'ptt',
    'Glucose': 'glucose', 'HgbA1C': 'hba1c',
    'Lactate': 'lactate', 'Amylase': 'amylase', 'Lipase': 'lipase',
    'TroponinI': 'troponin', 'CKMB': 'ck_mb', 'BNP': 'bnp',
    'Sodium': 'sodium', 'K': 'potassium', 'CI': 'chloride', 'CO2': 'bicarbonate',
    'Calcium': 'calcium', 'Mg': 'magnesium', 'Phosphorous': 'phosphorus'
}

ABG_MAPPINGS = {
    'PH': 'ph', 'PCO2': 'paco2', 'PO2': 'pao2',
    'HCO3': 'hco3', 'BE': 'base_excess', 'O2SAT': 'spo2',
    'FIO2': 'fio2', 'PEEP': 'peep', 'Rate': 'respiratory_rate',
    'TV': 'tidal_volume', 'PIP': 'peak_pressure'
}

HEMO_MAPPINGS = {
    'BPSystolic': 'sbp', 'BPDiastolic': 'dbp',
    'HeartRate': 'hr', 'Temperature': 'temperature',
    'UrineOutput': 'urine_output'
}

CBC_MAPPINGS = {
    'WBC': 'wbc', 'Hgb': 'hemoglobin', 'Hct': 'hematocrit',
    'RBC': 'rbc', 'Ptl': 'platelet',
    'Lymp': 'lymphocyte', 'Mono': 'monocyte',
    'Eos': 'eosinophil', 'Segs': 'neutrophil', 'Band': 'band'
}

OPO_NAME_MAPPINGS = {
    'OPO1': {'name': 'Southwest Transplant Alliance', 'code': 'TXSB'},
    'OPO2': {'name': 'Louisiana Organ Procurement Agency', 'code': 'LAOP'},
    'OPO3': {'name': 'Life Connection of Ohio', 'code': 'OHOV'},
    'OPO4': {'name': 'Donor Network West', 'code': 'CADN'},
    'OPO5': {'name': 'Mid-America Transplant', 'code': 'MOSL'},
    'OPO6': {'name': 'OurLegacy', 'code': 'FLUF'}
}

OPO_GEOGRAPHIC_FEATURES = {
    "OPO1": {"opo_dsa_area_km2": 154035.57, "opo_centroid_lat": 32.405862, "opo_centroid_lon": -96.206309},
    "OPO2": {"opo_dsa_area_km2": 140153.55, "opo_centroid_lat": 30.059951, "opo_centroid_lon": -90.886263},
    "OPO3": {"opo_dsa_area_km2": 34448.97, "opo_centroid_lat": 41.435935, "opo_centroid_lon": -83.187},
    "OPO4": {"opo_dsa_area_km2": 329883.9, "opo_centroid_lat": 38.063363, "opo_centroid_lon": -122.076913},
    "OPO5": {"opo_dsa_area_km2": 142787.07, "opo_centroid_lat": 37.689319, "opo_centroid_lon": -90.049587},
    "OPO6": {"opo_dsa_area_km2": 25593.71, "opo_centroid_lat": 28.401476, "opo_centroid_lon": -81.05129}
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def compute_temporal_features(values, times, window_hours=24):
    """Compute temporal features."""
    if len(values) == 0:
        return {'last': np.nan, 'mean': np.nan, 'slope': np.nan, 'volatility': np.nan, 'n_measurements': 0}
    
    mask = times <= window_hours
    values_window = values[mask]
    times_window = times[mask]
    
    if len(values_window) == 0:
        return {'last': values[-1], 'mean': np.nan, 'slope': np.nan, 'volatility': np.nan, 'n_measurements': len(values)}
    
    result = {
        'last': values_window[-1],
        'mean': np.mean(values_window),
        'n_measurements': len(values_window)
    }
    
    if len(values_window) >= 3:
        try:
            slope, _, _, _, _ = stats.linregress(times_window, values_window)
            result['slope'] = slope
        except ValueError:
            # All x values are identical (measurements at same time)
            result['slope'] = np.nan
    else:
        result['slope'] = np.nan
    
    if len(values_window) >= 2:
        result['volatility'] = np.std(values_window)
    else:
        result['volatility'] = np.nan
    
    return result


def extract_features_from_grouped(patient_id, grouped_events, approach_time, param_mappings, prefix, param_col='chem_name'):
    """Extract features from pre-grouped events."""
    features = {}
    
    try:
        patient_events = grouped_events.get_group(patient_id).copy()
    except KeyError:
        return features
    
    patient_events['hours_before_approach'] = (
        pd.to_datetime(approach_time) - pd.to_datetime(patient_events['time_event'])
    ).dt.total_seconds() / 3600
    patient_events = patient_events[patient_events['hours_before_approach'] >= 0]
    
    for orchid_name, standard_name in param_mappings.items():
        param_data = patient_events[patient_events[param_col] == orchid_name]
        if len(param_data) > 0:
            temporal = compute_temporal_features(
                param_data['value'].values,
                param_data['hours_before_approach'].values,
                window_hours=24
            )
            for key, value in temporal.items():
                features[f'{prefix}_{standard_name}_{key}'] = value
    
    return features


def extract_hemo_features_grouped(patient_id, grouped_events, approach_time):
    """Extract hemodynamic features from pre-grouped events."""
    features = {}
    
    try:
        patient_events = grouped_events.get_group(patient_id).copy()
    except KeyError:
        return features
    
    patient_events['hours_before_approach'] = (
        pd.to_datetime(approach_time) - pd.to_datetime(patient_events['time_event_start'])
    ).dt.total_seconds() / 3600
    patient_events = patient_events[patient_events['hours_before_approach'] >= 0]
    
    for orchid_name, standard_name in HEMO_MAPPINGS.items():
        if orchid_name == 'UrineOutput':
            meas_data = patient_events[
                (patient_events['measurement_name'] == orchid_name) &
                (patient_events['measurement_type'] == 'Total')
            ]
        else:
            meas_data = patient_events[
                (patient_events['measurement_name'] == orchid_name) &
                (patient_events['measurement_type'] == 'Average')
            ]
        
        if len(meas_data) > 0:
            temporal = compute_temporal_features(
                meas_data['value'].values,
                meas_data['hours_before_approach'].values,
                window_hours=24
            )
            for key, value in temporal.items():
                features[f'hemo_{standard_name}_{key}'] = value
    
    return features


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all ORCHID data."""
    print("\n" + "=" * 80)
    print("LOADING ORCHID DATA")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load referrals
    print("Loading referrals...")
    referrals = pd.read_csv(DATA_DIR / 'OPOReferrals.csv', low_memory=False)
    print(f"  ✓ {len(referrals):,} referrals")
    
    referrals['time_referred'] = pd.to_datetime(referrals['time_referred'], errors='coerce')
    referrals['time_approached'] = pd.to_datetime(referrals['time_approached'], errors='coerce')
    referrals['donation_type'] = referrals['brain_death'].map({True: 'DBD', False: 'DCD'})
    
    # Parse transplant outcomes
    print("Parsing transplant outcomes...")
    for organ in ORGANS:
        outcome_col = f'outcome_{organ}'
        if outcome_col in referrals.columns:
            referrals[f'{outcome_col}_procured'] = referrals[outcome_col].notna().astype(int)
            referrals[f'{outcome_col}_transplanted'] = (referrals[outcome_col] == 'Transplanted').astype(int)
            referrals[f'{outcome_col}_placement_failure'] = (
                referrals[outcome_col] == 'Recovered for Transplant but not Transplanted'
            ).astype(int)
            referrals[f'{outcome_col}_research'] = (referrals[outcome_col] == 'Recovered for Research').astype(int)
    
    referrals['num_organs_procured'] = sum(referrals[f'outcome_{organ}_procured'] for organ in ORGANS)
    referrals['num_organs_transplanted'] = sum(referrals[f'outcome_{organ}_transplanted'] for organ in ORGANS)
    referrals['multi_organ_donor'] = (referrals['num_organs_procured'] > 1).astype(int)
    
    # Load and group event tables
    print("Loading and grouping event tables...")
    grouped_events = {}
    
    for table_name, key in [('ChemistryEvents', 'chemistry'), ('ABGEvents', 'abg'),
                             ('HemoEvents', 'hemo'), ('CBCEvents', 'cbc')]:
        try:
            df = pd.read_csv(DATA_DIR / f'{table_name}.csv', low_memory=False)
            print(f"  ✓ {table_name}: {len(df):,} events")
            print(f"    Grouping by patient_id...")
            grouped_events[key] = df.groupby('patient_id')
            print(f"    ✓ Grouped")
        except FileNotFoundError:
            grouped_events[key] = None
            print(f"  ⚠ {table_name}: not found")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Data loaded in {elapsed:.1f} seconds")
    
    return {'referrals': referrals, 'grouped_events': grouped_events}


# ============================================================================
# FEATURE EXTRACTION (OPTIMIZED)
# ============================================================================

def extract_all_features(referrals_df, grouped_events):
    """Extract features using pre-grouped events."""
    print("\n" + "=" * 80)
    print("EXTRACTING CLINICAL FEATURES (OPTIMIZED)")
    print("=" * 80)
    
    start_time = time.time()
    all_features = []
    total = len(referrals_df)
    
    for i, (_, patient) in enumerate(referrals_df.iterrows()):
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i) / speed if speed > 0 else 0
            print(f"\r[build_unified_orchid_database] Progress: {i+1:,}/{total:,} ({(i+1)/total*100:.1f}%) | "
                  f"Speed: {speed:.1f} patients/sec | ETA: {eta/60:.1f}m | Elapsed: {elapsed/60:.1f}m", end='')
        
        patient_id = patient['patient_id']
        approach_time = patient.get('time_approached', patient.get('time_referred'))
        
        features = {'patient_id': patient_id}
        
        # Extract features from each event type (using pre-grouped data)
        if grouped_events['chemistry'] is not None:
            features.update(extract_features_from_grouped(
                patient_id, grouped_events['chemistry'], approach_time, CHEM_MAPPINGS, 'chem', 'chem_name'
            ))
        if grouped_events['abg'] is not None:
            features.update(extract_features_from_grouped(
                patient_id, grouped_events['abg'], approach_time, ABG_MAPPINGS, 'abg', 'abg_name'
            ))
        if grouped_events['hemo'] is not None:
            features.update(extract_hemo_features_grouped(patient_id, grouped_events['hemo'], approach_time))
        if grouped_events['cbc'] is not None:
            features.update(extract_features_from_grouped(
                patient_id, grouped_events['cbc'], approach_time, CBC_MAPPINGS, 'cbc', 'cbc_name'
            ))
        
        all_features.append(features)
    
    elapsed = time.time() - start_time
    print(f"\n\n✓ Feature extraction complete!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    
    return pd.DataFrame(all_features).set_index('patient_id')


# ============================================================================
# ORGAN-LEVEL EXPLOSION (VECTORIZED)
# ============================================================================

def create_organ_records(referrals_df, all_features_df):
    """Create organ-level records."""
    print("\n" + "=" * 80)
    print("CREATING ORGAN-LEVEL RECORDS")
    print("=" * 80)
    
    start_time = time.time()
    
    # Explode to organs
    referrals_df['key'] = 1
    organs_df_temp = pd.DataFrame({'organ_type': ORGANS, 'key': 1})
    organs_df = referrals_df.merge(organs_df_temp, on='key').drop('key', axis=1)
    print(f"  ✓ Created {len(organs_df):,} organ-donor pairs")
    
    # Add organ-specific outcomes
    for organ in ORGANS:
        mask = organs_df['organ_type'] == organ
        organs_df.loc[mask, 'outcome_procured'] = organs_df.loc[mask, f'outcome_{organ}_procured']
        organs_df.loc[mask, 'transplanted'] = organs_df.loc[mask, f'outcome_{organ}_transplanted']
        organs_df.loc[mask, 'procured_but_not_transplanted'] = organs_df.loc[mask, f'outcome_{organ}_placement_failure']
        organs_df.loc[mask, 'recovered_for_research'] = organs_df.loc[mask, f'outcome_{organ}_research']
    
    organs_df['placement_outcome'] = 'Not Procured'
    organs_df.loc[organs_df['transplanted'] == 1, 'placement_outcome'] = 'Transplanted'
    organs_df.loc[organs_df['procured_but_not_transplanted'] == 1, 'placement_outcome'] = 'Placement Failure'
    organs_df.loc[organs_df['recovered_for_research'] == 1, 'placement_outcome'] = 'Research'
    
    organs_df['placement_success'] = np.nan
    procured_mask = organs_df['outcome_procured'] == 1
    organs_df.loc[procured_mask, 'placement_success'] = organs_df.loc[procured_mask, 'transplanted']
    
    # Merge temporal features
    organs_df = organs_df.merge(all_features_df, left_on='patient_id', right_index=True, how='left')
    
    # Add OPO info
    organs_df['opo_name'] = organs_df['opo'].map(lambda x: OPO_NAME_MAPPINGS.get(x, {}).get('name', x) if pd.notna(x) else np.nan)
    organs_df['opo_optn_code'] = organs_df['opo'].map(lambda x: OPO_NAME_MAPPINGS.get(x, {}).get('code', x) if pd.notna(x) else np.nan)
    
    for geo_col in ['opo_dsa_area_km2', 'opo_centroid_lat', 'opo_centroid_lon']:
        organs_df[geo_col] = organs_df['opo'].map(
            lambda x: OPO_GEOGRAPHIC_FEATURES.get(x, {}).get(geo_col, np.nan) if pd.notna(x) else np.nan
        )
    
    organs_df['organ_id'] = organs_df['patient_id'].astype(str) + '_' + organs_df['organ_type']
    
    # Clean up
    cols_to_drop = [col for col in organs_df.columns if col.startswith('outcome_') and '_' in col and col not in ['outcome_procured']]
    organs_df = organs_df.drop(columns=cols_to_drop, errors='ignore')
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed/60:.1f} minutes")
    
    return organs_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main pipeline."""
    print("\n" + "=" * 80)
    print("UNIFIED ORCHID DATABASE BUILDER - PROPERLY OPTIMIZED")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    
    data = load_data()
    referrals_df = data['referrals']
    grouped_events = data['grouped_events']
    
    all_features_df = extract_all_features(referrals_df, grouped_events)
    print(f"Features extracted: {len(all_features_df.columns)} columns")
    
    organs_df = create_organ_records(referrals_df, all_features_df)
    
    print("\n" + "=" * 80)
    print("SAVING DATABASE")
    print("=" * 80)
    
    output_path = OUTPUT_DIR / 'organs_database_with_transplant.parquet'
    organs_df.to_parquet(output_path, index=False, compression='snappy')
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Database: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Records: {len(organs_df):,}")
    print(f"  Columns: {len(organs_df.columns)}")
    
    print("\n" + "=" * 80)
    print("TRANSPLANTATION OUTCOME SUMMARY")
    print("=" * 80)
    
    total_procured = organs_df['outcome_procured'].sum()
    total_transplanted = organs_df['transplanted'].sum()
    
    print(f"Total procured:     {total_procured:,}")
    print(f"Total transplanted: {total_transplanted:,} ({total_transplanted/total_procured*100:.1f}%)")
    
    print("\nBy organ type:")
    for organ in ORGANS:
        organ_data = organs_df[organs_df['organ_type'] == organ]
        procured = organ_data['outcome_procured'].sum()
        transplanted = organ_data['transplanted'].sum()
        if procured > 0:
            print(f"  {organ:15s}: {transplanted:5,} / {procured:5,} = {transplanted/procured*100:5.1f}%")
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
