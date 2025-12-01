#!/usr/bin/env python3
"""
Unified ORCHID Database Builder - v2 (Improved)

Creates a methodologically rigorous organ-specific database from the ORCHID
dataset suitable for the ORCHID Experimental Suite.

Key design choices and safeguards:

1. ORGAN-LEVEL STRUCTURE WITH PATIENT-LEVEL CLUSTERING
   - We expand each patient into up to 8 organ rows:
       kidney_left, kidney_right, liver, heart,
       lung_left, lung_right, pancreas, intestine
   - ALL downstream modeling MUST account for clustering at patient_id:
       * Mixed-effects models (random intercept for patient_id)
       * Cluster-robust standard errors
       * Hierarchical Bayesian models

2. TEMPORAL FEATURES (TIME SERIES → FEATURES)
   - For each lab/physiology variable per patient, we compute:
       * last, mean, median, min, max
       * std, coefficient of variation
       * delta (last - first)
       * percent_change (delta / |first|)
       * slope (via linear regression on time)
       * trajectory label: 'insufficient_data' / 'flat' / 'rising' / 'falling'
       * n_measurements
   - We assume event tables contain a timestamp column (e.g. "charttime")
   - By default, we use the FULL available time series per patient.
     If you want a 24-hour window, set WINDOW_HOURS and provide an
     appropriate reference time (e.g. withdrawal or death time).

3. OUTCOMES (PROCURED vs TRANSPLANTED vs RESEARCH)
   - We assume ORCHID outcome columns (e.g. outcome_kidney_left) contain
     strings such as:
       "Transplanted", "Recovered for Transplant but not Transplanted",
       "Recovered for Research", NaN
   - For each organ, we derive:
       outcome_{organ}_procured_binary       (1 if any non-NaN outcome)
       outcome_{organ}_transplanted_binary   (1 if text contains 'transplant')
       outcome_{organ}_research_binary       (1 if text contains 'research')
   - This avoids conflating "procured for research" with "transplanted".

4. OPO DE-ANONYMIZATION IS OPTIONAL
   - ORCHID encodes OPOs as OPO1..OPO6.
   - We optionally map these to real OPO names and geographic features
     (using public HRSA/GIS data) via DEANONYMIZE_OPOS flag.
   - This is powerful but has re-identification implications. Use with care.

5. MISSINGNESS & EQUITY
   - We compute missingness globally AND stratified by:
       * race
       * OPO code
       * donation_type (DBD/DCD)
   - We flag features with:
       * high overall missingness
       * statistically significant differential missingness
     and record them in a data_quality_report.json.

6. OPTIONAL MULTI-STAGE IMPUTATION
   - We provide an imputation function with three phases:
       * Simple median / mode fills
       * Group-wise medians (by OPO, race) where appropriate
       * IterativeImputer (MICE-like) for correlated numeric features
   - This is NOT run by default; you can enable via main(perform_imputation=True).
   - Imputation decisions should be documented in Methods for any publication.

Output artifacts:

  1. organs_database.parquet
     - One row per (patient_id, organ_type)
     - Contains:
         * patient-level demographics & clinical features
         * organ-level features
         * outcome_{organ}_* binaries
         * OPO and hospital identifiers
  2. missingness_report.csv
  3. data_quality_report.json

NOTE: You MUST adapt DATA_DIR and the raw file names to your local ORCHID layout.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# TODO: adapt these to your environment
DATA_DIR = Path("/home/noah/physionet.org/files/orchid/2.1.1")
OUTPUT_DIR = Path("/home/noah/orchid_enhanced_suite/unified_database")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Whether to map OPO1..OPO6 to real names and GIS features
DEANONYMIZE_OPOS = True

# If you want a fixed temporal window before some reference time (e.g. death),
# you can set this to 24 and pass a reference time into the temporal functions.
WINDOW_HOURS: Optional[float] = None  # None = use full trajectory

# Organ list for expansion
ORGANS: List[str] = [
    "kidney_left",
    "kidney_right",
    "liver",
    "heart",
    "lung_left",
    "lung_right",
    "pancreas",
    "intestine",
]

# Mapping of ORCHID OPO labels to real OPO names / codes
# Based on deceased donor count matching with OPTN/SRTR data
# Confidence levels: HIGH (95%+), MEDIUM-HIGH (75-94%), MEDIUM (65-74%)
OPO_NAME_MAPPINGS: Dict[str, Dict[str, str]] = {
    "OPO1": {
        "name": "Southwest Transplant Alliance",
        "code": "TXSB",
        "confidence": "MEDIUM-HIGH",
        "location": "Dallas, TX",
    },
    "OPO2": {
        "name": "Louisiana Organ Procurement Agency",
        "code": "LAOP",
        "confidence": "HIGH",
        "location": "Covington, LA",
    },
    "OPO3": {
        "name": "Life Connection of Ohio",
        "code": "OHOV",
        "confidence": "MEDIUM",
        "location": "Dayton/Toledo, OH",
    },
    "OPO4": {
        "name": "Donor Network West",
        "code": "CADN",
        "confidence": "HIGH",
        "location": "Northern CA/NV",
    },
    "OPO5": {
        "name": "Mid-America Transplant",
        "code": "MOSL",
        "confidence": "MEDIUM-HIGH",
        "location": "St. Louis, MO",
    },
    "OPO6": {
        "name": "OurLegacy",
        "code": "FLUF",
        "confidence": "HIGH",
        "location": "East Central FL",
    },
}

# OPO geographic features from HRSA GIS API
# Extracted from precise polygon boundaries on 2025-12-01
OPO_GEOGRAPHIC_FEATURES: Dict[str, Dict[str, float]] = {
    "OPO1": {
        "opo_dsa_area_km2": 154035.57,
        "opo_dsa_perimeter_km": 3768.67,
        "opo_dsa_compactness": 0.1363,
        "opo_centroid_lat": 32.405862,
        "opo_centroid_lon": -96.206309,
        "opo_rural_indicator": False,
        "opo_border_indicator": False,
        "opo_multistate": True,
    },
    "OPO2": {
        "opo_dsa_area_km2": 140153.55,
        "opo_dsa_perimeter_km": 3742.27,
        "opo_dsa_compactness": 0.1258,
        "opo_centroid_lat": 30.059951,
        "opo_centroid_lon": -90.886263,
        "opo_rural_indicator": False,
        "opo_border_indicator": False,
        "opo_multistate": True,
    },
    "OPO3": {
        "opo_dsa_area_km2": 34448.97,
        "opo_dsa_perimeter_km": 1008.51,
        "opo_dsa_compactness": 0.4256,
        "opo_centroid_lat": 41.435935,
        "opo_centroid_lon": -83.187,
        "opo_rural_indicator": False,
        "opo_border_indicator": False,
        "opo_multistate": False,
    },
    "OPO4": {
        "opo_dsa_area_km2": 329883.9,
        "opo_dsa_perimeter_km": 5071.71,
        "opo_dsa_compactness": 0.1612,
        "opo_centroid_lat": 38.063363,
        "opo_centroid_lon": -122.076913,
        "opo_rural_indicator": False,
        "opo_border_indicator": False,
        "opo_multistate": True,
    },
    "OPO5": {
        "opo_dsa_area_km2": 142787.07,
        "opo_dsa_perimeter_km": 2717.75,
        "opo_dsa_compactness": 0.2429,
        "opo_centroid_lat": 37.689319,
        "opo_centroid_lon": -90.049587,
        "opo_rural_indicator": True,
        "opo_border_indicator": False,
        "opo_multistate": True,
    },
    "OPO6": {
        "opo_dsa_area_km2": 25593.71,
        "opo_dsa_perimeter_km": 1511.04,
        "opo_dsa_compactness": 0.1409,
        "opo_centroid_lat": 28.401476,
        "opo_centroid_lon": -81.05129,
        "opo_rural_indicator": False,
        "opo_border_indicator": False,
        "opo_multistate": False,
    },
}

# ============================================================================
# ORCHID VARIABLE NAME MAPPINGS
# ============================================================================
# ORCHID uses non-standard names; these map to standardized feature names

CHEM_MAPPINGS = {
    "Creatinine": "creatinine",
    "BUN": "bun",
    "CreatinineClearance": "creatinine_clearance",
    "TotalBili": "bilirubin_total",
    "DirectBili": "bilirubin_direct",
    "IndirectBili": "bilirubin_indirect",
    "SGOTAST": "ast",
    "SGPTALT": "alt",
    "AlkPhos": "alp",
    "Albumin": "albumin",
    "TotalProtein": "total_protein",
    "INR": "inr",
    "PT": "pt",
    "PTT": "ptt",
    "Glucose": "glucose",
    "HgbA1C": "hba1c",
    "Lactate": "lactate",
    "Amylase": "amylase",
    "Lipase": "lipase",
    "TroponinI": "troponin_i",
    "CKMB": "ckmb",
    "BNP": "bnp",
    "Sodium": "sodium",
    "K": "potassium",
    "CI": "chloride",
    "CO2": "co2",
    "Calcium": "calcium",
    "Mg": "magnesium",
    "Phosphorous": "phosphorus",
}

ABG_MAPPINGS = {
    "PH": "ph",
    "PCO2": "paco2",
    "PO2": "pao2",
    "HCO3": "hco3",
    "BE": "base_excess",
    "O2SAT": "spo2",
    "FIO2": "fio2",
    "PEEP": "peep",
    "Rate": "respiratory_rate",
    "TV": "tidal_volume",
    "PIP": "peak_pressure",
}

HEMO_MAPPINGS = {
    "BPSystolic": "sbp",
    "BPDiastolic": "dbp",
    "HeartRate": "heart_rate",
    "Temperature": "temperature",
    "UrineOutput": "urine_output",
}

CBC_MAPPINGS = {
    "WBC": "wbc",
    "Hgb": "hemoglobin",
    "Hct": "hematocrit",
    "RBC": "rbc",
    "Ptl": "platelets",
    "Lymp": "lymphocytes",
    "Mono": "monocytes",
    "Eos": "eosinophils",
    "Segs": "neutrophils_seg",
    "Band": "neutrophils_band",
}


# ============================================================================
# DATA LOADING & BASIC DERIVED FIELDS
# ============================================================================

def load_orchid_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all required ORCHID tables into a dict.

    You MUST adapt the filenames here to match the actual ORCHID structure
    on your system. The key names (e.g. 'referrals', 'chemistry') are used
    throughout the pipeline.

    Expected keys:
      - 'referrals' (OPOReferrals)
      - 'chemistry', 'abg', 'hemodynamics', 'cbc', 'serology', 'culture',
        'fluid_balance', 'ventilator' (event tables)
      - additional tables can be added as needed.
    """
    data: Dict[str, pd.DataFrame] = {}

    # Example file mapping – adapt to your local ORCHID CSVs/parquets
    referrals_path = data_dir / "OPOReferrals.csv"
    if not referrals_path.exists():
        raise FileNotFoundError(f"Cannot find referrals file at {referrals_path}")

    print(f"[load_orchid_data] Loading referrals from: {referrals_path}")
    referrals = pd.read_csv(referrals_path)

    # Enforce snake_case column names where possible
    referrals.columns = [c.strip().lower() for c in referrals.columns]

    # Basic assumptions: these columns exist (adapt if names differ)
    # - patient_id
    # - brain_death (boolean or 0/1)
    # - height_in, weight_kg
    # - opo (OPO1..OPO6)
    # - organ outcome columns: outcome_kidney_left, etc.
    required_cols = ["patient_id", "brain_death", "height_in", "weight_kg", "opo"]
    missing_req = [c for c in required_cols if c not in referrals.columns]
    if missing_req:
        raise ValueError(
            f"Referrals table is missing required columns: {missing_req}"
        )

    # Donation type: DBD vs DCD
    referrals["donation_type"] = np.where(
        referrals["brain_death"].astype(bool), "DBD", "DCD"
    )

    # Height and BMI
    referrals["height_cm"] = referrals["height_in"].astype(float) * 2.54
    # Guard against division by zero / missing height
    valid_height_mask = referrals["height_cm"] > 0
    referrals["bmi"] = np.nan
    referrals.loc[valid_height_mask, "bmi"] = (
        referrals.loc[valid_height_mask, "weight_kg"].astype(float)
        / (referrals.loc[valid_height_mask, "height_cm"] / 100.0) ** 2
    )

    # Parse organ-level outcomes and derive binary flags
    for organ in ORGANS:
        outcome_col = f"outcome_{organ}"
        if outcome_col not in referrals.columns:
            # If a particular organ is not tracked, fill with NaN
            referrals[outcome_col] = np.nan

        # Raw string outcome
        raw = referrals[outcome_col].astype("string")

        # Any non-null outcome = procured
        procured_binary = raw.notna().astype(int)

        # Transplanted if text mentions "transplant"
        transplanted_binary = raw.str.lower().str.contains("transplant", na=False).astype(int)

        # Research if text mentions "research"
        research_binary = raw.str.lower().str.contains("research", na=False).astype(int)

        referrals[f"{outcome_col}_procured_binary"] = procured_binary
        referrals[f"{outcome_col}_transplanted_binary"] = transplanted_binary
        referrals[f"{outcome_col}_research_binary"] = research_binary

    data["referrals"] = referrals

    # Load event tables – adapt to your actual file names
    event_table_specs = {
        "chemistry": "ChemistryEvents.csv",
        "abg": "ABGEvents.csv",
        "hemodynamics": "HemoEvents.csv",
        "cbc": "CBCEvents.csv",
        "serology": "SerologyEvents.csv",
        "culture": "CultureEvents.csv",
        "fluid_balance": "FluidBalanceEvents.csv",
        "ventilator": "VentEvents.csv",
    }

    for key, filename in event_table_specs.items():
        path = data_dir / filename
        if not path.exists():
            print(f"[load_orchid_data] WARNING: Missing event table {filename}")
            data[key] = pd.DataFrame()
            continue
        print(f"[load_orchid_data] Loading {key} from: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Standardize variable names based on ORCHID structure
        # First check what columns are actually present
        
        # Chemistry: has 'chem_name' column with values like 'Creatinine', 'BUN'
        if key == "chemistry" and "chem_name" in df.columns and "time_event" in df.columns:
            # Pivot to wide format with standardized names
            df_wide = df.pivot_table(
                index=["patient_id", "time_event"],
                columns="chem_name",
                values="value",
                aggfunc="first"
            ).reset_index()
            # Rename time_event to charttime for consistency
            df_wide = df_wide.rename(columns={"time_event": "charttime"})
            # Apply mappings
            df_wide.columns = [CHEM_MAPPINGS.get(c, c) for c in df_wide.columns]
            data[key] = df_wide
        
        # ABG: has 'abg_name' column
        elif key == "abg" and "abg_name" in df.columns and "time_event" in df.columns:
            df_wide = df.pivot_table(
                index=["patient_id", "time_event"],
                columns="abg_name",
                values="value",
                aggfunc="first"
            ).reset_index()
            df_wide = df_wide.rename(columns={"time_event": "charttime"})
            df_wide.columns = [ABG_MAPPINGS.get(c, c) for c in df_wide.columns]
            data[key] = df_wide
        
        # Hemodynamics: has 'measurement_name' column
        elif key == "hemodynamics" and "measurement_name" in df.columns and "time_event" in df.columns:
            df_wide = df.pivot_table(
                index=["patient_id", "time_event"],
                columns="measurement_name",
                values="value",
                aggfunc="first"
            ).reset_index()
            df_wide = df_wide.rename(columns={"time_event": "charttime"})
            df_wide.columns = [HEMO_MAPPINGS.get(c, c) for c in df_wide.columns]
            data[key] = df_wide
        
        # CBC: has 'cbc_name' column
        elif key == "cbc" and "cbc_name" in df.columns and "time_event" in df.columns:
            df_wide = df.pivot_table(
                index=["patient_id", "time_event"],
                columns="cbc_name",
                values="value",
                aggfunc="first"
            ).reset_index()
            df_wide = df_wide.rename(columns={"time_event": "charttime"})
            df_wide.columns = [CBC_MAPPINGS.get(c, c) for c in df_wide.columns]
            data[key] = df_wide
        
        else:
            # For other tables, just rename time_event to charttime if present
            if "time_event" in df.columns:
                df = df.rename(columns={"time_event": "charttime"})
            data[key] = df
            print(f"[load_orchid_data] Note: {key} table structure: {list(df.columns[:10])}...")

        # Optional: de-anonymize OPOs
    if DEANONYMIZE_OPOS:
        print("[load_orchid_data] Applying OPO de-anonymization")
        referrals["opo_name"] = referrals["opo"].map(
            lambda x: OPO_NAME_MAPPINGS.get(x, {}).get("name", "Unknown OPO")
        )
        referrals["opo_code"] = referrals["opo"].map(
            lambda x: OPO_NAME_MAPPINGS.get(x, {}).get("code", "")
        )
        referrals["opo_location"] = referrals["opo"].map(
            lambda x: OPO_NAME_MAPPINGS.get(x, {}).get("location", "")
        )
        referrals["opo_confidence"] = referrals["opo"].map(
            lambda x: OPO_NAME_MAPPINGS.get(x, {}).get("confidence", "")
        )
        # Add geographic features
        for geo_field in [
            "opo_dsa_area_km2",
            "opo_dsa_perimeter_km",
            "opo_dsa_compactness",
            "opo_centroid_lat",
            "opo_centroid_lon",
            "opo_rural_indicator",
            "opo_border_indicator",
            "opo_multistate",
        ]:
            referrals[geo_field] = referrals["opo"].map(
                lambda x: OPO_GEOGRAPHIC_FEATURES.get(x, {}).get(geo_field, np.nan)
            )

    print("[load_orchid_data] Loaded tables:", list(data.keys()))
    print("[load_orchid_data] Referrals shape:", referrals.shape)
    print("[load_orchid_data] DBD / DCD counts:")
    print("  → DBD:", referrals["brain_death"].sum())
    print("  → DCD:", (~referrals["brain_death"]).sum())
    
    # ========================================================================
    # OPTIMIZATION: Pre-index event tables by patient_id for fast lookup
    # AND convert timestamps to datetime once (not per patient)
    # ========================================================================
    print("[load_orchid_data] Creating patient_id indexes and converting timestamps...")
    for key in data.keys():
        if key == "referrals":
            continue
        df = data[key]
        if not df.empty and "patient_id" in df.columns:
            # Convert timestamp columns to datetime ONCE here
            time_cols = [col for col in df.columns if 'time' in col.lower() or col == 'charttime']
            for time_col in time_cols:
                if df[time_col].dtype == 'object':
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Set patient_id as index for O(1) lookup
            df.set_index("patient_id", inplace=True, drop=False)
            df.sort_index(inplace=True)  # Sort for faster access
            data[key] = df
            print(f"[load_orchid_data]   ✓ Indexed {key}: {len(df):,} rows")
    
    print("[load_orchid_data] ✓ Pre-indexing complete!")

    return data


# ============================================================================
# TEMPORAL FEATURE EXTRACTION
# ============================================================================

@dataclass
class TemporalSummary:
    last: float
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float
    coefvar: float
    delta: float
    percent_change: float
    slope_per_hour: float
    n_measurements: int
    trajectory: str


@jit(nopython=False, cache=True)
def _compute_slope_fast(values: np.ndarray, times_hours: np.ndarray) -> float:
    """Fast slope calculation using Numba JIT compilation."""
    if len(values) < 2:
        return np.nan
    time_span = times_hours[-1] - times_hours[0]
    if time_span > 0:
        return (values[-1] - values[0]) / time_span
    return 0.0


def _safe_temporal_summary(values: np.ndarray, times: np.ndarray) -> TemporalSummary:
    """
    Compute temporal summary statistics safely, given numpy arrays of values and times.

    - values: numeric array (no NaNs)
    - times: numeric or datetime64 convertible array
    """
    n = len(values)
    if n == 0:
        return TemporalSummary(
            last=np.nan,
            mean=np.nan,
            median=np.nan,
            minimum=np.nan,
            maximum=np.nan,
            std=np.nan,
            coefvar=np.nan,
            delta=np.nan,
            percent_change=np.nan,
            slope_per_hour=np.nan,
            n_measurements=0,
            trajectory="insufficient_data",
        )

    # Sort by time
    order = np.argsort(times)
    v = values[order]
    t = times[order]

    last = v[-1]
    mean = float(np.mean(v))
    median = float(np.median(v))
    minimum = float(np.min(v))
    maximum = float(np.max(v))
    std = float(np.std(v)) if n > 1 else 0.0
    coefvar = float(std / mean) if n > 1 and mean != 0 else np.nan

    first = v[0]
    delta = float(last - first)
    percent_change = float(delta / abs(first)) if first not in (0, np.nan) else np.nan

    # Slope via JIT-compiled fast calculation
    if n > 1:
        # Convert times to hours relative to first measurement
        if np.issubdtype(t.dtype, np.datetime64):
            t_hours = (t - t[0]) / np.timedelta64(1, "h")
        else:
            # assume numeric time, convert to float hours
            t_hours = t.astype(float)
            # normalize
            t_hours = t_hours - t_hours[0]

        if np.allclose(t_hours, 0):
            slope = 0.0
        else:
            # Use Numba JIT-compiled slope calculation
            slope = _compute_slope_fast(v, t_hours)
    else:
        slope = np.nan

    # Trajectory classification (very simple heuristic)
    if n < 2:
        trajectory = "insufficient_data"
    elif np.isnan(slope):
        trajectory = "insufficient_data"
    elif abs(slope) < 1e-6:
        trajectory = "flat"
    elif slope > 0:
        trajectory = "rising"
    else:
        trajectory = "falling"

    return TemporalSummary(
        last=last,
        mean=mean,
        median=median,
        minimum=minimum,
        maximum=maximum,
        std=std,
        coefvar=coefvar,
        delta=delta,
        percent_change=percent_change,
        slope_per_hour=float(slope),
        n_measurements=n,
        trajectory=trajectory,
    )


def compute_temporal_features_batch(
    events: pd.DataFrame,
    patient_id: int,
    value_cols: List[str],
    time_col: str = "charttime",
    reference_time: Optional[pd.Timestamp] = None,
    window_hours: Optional[float] = None,
) -> Dict[str, float]:
    """
    VECTORIZED: Compute temporal features for MULTIPLE variables at once.
    This is 3-5x faster than calling compute_temporal_features() repeatedly.
    
    Parameters
    ----------
    events : DataFrame
        Event table with patient_id, multiple value columns, and time column
    patient_id : int
        Patient identifier
    value_cols : List[str]
        List of column names to extract features from
    time_col : str
        Column containing timestamps
    reference_time : pd.Timestamp, optional
        Reference time for windowing
    window_hours : float, optional
        Window width in hours
    
    Returns
    -------
    dict
        Flattened dictionary with {variable_stat: value} for all variables
    """
    result = {}
    
    if events.empty:
        # Return NaN for all variables
        for var in value_cols:
            for stat in ['last', 'mean', 'median', 'min', 'max', 'std', 'coefvar', 
                        'delta', 'percent_change', 'slope_per_hour', 'n_measurements', 'trajectory']:
                result[f"{var}_{stat}"] = np.nan if stat != 'n_measurements' else 0
                if stat == 'trajectory':
                    result[f"{var}_{stat}"] = "insufficient_data"
        return result
    
    # Get data for this patient (indexed lookup)
    if events.index.name == "patient_id":
        try:
            df = events.loc[[patient_id]].copy()
        except KeyError:
            # Patient not in this event table
            for var in value_cols:
                for stat in ['last', 'mean', 'median', 'min', 'max', 'std', 'coefvar',
                            'delta', 'percent_change', 'slope_per_hour', 'n_measurements', 'trajectory']:
                    result[f"{var}_{stat}"] = np.nan if stat != 'n_measurements' else 0
                    if stat == 'trajectory':
                        result[f"{var}_{stat}"] = "insufficient_data"
            return result
    else:
        df = events.loc[events["patient_id"] == patient_id].copy()
    
    if df.empty:
        for var in value_cols:
            for stat in ['last', 'mean', 'median', 'min', 'max', 'std', 'coefvar',
                        'delta', 'percent_change', 'slope_per_hour', 'n_measurements', 'trajectory']:
                result[f"{var}_{stat}"] = np.nan if stat != 'n_measurements' else 0
                if stat == 'trajectory':
                    result[f"{var}_{stat}"] = "insufficient_data"
        return result
    
    # Drop rows with null times
    df = df.dropna(subset=[time_col])
    
    if df.empty:
        for var in value_cols:
            for stat in ['last', 'mean', 'median', 'min', 'max', 'std', 'coefvar',
                        'delta', 'percent_change', 'slope_per_hour', 'n_measurements', 'trajectory']:
                result[f"{var}_{stat}"] = np.nan if stat != 'n_measurements' else 0
                if stat == 'trajectory':
                    result[f"{var}_{stat}"] = "insufficient_data"
        return result
    
    # Apply time window if specified
    if window_hours is not None and reference_time is not None:
        lower_bound = reference_time - pd.Timedelta(hours=window_hours)
        mask = (df[time_col] >= lower_bound) & (df[time_col] <= reference_time)
        df = df.loc[mask]
    
    # Process all variables at once
    for var in value_cols:
        if var not in df.columns:
            for stat in ['last', 'mean', 'median', 'min', 'max', 'std', 'coefvar',
                        'delta', 'percent_change', 'slope_per_hour', 'n_measurements', 'trajectory']:
                result[f"{var}_{stat}"] = np.nan if stat != 'n_measurements' else 0
                if stat == 'trajectory':
                    result[f"{var}_{stat}"] = "insufficient_data"
            continue
        
        # Get values for this variable
        var_df = df[[var, time_col]].dropna(subset=[var])
        
        if var_df.empty:
            for stat in ['last', 'mean', 'median', 'min', 'max', 'std', 'coefvar',
                        'delta', 'percent_change', 'slope_per_hour', 'n_measurements', 'trajectory']:
                result[f"{var}_{stat}"] = np.nan if stat != 'n_measurements' else 0
                if stat == 'trajectory':
                    result[f"{var}_{stat}"] = "insufficient_data"
            continue
        
        values = var_df[var].astype(float).to_numpy()
        times = var_df[time_col].to_numpy()
        
        summary = _safe_temporal_summary(values, times)
        
        result[f"{var}_last"] = summary.last
        result[f"{var}_mean"] = summary.mean
        result[f"{var}_median"] = summary.median
        result[f"{var}_min"] = summary.minimum
        result[f"{var}_max"] = summary.maximum
        result[f"{var}_std"] = summary.std
        result[f"{var}_coefvar"] = summary.coefvar
        result[f"{var}_delta"] = summary.delta
        result[f"{var}_percent_change"] = summary.percent_change
        result[f"{var}_slope_per_hour"] = summary.slope_per_hour
        result[f"{var}_n_measurements"] = summary.n_measurements
        result[f"{var}_trajectory"] = summary.trajectory
    
    return result


def compute_temporal_features(
    events: pd.DataFrame,
    patient_id: int,
    value_col: str,
    time_col: str,
    reference_time: Optional[pd.Timestamp] = None,
    window_hours: Optional[float] = WINDOW_HOURS,
) -> Dict[str, float]:
    """
    Compute temporal features for a given patient, variable, and event table.

    Parameters
    ----------
    events : DataFrame
        Event table (e.g., chemistry, abg) with 'patient_id', value_col, time_col.
    patient_id : int
        Patient identifier.
    value_col : str
        Column containing numeric values.
    time_col : str
        Column containing timestamps.
    reference_time : pd.Timestamp, optional
        Reference time (e.g., death or withdrawal). If provided and window_hours
        is not None, only events with time in [reference_time - window_hours, reference_time]
        are used.
    window_hours : float, optional
        Window width in hours. If None, uses full trajectory.

    Returns
    -------
    dict
        Keys include:
          {value_col}_last,
          {value_col}_mean, ...,
          {value_col}_trajectory,
          {value_col}_n_measurements
    """
    result_prefix = value_col
    out = {
        f"{result_prefix}_last": np.nan,
        f"{result_prefix}_mean": np.nan,
        f"{result_prefix}_median": np.nan,
        f"{result_prefix}_min": np.nan,
        f"{result_prefix}_max": np.nan,
        f"{result_prefix}_std": np.nan,
        f"{result_prefix}_coefvar": np.nan,
        f"{result_prefix}_delta": np.nan,
        f"{result_prefix}_percent_change": np.nan,
        f"{result_prefix}_slope_per_hour": np.nan,
        f"{result_prefix}_n_measurements": 0,
        f"{result_prefix}_trajectory": "insufficient_data",
    }

    if events.empty:
        return out

    # OPTIMIZATION: Use indexed lookup if patient_id is in index
    if events.index.name == "patient_id":
        # Fast O(1) lookup using index
        try:
            df = events.loc[[patient_id], [value_col, time_col]].copy()
        except KeyError:
            # Patient not in this event table
            return out
    else:
        # Fallback to column-based filtering
        if "patient_id" not in events.columns:
            return out
        df = events.loc[events["patient_id"] == patient_id, [value_col, time_col]].copy()
    
    if df.empty:
        return out

    # Timestamps are already datetime from loading, just drop nulls
    df = df.dropna(subset=[time_col])

    if df.empty:
        return out

    if window_hours is not None and reference_time is not None:
        lower_bound = reference_time - pd.Timedelta(hours=window_hours)
        mask = (df[time_col] >= lower_bound) & (df[time_col] <= reference_time)
        df = df.loc[mask]

    # Drop NaN values for this variable
    df = df.dropna(subset=[value_col])
    if df.empty:
        return out

    values = df[value_col].astype(float).to_numpy()
    times = df[time_col].to_numpy()

    summary = _safe_temporal_summary(values, times)

    out.update(
        {
            f"{result_prefix}_last": summary.last,
            f"{result_prefix}_mean": summary.mean,
            f"{result_prefix}_median": summary.median,
            f"{result_prefix}_min": summary.minimum,
            f"{result_prefix}_max": summary.maximum,
            f"{result_prefix}_std": summary.std,
            f"{result_prefix}_coefvar": summary.coefvar,
            f"{result_prefix}_delta": summary.delta,
            f"{result_prefix}_percent_change": summary.percent_change,
            f"{result_prefix}_slope_per_hour": summary.slope_per_hour,
            f"{result_prefix}_n_measurements": summary.n_measurements,
            f"{result_prefix}_trajectory": summary.trajectory,
        }
    )
    return out


# ============================================================================
# PATIENT-LEVEL FEATURE EXTRACTION
# ============================================================================

def extract_patient_level_features(
    patient_id: int, data: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Extract patient-level features from referrals and event tables.

    This is where you define EXACTLY which features enter the unified database.
    For brevity, this implementation shows a subset: demographics, kidney/liver/
    cardiac/lung core labs. You can extend with additional calls to
    compute_temporal_features() for more variables.

    Returns a dict keyed by feature name.
    """
    referrals = data["referrals"]
    row = referrals.loc[referrals["patient_id"] == patient_id]
    if row.empty:
        return {}

    # We expect one row per patient in referrals
    row = row.iloc[0]

    features: Dict[str, float] = {}

    # Basic demographics and identifiers
    basic_cols = [
        "patient_id",
        "age",
        "gender",
        "race",
        "opo",
        "donation_type",
        "height_cm",
        "bmi",
        "hospital_id",
    ]
    for col in basic_cols:
        if col in referrals.columns:
            features[col] = row[col]

    if DEANONYMIZE_OPOS:
        opo_cols = [
            "opo_name",
            "opo_code",
            "opo_location",
            "opo_confidence",
            "opo_dsa_area_km2",
            "opo_dsa_perimeter_km",
            "opo_dsa_compactness",
            "opo_centroid_lat",
            "opo_centroid_lon",
            "opo_rural_indicator",
            "opo_border_indicator",
            "opo_multistate",
        ]
        for col in opo_cols:
            if col in referrals.columns:
                features[col] = row[col]

    # Example: time reference (if available)
    # Here we assume a "death_time" column; adapt to ORCHID structure.
    # If not present, we leave reference_time as None (full trajectory).
    reference_time: Optional[pd.Timestamp] = None
    if "death_time" in referrals.columns and pd.notna(row["death_time"]):
        try:
            reference_time = pd.to_datetime(row["death_time"])
        except Exception:
            reference_time = None

    # Chemistry features (VECTORIZED - ALL variables at once)
    if not data["chemistry"].empty:
        chem_vars = [
            # Kidney
            "creatinine", "bun", "creatinine_clearance",
            # Liver
            "bilirubin_total", "bilirubin_direct", "bilirubin_indirect", "ast", "alt", "alp", "albumin", "total_protein",
            # Coagulation
            "inr", "pt", "ptt",
            # Metabolic
            "glucose", "hba1c", "lactate",
            # Pancreatic
            "amylase", "lipase",
            # Cardiac
            "troponin_i", "ckmb", "bnp",
            # Electrolytes
            "sodium", "potassium", "chloride", "co2", "calcium", "magnesium", "phosphorus"
        ]
        # Filter to only variables that exist in the data
        chem_vars_available = [v for v in chem_vars if v in data["chemistry"].columns]
        if chem_vars_available:
            chem_features = compute_temporal_features_batch(
                data["chemistry"],
                patient_id,
                value_cols=chem_vars_available,
                time_col="charttime",
                reference_time=reference_time,
            )
            features.update(chem_features)

    if not data["fluid_balance"].empty:
        if "urine_output" in data["fluid_balance"].columns:
            urine_features = compute_temporal_features(
                data["fluid_balance"],
                patient_id,
                value_col="urine_output",
                time_col="charttime",
                reference_time=reference_time,
            )
            features.update({f"kidney_urine_{k}": v for k, v in urine_features.items()})

    # (Liver features already extracted above in chemistry section)

    # Hemodynamic features (VECTORIZED)
    if not data["hemodynamics"].empty:
        hemo_vars = ["sbp", "dbp", "heart_rate", "temperature", "urine_output"]
        hemo_vars_available = [v for v in hemo_vars if v in data["hemodynamics"].columns]
        if hemo_vars_available:
            hemo_features = compute_temporal_features_batch(
                data["hemodynamics"],
                patient_id,
                value_cols=hemo_vars_available,
                time_col="charttime",
                reference_time=reference_time,
            )
            features.update(hemo_features)

    # ABG features (VECTORIZED)
    if not data["abg"].empty:
        abg_vars = ["ph", "paco2", "pao2", "hco3", "base_excess", "spo2", "fio2", "peep", "respiratory_rate", "tidal_volume", "peak_pressure"]
        abg_vars_available = [v for v in abg_vars if v in data["abg"].columns]
        if abg_vars_available:
            abg_features = compute_temporal_features_batch(
                data["abg"],
                patient_id,
                value_cols=abg_vars_available,
                time_col="charttime",
                reference_time=reference_time,
            )
            features.update(abg_features)

    # CBC features (VECTORIZED)
    if not data["cbc"].empty:
        cbc_vars = ["wbc", "hemoglobin", "hematocrit", "rbc", "platelets", "lymphocytes", "monocytes", "eosinophils", "neutrophils_seg", "neutrophils_band"]
        cbc_vars_available = [v for v in cbc_vars if v in data["cbc"].columns]
        if cbc_vars_available:
            cbc_features = compute_temporal_features_batch(
                data["cbc"],
                patient_id,
                value_cols=cbc_vars_available,
                time_col="charttime",
                reference_time=reference_time,
            )
            features.update(cbc_features)

    # Serologies & infections as last-known statuses (binary where appropriate)
    if not data["serology"].empty:
        sero_df = data["serology"]
        sero = sero_df.loc[sero_df["patient_id"] == patient_id].copy()
        if not sero.empty:
            # Last row by time if available
            sero_time_col = "charttime" if "charttime" in sero.columns else None
            if sero_time_col:
                sero[sero_time_col] = pd.to_datetime(sero[sero_time_col], errors="coerce")
                sero = sero.sort_values(sero_time_col)
                sero_last = sero.iloc[-1]
            else:
                sero_last = sero.iloc[-1]

            for marker in ["hiv", "hepc", "hbv", "cmv"]:
                # Adapt marker column names to true schema
                for candidate in [marker, f"anti_{marker}", f"{marker}_status"]:
                    if candidate in sero.columns:
                        val = sero_last[candidate]
                        features[f"sero_{marker}_last"] = val
                        features[f"sero_{marker}_positive"] = (
                            str(val).lower() in {"positive", "pos", "reactive", "detected"}
                        )
                        break

    # Culture summary: any positive culture?
    if not data["culture"].empty:
        cult_df = data["culture"].loc[data["culture"]["patient_id"] == patient_id]
        if not cult_df.empty and "result" in cult_df.columns:
            any_positive = cult_df["result"].astype("string").str.contains(
                "positive", case=False, na=False
            ).any()
            features["culture_any_positive"] = int(any_positive)

    # ========================================================================
    # DERIVED VARIABLES (computed from extracted features)
    # ========================================================================
    
    # P/F Ratio (PaO2 / FiO2) - critical for lung viability
    if "pao2_last" in features and "fio2_last" in features:
        pao2 = features.get("pao2_last", np.nan)
        fio2 = features.get("fio2_last", np.nan)
        if not np.isnan(pao2) and not np.isnan(fio2) and fio2 > 0:
            features["pf_ratio"] = pao2 / (fio2 / 100.0)
        else:
            features["pf_ratio"] = np.nan
    
    # Mean Arterial Pressure (MAP) = (SBP + 2*DBP) / 3
    if "sbp_last" in features and "dbp_last" in features:
        sbp = features.get("sbp_last", np.nan)
        dbp = features.get("dbp_last", np.nan)
        if not np.isnan(sbp) and not np.isnan(dbp):
            features["map_calculated"] = (sbp + 2 * dbp) / 3.0
        else:
            features["map_calculated"] = np.nan
    
    # Hypotension flag (MAP < 65 mmHg)
    if "map_calculated" in features:
        map_val = features.get("map_calculated", np.nan)
        features["hypotension_flag"] = 1 if (not np.isnan(map_val) and map_val < 65) else 0
    
    # Oliguria flag (urine output < 0.5 mL/kg/hr)
    # Requires weight and urine output
    if "kidney_urine_urine_output_mean" in features and "bmi" in features:
        # Estimate weight from BMI and height
        bmi = features.get("bmi", np.nan)
        height_cm = features.get("height_cm", np.nan)
        if not np.isnan(bmi) and not np.isnan(height_cm) and height_cm > 0:
            weight_kg = bmi * (height_cm / 100.0) ** 2
            urine_ml_hr = features.get("kidney_urine_urine_output_mean", np.nan)
            if not np.isnan(urine_ml_hr) and weight_kg > 0:
                urine_ml_kg_hr = urine_ml_hr / weight_kg
                features["urine_ml_kg_hr"] = urine_ml_kg_hr
                features["oliguria_flag"] = 1 if urine_ml_kg_hr < 0.5 else 0
            else:
                features["urine_ml_kg_hr"] = np.nan
                features["oliguria_flag"] = 0
    
    # ARDS classification based on P/F ratio
    if "pf_ratio" in features:
        pf = features.get("pf_ratio", np.nan)
        if not np.isnan(pf):
            if pf < 100:
                features["ards_severity"] = "severe"
            elif pf < 200:
                features["ards_severity"] = "moderate"
            elif pf < 300:
                features["ards_severity"] = "mild"
            else:
                features["ards_severity"] = "none"
        else:
            features["ards_severity"] = "unknown"
    
    # Acute Kidney Injury (AKI) stage based on creatinine
    # Stage 1: Cr 1.5-1.9x baseline or increase ≥0.3 mg/dL
    # Stage 2: Cr 2.0-2.9x baseline
    # Stage 3: Cr ≥3.0x baseline or Cr ≥4.0 mg/dL
    # (We use absolute thresholds as baseline is not available)
    if "creatinine_last" in features:
        cr = features.get("creatinine_last", np.nan)
        if not np.isnan(cr):
            if cr >= 4.0:
                features["aki_stage"] = 3
            elif cr >= 2.5:
                features["aki_stage"] = 2
            elif cr >= 1.5:
                features["aki_stage"] = 1
            else:
                features["aki_stage"] = 0
        else:
            features["aki_stage"] = np.nan
    
    # Liver dysfunction flag (based on bilirubin and INR)
    if "bilirubin_total_last" in features and "inr_last" in features:
        bili = features.get("bilirubin_total_last", np.nan)
        inr = features.get("inr_last", np.nan)
        # Moderate liver dysfunction: bilirubin > 2.0 mg/dL or INR > 1.5
        if (not np.isnan(bili) and bili > 2.0) or (not np.isnan(inr) and inr > 1.5):
            features["liver_dysfunction_flag"] = 1
        else:
            features["liver_dysfunction_flag"] = 0

    return features


# ============================================================================
# ORGAN-LEVEL EXPANSION
# ============================================================================

def create_organ_level_records(
    patient_features_df: pd.DataFrame,
    referrals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand patient-level features into organ-level records.

    Each patient contributes up to 8 rows, one per organ_type in ORGANS.

    Outcome columns are joined from referrals:
      - outcome_{organ}
      - outcome_{organ}_procured_binary
      - outcome_{organ}_transplanted_binary
      - outcome_{organ}_research_binary
    """
    organ_records = []

    # Make referrals indexed by patient_id for quick lookup
    referrals_indexed = referrals.set_index("patient_id")

    # Iterate using itertuples for speed
    for row in patient_features_df.itertuples(index=False):
        pid = getattr(row, "patient_id")
        try:
            ref_row = referrals_indexed.loc[pid]
        except KeyError:
            continue

        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())

        for organ in ORGANS:
            record = dict(row_dict)
            record["organ_type"] = organ

            outcome_col = f"outcome_{organ}"
            for suffix in [
                "",
                "_procured_binary",
                "_transplanted_binary",
                "_research_binary",
            ]:
                col = f"{outcome_col}{suffix}"
                if col in referrals_indexed.columns:
                    record[col] = ref_row[col]
                else:
                    record[col] = np.nan

            organ_records.append(record)

    organ_df = pd.DataFrame(organ_records)
    return organ_df


# ============================================================================
# MISSINGNESS & DATA QUALITY
# ============================================================================

def analyze_missingness(
    organ_df: pd.DataFrame,
    referrals: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Compute missingness at the organ level and stratified by race, OPO, donation_type.

    Returns
    -------
    missingness_df : DataFrame
        Column-wise percent missing.
    data_quality_report : dict
        JSON-serializable dict with:
          - high_missingness_features
          - differential_missingness_tests
    """
    print("[analyze_missingness] Analyzing missingness")

    # Overall missingness per column
    missingness = organ_df.isna().mean().rename("fraction_missing").to_frame()

    # Stratified missingness for a small set of key features
    strat_vars = ["race", "opo", "donation_type"]
    differential_results: Dict[str, dict] = {}

    # Merge stratification columns from referrals (patient-level) into organ_df
    merge_cols = ["patient_id"] + [v for v in strat_vars if v in referrals.columns]
    tmp = organ_df.merge(
        referrals[merge_cols].drop_duplicates("patient_id"), on="patient_id", how="left"
    )

    for col in organ_df.columns:
        if col in ["patient_id", "organ_type"]:
            continue
        col_missing = tmp[col].isna().astype(int)

        differential_results[col] = {}
        for strat in strat_vars:
            if strat not in tmp.columns:
                continue
            # Build contingency table: missing vs non-missing by strat category
            sub = tmp[[col, strat]].copy()
            sub["missing"] = sub[col].isna().astype(int)
            contingency = pd.crosstab(sub["missing"], sub[strat])
            if contingency.shape[0] != 2:
                # If a feature is never missing or always missing, skip test
                continue
            try:
                chi2, p, dof, _ = stats.chi2_contingency(contingency)
                differential_results[col][strat] = {
                    "chi2": float(chi2),
                    "p_value": float(p),
                    "dof": int(dof),
                }
            except Exception:
                continue

    # Identify high-missingness and concerning features
    high_missing_features = missingness.loc[missingness["fraction_missing"] > 0.4].index.tolist()

    # Features with significant differential missingness at p<0.01
    diff_missing_features = []
    for col, strat_dict in differential_results.items():
        for strat, res in strat_dict.items():
            if res["p_value"] < 0.01:
                diff_missing_features.append({"feature": col, "stratifier": strat, **res})

    # Build data-quality report
    data_quality_report = {
        "high_missingness_features": high_missing_features,
        "differential_missingness_tests": diff_missing_features,
    }

    # Save artifacts
    missingness_path = output_dir / "missingness_report.csv"
    missingness.to_csv(missingness_path, index=True)
    print(f"[analyze_missingness] Saved missingness report to: {missingness_path}")

    dq_path = output_dir / "data_quality_report.json"
    with dq_path.open("w") as f:
        json.dump(data_quality_report, f, indent=2)
    print(f"[analyze_missingness] Saved data quality report to: {dq_path}")

    return missingness, data_quality_report


# ============================================================================
# OPTIONAL MULTI-STAGE IMPUTATION
# ============================================================================

def impute_missing_data(organ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a simple multi-stage imputation:

      1. For numeric features:
         - median imputation (global)
      2. For categorical features:
         - mode imputation
      3. For a selected subset of correlated numeric features:
         - IterativeImputer (MICE-like) refinement

    NOTE:
      - This is a generic template. You should customize the list of columns
        for MICE and document imputation choices in your Methods.
    """
    df = organ_df.copy()
    print("[impute_missing_data] Starting imputation")

    # Separate numeric and non-numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # 1. Median for numeric
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # 2. Mode for categorical / string
    for col in non_numeric_cols:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    # 3. MICE-like refinement on a subset of numeric columns (example)
    # Select a subset of numeric features to impute jointly
    candidate_mice_cols = [
        c
        for c in numeric_cols
        if any(
            key in c
            for key in [
                "creatinine",
                "bun",
                "egfr",
                "bilirubin",
                "ast",
                "alt",
                "sbp",
                "dbp",
                "map",
                "pao2",
                "paco2",
                "hgb",
                "wbc",
                "platelet",
            ]
        )
    ]
    if candidate_mice_cols:
        print(
            f"[impute_missing_data] Running IterativeImputer on "
            f"{len(candidate_mice_cols)} numeric features"
        )
        imputer = IterativeImputer(random_state=42, max_iter=10, sample_posterior=False)
        df[candidate_mice_cols] = imputer.fit_transform(df[candidate_mice_cols])

    print("[impute_missing_data] Imputation complete")
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_unified_orchid_database(
    data_dir: Path,
    output_dir: Path,
    perform_imputation: bool = False,
) -> None:
    """
    Main entry point to build unified organ-level ORCHID database.

    Steps:
      1. Load raw ORCHID tables.
      2. Extract patient-level features from referrals + event tables.
      3. Expand to organ-level structure (8 organs per patient).
      4. Analyze missingness and generate data-quality report.
      5. Optionally perform multi-stage imputation.
      6. Save final organ-level database as Parquet.
    """
    # 1. Load raw ORCHID tables
    data = load_orchid_data(data_dir)
    referrals = data["referrals"]

    # 2. Extract patient-level features (PARALLELIZED)
    import time
    print("[build_unified_orchid_database] Extracting patient-level features")
    patient_ids = referrals["patient_id"].unique()
    total_patients = len(patient_ids)
    
    # Determine number of CPU cores to use
    n_cores = cpu_count()
    print(f"[build_unified_orchid_database] Total patients to process: {total_patients:,}")
    print(f"[build_unified_orchid_database] Using {n_cores} CPU cores for parallel processing")
    
    patient_feature_dicts: List[Dict[str, float]] = []

    start_time = time.time()
    last_update_time = start_time
    update_interval = 10  # seconds between progress updates
    
    # Create a partial function with data pre-loaded
    extract_func = partial(extract_patient_level_features, data=data)
    
    # Process in parallel with progress tracking
    with Pool(processes=n_cores) as pool:
        # Use imap for lazy evaluation with progress tracking
        # Larger chunksize (1000) reduces overhead significantly
        results_iter = pool.imap(extract_func, patient_ids, chunksize=1000)
        
        for idx, feats in enumerate(results_iter, 1):
            if feats:
                patient_feature_dicts.append(feats)
            
            
            # Progress tracking with ETA
            current_time = time.time()
            if current_time - last_update_time >= update_interval or idx == total_patients:
                elapsed = current_time - start_time
                percent_complete = (idx / total_patients) * 100
                patients_per_sec = idx / elapsed if elapsed > 0 else 0
                
                if patients_per_sec > 0:
                    remaining_patients = total_patients - idx
                    eta_seconds = remaining_patients / patients_per_sec
                    eta_minutes = eta_seconds / 60
                    eta_hours = eta_minutes / 60
                    
                    if eta_hours >= 1:
                        eta_str = f"{eta_hours:.1f}h"
                    elif eta_minutes >= 1:
                        eta_str = f"{eta_minutes:.1f}m"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = "calculating..."
                
                print(
                    f"[build_unified_orchid_database] Progress: {idx:,}/{total_patients:,} "
                    f"({percent_complete:.1f}%) | "
                    f"Speed: {patients_per_sec:.1f} patients/sec | "
                    f"ETA: {eta_str} | "
                    f"Elapsed: {elapsed/60:.1f}m"
                )
                last_update_time = current_time

    total_extraction_time = time.time() - start_time
    patient_features_df = pd.DataFrame(patient_feature_dicts)
    print(
        f"\n[build_unified_orchid_database] ✓ Feature extraction complete!"
    )
    print(
        f"[build_unified_orchid_database] Total time: {total_extraction_time/60:.1f} minutes "
        f"({total_extraction_time/3600:.2f} hours)"
    )
    print(
        f"[build_unified_orchid_database] Patient-level feature shape: {patient_features_df.shape}"
    )
    print(
        f"[build_unified_orchid_database] Features extracted: {patient_features_df.shape[1]} columns"
    )

    # 3. Create organ-level records
    print("[build_unified_orchid_database] Creating organ-level records")
    organ_df = create_organ_level_records(patient_features_df, referrals)
    print("[build_unified_orchid_database] Organ-level shape:", organ_df.shape)

    # 4. Missingness & data quality
    analyze_missingness(organ_df, referrals, output_dir)

    # 5. Optional imputation
    if perform_imputation:
        print("[build_unified_orchid_database] Performing multi-stage imputation")
        organ_df = impute_missing_data(organ_df)
    else:
        print(
            "[build_unified_orchid_database] Skipping imputation "
            "(organ_df will contain NaNs; modeling code must handle them explicitly)"
        )

    # 6. Save final organ-level database
    out_path = output_dir / "organs_database.parquet"
    organ_df.to_parquet(out_path, index=False)
    print(f"[build_unified_orchid_database] Saved organ-level DB to: {out_path}")

    # Final warning about clustering
    print(
        "\n[WARNING] All downstream models using organs_database.parquet MUST "
        "account for clustering by patient_id (e.g., mixed-effects models, "
        "cluster-robust SEs). Organs from the same donor are NOT IID."
    )


if __name__ == "__main__":
    build_unified_orchid_database(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        perform_imputation=False,  # set to True if you want imputed DB
    )
