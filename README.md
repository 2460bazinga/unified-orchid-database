# Unified ORCHID Database Builder

A comprehensive data transformation pipeline that converts the ORCHID (Organ Retrieval and Collection of Health Information for Donation) dataset into an organ-level analytical database optimized for machine learning and statistical analysis of organ donation outcomes.

## Why This Database?

The raw ORCHID dataset, while comprehensive, presents significant challenges for organ-specific research and machine learning:

### Limitations of Raw ORCHID Data
- **Patient-level structure**: Each row represents a patient, not an organ, making organ-specific analysis cumbersome
- **Sparse temporal data**: Clinical measurements are scattered across multiple event tables with timestamps
- **Missing geographic context**: No donor service area characteristics or spatial features
- **Complex joins required**: Analyzing a single organ requires joining 8+ tables
- **No temporal trajectories**: Raw measurements lack slopes, trends, and clinical interpretations
- **Inconsistent variable names**: Different tables use different naming conventions

### What This Database Provides

**1. Organ-Level Granularity**
- Transforms 133,101 patients → **1,064,808 organ-donor pairs** (8 organs per patient)
- Each row represents a single organ from a single donor
- Enables organ-specific machine learning models and statistical analyses
- Preserves patient-level clustering information for proper statistical inference

**2. Temporal Feature Engineering (~750 features)**
- Extracts clinical trajectories from time-series data (slopes, deltas, trends)
- Computes summary statistics (last, mean, min, max, std) for each variable
- Identifies "improving" vs. "worsening" clinical patterns
- Calculates derived metrics (P/F ratio for lungs, KDPI components for kidneys, MELD for liver)

**3. Geographic Integration**
- Adds donor service area characteristics (size, compactness, rural/urban)
- Enables spatial analysis of procurement disparities
- Supports policy research on geographic equity

**4. Comprehensive Missingness Analysis**
- Identifies systematic bias in data collection
- Stratifies missingness by race, OPO, and donation type
- Provides transparency for fairness-aware modeling

**5. Ready for Machine Learning**
- Single, wide table optimized for analytical queries
- Parquet format (columnar, compressed) for fast loading
- No complex joins required
- Proper handling of missing data with multiple imputation strategies

## Use Cases

This database enables research that is difficult or impossible with raw ORCHID data:

- **Organ-specific procurement prediction**: Build separate models for kidneys, liver, heart, lungs
- **Fairness audits**: Analyze disparities in organ procurement by race, geography, OPO
- **Restless Multi-Armed Bandits (RMAB)**: Optimize donor approach timing with organ-level state spaces
- **Temporal pattern analysis**: Identify clinical trajectories associated with successful procurement
- **Geographic equity studies**: Assess rural/urban and regional disparities
- **Causal inference**: Three-stage outcome decomposition (approach → authorization → procurement)

## Dataset

This pipeline processes the **ORCHID (Organ Retrieval and Collection of Health Information for Donation)** dataset, a multi-institutional database of deceased organ donors from 6 U.S. organ procurement organizations (OPOs).

**Source:** PhysioNet (requires credentialed access)  
**Citation:** Moazami N, et al. (2024). ORCHID: Organ Retrieval and Collection of Health Information for Donation. PhysioNet.

## Architecture

### Input Data Structure
```
ORCHID/
├── OPOReferrals.csv          # Patient demographics and outcomes (133,101 patients)
├── ChemistryEvents.csv        # Laboratory values (335,206 measurements)
├── ABGEvents.csv              # Arterial blood gases (243,242 measurements)
├── HemoEvents.csv             # Vital signs (1,588,272 measurements)
├── CBCEvents.csv              # Complete blood counts (306,745 measurements)
├── SerologyEvents.csv         # Infectious disease testing (298,544 tests)
├── CultureEvents.csv          # Microbiology cultures (74,315 cultures)
└── FluidBalanceEvents.csv     # Fluid intake/output (128,409 records)
```

### Output Database Structure
```
organs_database.parquet        # Main analytical database (1,064,808 rows, ~750 features)
missingness_report.csv         # Missingness analysis stratified by demographics
data_quality_report.json       # Summary statistics and quality metrics
```

## Features Extracted

### Patient-Level Features
- Demographics: age, gender, race, BMI, blood type
- Donation characteristics: brain death vs. DCD, cause of death
- Geographic: OPO, hospital, donor service area characteristics
- Timing: referral, approach, authorization, procurement timestamps

### Temporal Clinical Features (per variable)
For each clinical variable (chemistry, ABG, hemodynamics, CBC), the pipeline extracts:
- **Last value**: Most recent measurement before procurement
- **Mean, min, max**: Summary statistics over observation window
- **Slope**: Rate of change (temporal trajectory)
- **Delta**: Change from first to last measurement
- **Coefficient of variation**: Measure of stability
- **Number of measurements**: Data density indicator

### Organ-Specific Features
- **Kidneys**: Creatinine trajectory, urine output, kidney-specific risk scores
- **Liver**: Bilirubin, transaminases, INR, liver-specific viability indicators
- **Heart**: Troponin, ejection fraction, inotrope requirements
- **Lungs**: P/F ratio, ventilator settings, ARDS classification
- **Pancreas**: Glucose control, amylase, lipase
- **Intestines**: Lactate, bowel function indicators

### Geographic Features (per OPO)
- Donor service area size (km²)
- Service area compactness (shape metric)
- Rural vs. urban classification
- Multi-state indicator
- Border proximity
- Centroid coordinates

## Installation

### Requirements
- Python 3.8+
- 16 GB RAM minimum (for full dataset processing)
- 100 GB disk space

### Dependencies
```bash
pip install pandas numpy scipy scikit-learn numba
```

### Optional (for faster processing)
```bash
pip install numba  # JIT compilation for 2-3x speedup
```

## Usage

### Basic Usage
```bash
python create_unified_orchid_database_HYPER.py
```

### Configuration
Edit the configuration section at the top of the script to customize:
- Input data paths
- Output file locations
- Temporal window (default: full observation period)
- Imputation strategies
- Feature selection

### Expected Runtime
- **e2-standard-4 VM** (4 vCPUs, 16 GB RAM): ~85-90 minutes
- **e2-standard-8 VM** (8 vCPUs, 32 GB RAM): ~45-60 minutes
- **c2-standard-8 VM** (8 vCPUs, optimized): ~20-40 minutes

## Methodology

### 1. Data Loading and Indexing
- Loads all ORCHID CSV files
- Creates patient_id indexes for O(1) lookup
- Converts timestamps to datetime objects
- Standardizes variable names across tables

### 2. Temporal Feature Engineering
- Extracts time-series data for each patient
- Computes temporal statistics (mean, slope, delta, etc.)
- Uses vectorized operations for efficiency
- Applies Numba JIT compilation to critical functions

### 3. Organ-Level Disaggregation
- Expands each patient into 8 organ-donor pairs
- Assigns organ-specific features
- Creates binary outcome variables (procured: yes/no)
- Preserves patient-level clustering information

### 4. Geographic Integration
- Integrates HRSA GIS donor service area data
- Adds geographic features for spatial analysis
- Maps OPO characteristics to each record

### 5. Missingness Analysis
- Stratifies missingness by race, OPO, donation type
- Identifies systematic bias in data collection
- Generates comprehensive missingness report

### 6. Output Generation
- Writes final database to Parquet format (columnar, compressed)
- Generates missingness and quality reports
- Provides summary statistics

## Performance Optimizations

The hyper-optimized version includes:

1. **Pre-indexing**: Creates patient_id indexes on all event tables for O(1) lookup
2. **Vectorized feature extraction**: Processes all variables in a table simultaneously
3. **Numba JIT compilation**: Compiles hot functions to machine code
4. **Multiprocessing**: Distributes work across all CPU cores
5. **Large chunksizes**: Reduces multiprocessing overhead

These optimizations provide a **100-200x speedup** compared to naive implementations.

## Output Schema

The final `organs_database.parquet` file contains:

- **Identifiers**: patient_id, organ_type, opo_code, hospital_id
- **Demographics**: age, gender, race, bmi, blood_type
- **Donation characteristics**: donation_type (DBD/DCD), cause_of_death, brain_death
- **Temporal features**: ~600 clinical features with temporal statistics
- **Geographic features**: 8 OPO-level geographic variables
- **Outcomes**: outcome_procured (binary), outcome_transplanted, outcome_research
- **Timing**: time_referred, time_approached, time_authorized, time_procured

## Ethical Considerations

This database is designed for fairness-aware machine learning research:

- **Race and gender are NOT included as predictive features** in models
- **Race and gender ARE available for fairness audits** and disparity analysis
- **Collider bias avoidance**: Excludes brain_death and cause_of_death from predictive models
- **Three-stage outcome decomposition**: Approach → Authorization → Procurement
- **Comprehensive bias detection**: Identifies systematic disparities in data collection

## Example: Loading the Database

```python
import pandas as pd

# Load the unified database
df = pd.read_parquet('organs_database.parquet')

# Filter to kidney-specific analysis
kidneys = df[df['organ_type'].isin(['kidney_left', 'kidney_right'])]

# Analyze procurement rates by race
procurement_by_race = kidneys.groupby('race')['outcome_procured'].mean()
print(procurement_by_race)

# Build organ-specific model
from sklearn.ensemble import RandomForestClassifier
X = kidneys[feature_columns]  # Exclude race, gender, outcomes
y = kidneys['outcome_procured']
model = RandomForestClassifier()
model.fit(X, y)
```

## Citation

If you use this database builder in your research, please cite:

```
Noah (2024). Unified ORCHID Database Builder. 
GitHub repository: https://github.com/2460bazinga/unified-orchid-database
ORCID: 0009-0002-9412-6968
```

And cite the original ORCHID dataset:

```
Moazami N, et al. (2024). ORCHID: Organ Retrieval and Collection of Health 
Information for Donation. PhysioNet. https://doi.org/10.13026/orchid
```

## License

MIT License - see LICENSE file for details

## Contact

Noah - noah@2460.life  
ORCID: [0009-0002-9412-6968](https://orcid.org/0009-0002-9412-6968)

## Acknowledgments

This project builds upon the ORCHID dataset created by a multi-institutional team led by Dr. Nader Moazami at NYU Langone Health. The database builder was developed to support fairness-aware machine learning research in organ donation.
