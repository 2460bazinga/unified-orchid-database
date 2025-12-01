"""
Example Usage: Unified ORCHID Database

This script demonstrates how to load and analyze the unified ORCHID database.

Author: Noah (noah@2460.life)
ORCID: 0009-0002-9412-6968
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# =============================================================================
# 1. Load the Database
# =============================================================================

print("Loading unified ORCHID database...")
df = pd.read_parquet('organs_database.parquet')

print(f"Database shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Unique patients: {df['patient_id'].nunique():,}")
print(f"Organ types: {df['organ_type'].unique()}")

# =============================================================================
# 2. Basic Exploratory Analysis
# =============================================================================

print("\n" + "="*80)
print("EXPLORATORY ANALYSIS")
print("="*80)

# Overall procurement rates by organ
print("\nProcurement rates by organ type:")
procurement_by_organ = df.groupby('organ_type')['outcome_procured'].agg(['mean', 'sum', 'count'])
procurement_by_organ.columns = ['Rate', 'Procured', 'Total']
procurement_by_organ['Rate'] = (procurement_by_organ['Rate'] * 100).round(2)
print(procurement_by_organ.sort_values('Rate', ascending=False))

# Procurement rates by race
print("\nProcurement rates by race:")
procurement_by_race = df.groupby('race')['outcome_procured'].agg(['mean', 'count'])
procurement_by_race.columns = ['Rate', 'Total']
procurement_by_race['Rate'] = (procurement_by_race['Rate'] * 100).round(2)
print(procurement_by_race.sort_values('Rate', ascending=False))

# Procurement rates by OPO
print("\nProcurement rates by OPO:")
procurement_by_opo = df.groupby('opo_code')['outcome_procured'].agg(['mean', 'count'])
procurement_by_opo.columns = ['Rate', 'Total']
procurement_by_opo['Rate'] = (procurement_by_opo['Rate'] * 100).round(2)
print(procurement_by_opo.sort_values('Rate', ascending=False))

# =============================================================================
# 3. Organ-Specific Analysis: Kidneys
# =============================================================================

print("\n" + "="*80)
print("KIDNEY-SPECIFIC ANALYSIS")
print("="*80)

# Filter to kidneys only
kidneys = df[df['organ_type'].isin(['kidney_left', 'kidney_right'])].copy()
print(f"\nKidney records: {len(kidneys):,}")
print(f"Procurement rate: {kidneys['outcome_procured'].mean()*100:.2f}%")

# Analyze creatinine trajectory
print("\nCreatinine trajectory analysis:")
kidneys['creatinine_improving'] = kidneys['creatinine_slope'] < 0  # Negative slope = improving
improving_procurement = kidneys.groupby('creatinine_improving')['outcome_procured'].mean()
print(f"  Improving creatinine: {improving_procurement.get(True, 0)*100:.2f}% procurement")
print(f"  Worsening creatinine: {improving_procurement.get(False, 0)*100:.2f}% procurement")

# =============================================================================
# 4. Fairness Analysis
# =============================================================================

print("\n" + "="*80)
print("FAIRNESS ANALYSIS")
print("="*80)

# Analyze procurement disparities by race and OPO
print("\nProcurement rates by race and OPO:")
fairness_analysis = df.pivot_table(
    values='outcome_procured',
    index='race',
    columns='opo_code',
    aggfunc='mean'
) * 100
print(fairness_analysis.round(2))

# =============================================================================
# 5. Machine Learning Example: Kidney Procurement Prediction
# =============================================================================

print("\n" + "="*80)
print("MACHINE LEARNING EXAMPLE: KIDNEY PROCUREMENT PREDICTION")
print("="*80)

# Prepare data for ML
print("\nPreparing data for machine learning...")

# Select features (exclude identifiers, outcomes, and protected attributes)
exclude_cols = [
    'patient_id', 'organ_type', 'opo_code', 'hospital_id',
    'outcome_procured', 'outcome_transplanted', 'outcome_research',
    'race', 'gender',  # Exclude protected attributes from model
    'time_referred', 'time_approached', 'time_authorized', 'time_procured',
    'brain_death', 'cause_of_death'  # Avoid collider bias
]

feature_cols = [col for col in kidneys.columns if col not in exclude_cols]

# Handle missing values (simple approach for demonstration)
X = kidneys[feature_cols].fillna(kidneys[feature_cols].median())
y = kidneys['outcome_procured']

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X):,}")
print(f"Positive class (procured): {y.sum():,} ({y.mean()*100:.2f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Train model
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Procured', 'Procured']))

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# =============================================================================
# 6. Fairness Audit: Check for Disparities
# =============================================================================

print("\n" + "="*80)
print("FAIRNESS AUDIT")
print("="*80)

# Add predictions to test set
test_kidneys = kidneys.loc[X_test.index].copy()
test_kidneys['predicted_procurement'] = y_pred_proba

# Analyze prediction disparities by race
print("\nPrediction scores by race:")
race_audit = test_kidneys.groupby('race').agg({
    'predicted_procurement': 'mean',
    'outcome_procured': 'mean'
})
race_audit.columns = ['Mean Predicted Score', 'Actual Procurement Rate']
race_audit = race_audit * 100
print(race_audit.round(2))

# Check for calibration disparities
print("\nCalibration by race (predicted vs. actual):")
for race in test_kidneys['race'].unique():
    race_data = test_kidneys[test_kidneys['race'] == race]
    pred_mean = race_data['predicted_procurement'].mean()
    actual_mean = race_data['outcome_procured'].mean()
    diff = pred_mean - actual_mean
    print(f"  {race}: Predicted={pred_mean*100:.2f}%, Actual={actual_mean*100:.2f}%, Diff={diff*100:+.2f}%")

# =============================================================================
# 7. Geographic Analysis
# =============================================================================

print("\n" + "="*80)
print("GEOGRAPHIC ANALYSIS")
print("="*80)

# Analyze procurement by donor service area characteristics
print("\nProcurement by donor service area characteristics:")
geo_analysis = df.groupby('opo_rural_indicator').agg({
    'outcome_procured': 'mean',
    'patient_id': 'count'
})
geo_analysis.columns = ['Procurement Rate', 'Count']
geo_analysis['Procurement Rate'] = geo_analysis['Procurement Rate'] * 100
geo_analysis.index = ['Urban', 'Rural']
print(geo_analysis.round(2))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nThis example demonstrates:")
print("  ✓ Loading and exploring the unified database")
print("  ✓ Organ-specific analysis (kidneys)")
print("  ✓ Fairness analysis across race and OPO")
print("  ✓ Machine learning model training")
print("  ✓ Fairness audit of model predictions")
print("  ✓ Geographic disparity analysis")
print("\nFor more information, see the README.md file.")
