import pandas as pd

# =============================================================================
# WEEK 1 — SOLD DATASET CONCATENATION
# =============================================================================

# Load all monthly sold files (January 2024 through April 2026)
sold_files = [
    'data/01_raw/CRMLSSold202401.csv', 'data/01_raw/CRMLSSold202402.csv', 'data/01_raw/CRMLSSold202403.csv',
    'data/01_raw/CRMLSSold202404.csv', 'data/01_raw/CRMLSSold202405.csv', 'data/01_raw/CRMLSSold202406.csv',
    'data/01_raw/CRMLSSold202407.csv', 'data/01_raw/CRMLSSold202408.csv', 'data/01_raw/CRMLSSold202409.csv',
    'data/01_raw/CRMLSSold202410.csv', 'data/01_raw/CRMLSSold202411.csv', 'data/01_raw/CRMLSSold202412.csv',
    'data/01_raw/CRMLSSold202501.csv', 'data/01_raw/CRMLSSold202502.csv', 'data/01_raw/CRMLSSold202503.csv',
    'data/01_raw/CRMLSSold202504.csv', 'data/01_raw/CRMLSSold202505.csv', 'data/01_raw/CRMLSSold202506.csv',
    'data/01_raw/CRMLSSold202507.csv', 'data/01_raw/CRMLSSold202508.csv', 'data/01_raw/CRMLSSold202509.csv',
    'data/01_raw/CRMLSSold202510.csv', 'data/01_raw/CRMLSSold202511.csv', 'data/01_raw/CRMLSSold202512.csv',
    'data/01_raw/CRMLSSold202601.csv', 'data/01_raw/CRMLSSold202602.csv', 'data/01_raw/CRMLSSold202603.csv',
    'data/01_raw/CRMLSSold202604.csv'
]

# Load each file and concatenate into one combined dataset
sold_dfs = []
for file in sold_files:
    df = pd.read_csv(file, low_memory=False)
    sold_dfs.append(df)

sold = pd.concat(sold_dfs, ignore_index=True)

# Row count BEFORE Residential filter
print(f"\nSOLD - Total rows after concatenation (all property types): {len(sold)}")

# Filter to Residential only
sold = sold[sold['PropertyType'] == 'Residential']

# Row count AFTER Residential filter
print(f"SOLD - Total rows after Residential filter: {len(sold)}")

# =============================================================================
# WEEKS 2-3 — EDA & VALIDATION
# =============================================================================

# Shape: rows and columns
print(f"SOLD dataset shape: {sold.shape}")

# ── PROPERTY TYPE BREAKDOWN ───────────────────────────────────────────────────

# Load a single raw monthly file to show property type share before filtering
sample = pd.read_csv('data/01_raw/CRMLSSold202604.csv', low_memory=False)
print("\nProperty type breakdown (sample from April 2026):")
print(sample['PropertyType'].value_counts())
print(f"\nResidential share: {sample['PropertyType'].value_counts(normalize=True).get('Residential', 0)*100:.1f}%")

# ── MISSING VALUE ANALYSIS ────────────────────────────────────────────────────

# Calculate missing counts and percentages per column
missing = pd.DataFrame({
    'missing_count': sold.isnull().sum(),
    'missing_pct': (sold.isnull().sum() / len(sold) * 100).round(2)
})
missing = missing.sort_values('missing_pct', ascending=False)

# Flag columns above 90% missing
high_missing = missing[missing['missing_pct'] > 90]
print(f"Missing Value Report (90%+ null) ({len(high_missing)} total):")
print(high_missing.to_string())

# Flagged columns as a list. 
# Note: columns will not be dropped yet.
columns_to_drop = high_missing.index.tolist()

# ── NUMERIC DISTRIBUTION SUMMARY ─────────────────────────────────────────────

# Numeric distribution summary for deliverable fields
print("\nNumeric distribution summary (ClosePrice, LivingArea, DaysOnMarket):")
print(sold[['ClosePrice', 'LivingArea', 'DaysOnMarket']]
      .describe(percentiles=[.10, .25, .50, .75, .90, .95, .99])
      .to_string())

# =============================================================================
# WEEKS 4-5 — DATA CLEANING AND PREPARATION (coming soon)
# =============================================================================


# =============================================================================
# WEEK 6 — FEATURE ENGINEERING (coming soon)
# =============================================================================


# =============================================================================
# WEEK 7 — OUTLIER DETECTION (coming soon)
# =============================================================================


# =============================================================================
# FINAL OUTPUT — Save clean CSV for Tableau
# =============================================================================

sold.to_csv('data/02_intermediate/CRMLSSold_Combined.csv', index=False)
print(f"\nSaved CRMLSSold_Combined.csv — {len(sold):,} rows")