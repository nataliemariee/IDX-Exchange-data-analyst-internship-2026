import pandas as pd

# =============================================================================
# WEEK 1 — LISTINGS DATASET CONCATENATION
# =============================================================================

# Load all monthly listing files (January 2024 through April 2026)
listings_files = [
    'data/01_raw/CRMLSListing202401.csv', 'data/01_raw/CRMLSListing202402.csv', 'data/01_raw/CRMLSListing202403.csv',
    'data/01_raw/CRMLSListing202404.csv', 'data/01_raw/CRMLSListing202405.csv', 'data/01_raw/CRMLSListing202406.csv',
    'data/01_raw/CRMLSListing202407.csv', 'data/01_raw/CRMLSListing202408.csv', 'data/01_raw/CRMLSListing202409.csv',
    'data/01_raw/CRMLSListing202410.csv', 'data/01_raw/CRMLSListing202411.csv', 'data/01_raw/CRMLSListing202412.csv',
    'data/01_raw/CRMLSListing202501.csv', 'data/01_raw/CRMLSListing202502.csv', 'data/01_raw/CRMLSListing202503.csv',
    'data/01_raw/CRMLSListing202504.csv', 'data/01_raw/CRMLSListing202505.csv', 'data/01_raw/CRMLSListing202506.csv',
    'data/01_raw/CRMLSListing202507.csv', 'data/01_raw/CRMLSListing202508.csv', 'data/01_raw/CRMLSListing202509.csv',
    'data/01_raw/CRMLSListing202510.csv', 'data/01_raw/CRMLSListing202511.csv', 'data/01_raw/CRMLSListing202512.csv',
    'data/01_raw/CRMLSListing202601.csv', 'data/01_raw/CRMLSListing202602.csv', 'data/01_raw/CRMLSListing202603.csv',
    'data/01_raw/CRMLSListing202604.csv'
]

# Load each file and concatenate into one combined dataset
listings_dfs = []
for file in listings_files:
    df = pd.read_csv(file, low_memory=False)
    listings_dfs.append(df)

listings = pd.concat(listings_dfs, ignore_index=True)

# Row count BEFORE Residential filter
print(f"\nLISTINGS - Total rows after concatenation (all property types): {len(listings)}")

# Filter to Residential only
listings = listings[listings['PropertyType'] == 'Residential']

# Row count AFTER Residential filter
print(f"LISTINGS - Total rows after Residential filter: {len(listings)}")

# =============================================================================
# WEEKS 2-3 — EDA & VALIDATION
# =============================================================================

# Shape: rows and columns
print(f"LISTING dataset shape: {listings.shape}")

# ── PROPERTY TYPE BREAKDOWN ───────────────────────────────────────────────────

# Load a single raw monthly file to show property type share before filtering
sample = pd.read_csv('data/01_raw/CRMLSListing202604.csv', low_memory=False)
print("\nProperty type breakdown (sample from April 2026):")
print(sample['PropertyType'].value_counts())
print(f"\nResidential share: {sample['PropertyType'].value_counts(normalize=True).get('Residential', 0)*100:.1f}%")

# ── MISSING VALUE ANALYSIS ────────────────────────────────────────────────────

# Calculate missing counts and percentages per column
missing = pd.DataFrame({
    'missing_count': listings.isnull().sum(),
    'missing_pct': (listings.isnull().sum() / len(listings) * 100).round(2)
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
print(listings[['ClosePrice', 'LivingArea', 'DaysOnMarket']]
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

listings.to_csv('data/02_intermediate/CRMLSListing_Combined.csv', index=False)
print(f"\nSaved CRMLSListing_Combined.csv — {len(listings):,} rows")