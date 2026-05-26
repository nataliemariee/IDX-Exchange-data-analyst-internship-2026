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
    print(f"Loaded {file}: {len(df)} rows")

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

# ── DATASET UNDERSTANDING ─────────────────────────────────────────────────────

# Shape: rows and columns
print(f"SOLD dataset shape: {sold.shape}")

# Column data types
print("\nColumn data types:")
print(sold.dtypes.to_string())

# ── MISSING VALUE ANALYSIS ────────────────────────────────────────────────────

# Calculate missing counts and percentages per column
missing = pd.DataFrame({
    'missing_count': sold.isnull().sum(),
    'missing_pct': (sold.isnull().sum() / len(sold) * 100).round(2)
})
missing = missing.sort_values('missing_pct', ascending=False)

# Flag columns above 90% missing
high_missing = missing[missing['missing_pct'] > 90]
print(f"Columns with >90% missing values ({len(high_missing)} total):")
print(high_missing.to_string())

# Flagged columns as a list. 
# Note: columns will not be dropped yet.
columns_to_drop = high_missing.index.tolist()
print(f"Columns flagged for dropping ({len(columns_to_drop)} total):")
print(columns_to_drop)

# ── NUMERIC DISTRIBUTION SUMMARY ─────────────────────────────────────────────

# Numeric distribution summary for deliverable fields
print("\nNumeric distribution summary (ClosePrice, LivingArea, DaysOnMarket):")
print(sold[['ClosePrice', 'LivingArea', 'DaysOnMarket']]
      .describe(percentiles=[.10, .25, .50, .75, .90, .95, .99])
      .to_string())

# ── SUGGESTED INTERN QUESTIONS ────────────────────────────────────────────────

# Median and average close prices
print(f"\nMedian close price: ${sold['ClosePrice'].median():,.0f}")
print(f"Average close price: ${sold['ClosePrice'].mean():,.0f}")

# Homes sold above vs below list price
sold['sold_above_list'] = sold['ClosePrice'] >= sold['ListPrice']
pct_above = sold['sold_above_list'].mean() * 100
print(f"\nHomes sold at or above list price: {pct_above:.1f}%")
print(f"Homes sold below list price: {100 - pct_above:.1f}%")

# Top 10 counties by median close price
print("\nTop 10 counties by median close price:")
print(sold.groupby('CountyOrParish')['ClosePrice'].median()
      .sort_values(ascending=False).head(10)
      .apply(lambda x: f"${x:,.0f}"))

# Days on market distribution
print(f"\nDays on Market median: {sold['DaysOnMarket'].median():.0f} days")
print(f"Days on Market average: {sold['DaysOnMarket'].mean():.1f} days")
print(f"Negative DaysOnMarket (bad data): {(sold['DaysOnMarket'] < 0).sum()}")
print(f"DaysOnMarket over 365: {(sold['DaysOnMarket'] > 365).sum()}")

# ── DATE CONSISTENCY CHECKS ───────────────────────────────────────────────────

sold['CloseDate'] = pd.to_datetime(sold['CloseDate'])
sold['ListingContractDate'] = pd.to_datetime(sold['ListingContractDate'])
sold['PurchaseContractDate'] = pd.to_datetime(sold['PurchaseContractDate'])

print(f"\nClose date before listing date: {(sold['CloseDate'] < sold['ListingContractDate']).sum()}")
print(f"Close date before purchase contract date: {(sold['CloseDate'] < sold['PurchaseContractDate']).sum()}")

# ── PROPERTY TYPE BREAKDOWN ───────────────────────────────────────────────────

# Load a single raw monthly file to show property type share before filtering
sample = pd.read_csv('data/01_raw/CRMLSSold202604.csv', low_memory=False)
print("\nProperty type breakdown (sample from April 2026):")
print(sample['PropertyType'].value_counts())
print(f"\nResidential share: {sample['PropertyType'].value_counts(normalize=True).get('Residential', 0)*100:.1f}%")


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