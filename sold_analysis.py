import pandas as pd
import requests
import io

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

# ── MORTGAGE RATE ENRICHMENT ─────────────────────────────────────────────
# NOTE: Commented out due to FRED network timeout issue
# Will be re-enabled once network access to fred.stlouisfed.org is resolved
# Alternatively, load MORTGAGE30US.csv locally once downloaded manually

# Step 1 - Fetching FRED MORTGAGE30US  Series
# url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
# response = requests.get(url, verify=False)
# mortgage = pd.read_csv(io.StringIO(response.text), parse_dates=['observation_date'])
# mortgage.columns = ['date', 'rate_30yr_fixed']

# # Step 2 — Resample weekly rates to monthly averages
# mortgage['year_month'] = mortgage['date'].dt.to_period('M')
# mortgage_monthly = (
#     mortgage.groupby('year_month')['rate_30yr_fixed']
#     .mean().reset_index()
# )

# # Step 3 – Create a matching year_month key on the sold dataset
# # Sold dataset — key off CloseDate
# sold["year_month"] = pd.to_datetime(sold["CloseDate"]).dt.to_period("M")

# # Step 4 – Merge
# sold = sold.merge(mortgage_monthly, on="year_month", how="left")

# # Step 5 – Validate the merge
# # Check for any unmatched rows (rate should not be null)
# print(sold["rate_30yr_fixed"].isnull().sum())

# =============================================================================
# WEEKS 4-5 — DATA CLEANING AND PREPARATION
# =============================================================================

date_fields = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]

for field in date_fields:
    sold[field] = pd.to_datetime(sold[field])

sold_clean = sold.drop(columns=columns_to_drop)

sold_clean['invalid_close_price'] = sold_clean['ClosePrice'] <= 0
sold_clean['invalid_living_area'] = sold_clean['LivingArea'] <= 0
sold_clean['invalid_days_on_market'] = sold_clean['DaysOnMarket'] < 0
sold_clean['invalid_bedrooms'] = sold_clean['BedroomsTotal'] < 0
sold_clean['invalid_bathrooms'] = sold_clean['BathroomsTotalInteger'] < 0



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