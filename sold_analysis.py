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

# Before row count
print(f"Row count before cleaning: {len(sold):,}")

# 1. Convert date fields to datetime format
# Date fields are read as text (object) by default — must convert for date math
date_fields = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
for field in date_fields:
    sold[field] = pd.to_datetime(sold[field])

print("\nDate field types after conversion:")
print(sold[['CloseDate', 'PurchaseContractDate',
            'ListingContractDate', 'ContractStatusChangeDate']].dtypes)

# 2. Remove unnecessary or redundant columns
# Dropping all columns flagged as >90% missing in Weeks 2-3
# These columns are empty across all records and carry no analytical value
missing_sold = pd.DataFrame({
    'missing_count': sold.isnull().sum(),
    'missing_pct': (sold.isnull().sum() / len(sold) * 100).round(2)
})
high_missing_sold = missing_sold[missing_sold['missing_pct'] > 90]
columns_to_drop = high_missing_sold.index.tolist()

sold_clean = sold.drop(columns=columns_to_drop)
print(f"\nShape before dropping high-missing columns: {sold.shape}")
print(f"Shape after dropping high-missing columns: {sold_clean.shape}")
print(f"Columns dropped: {columns_to_drop}")

# 3. Flag invalid numeric values
# Flagging rather than deleting to preserve raw records for reference
# ClosePrice <= 0: a home cannot sell for zero or negative dollars
sold_clean['invalid_close_price'] = sold_clean['ClosePrice'] <= 0
# LivingArea <= 0: a home cannot have zero or negative square footage
sold_clean['invalid_living_area'] = sold_clean['LivingArea'] <= 0
# DaysOnMarket < 0: a home cannot sell before it is listed
sold_clean['invalid_days_on_market'] = sold_clean['DaysOnMarket'] < 0
# Negative bedrooms/bathrooms: physically impossible
sold_clean['invalid_bedrooms'] = sold_clean['BedroomsTotal'] < 0
sold_clean['invalid_bathrooms'] = sold_clean['BathroomsTotalInteger'] < 0

print("\nInvalid numeric value counts:")
print(f"  ClosePrice <= 0:         {sold_clean['invalid_close_price'].sum():,}")
print(f"  LivingArea <= 0:         {sold_clean['invalid_living_area'].sum():,}")
print(f"  DaysOnMarket < 0:        {sold_clean['invalid_days_on_market'].sum():,}")
print(f"  Negative Bedrooms:       {sold_clean['invalid_bedrooms'].sum():,}")
print(f"  Negative Bathrooms:      {sold_clean['invalid_bathrooms'].sum():,}")

# 4. Date consistency checks
# ListingContractDate should precede PurchaseContractDate which should precede CloseDate
sold_clean['listing_after_close_flag'] = sold_clean['CloseDate'] < sold_clean['ListingContractDate']
sold_clean['purchase_after_close_flag'] = sold_clean['CloseDate'] < sold_clean['PurchaseContractDate']
sold_clean['negative_timeline_flag'] = sold_clean['PurchaseContractDate'] < sold_clean['ListingContractDate']

print("\nDate consistency flag counts:")
print(f"  listing_after_close_flag:   {sold_clean['listing_after_close_flag'].sum():,}")
print(f"  purchase_after_close_flag:  {sold_clean['purchase_after_close_flag'].sum():,}")
print(f"  negative_timeline_flag:     {sold_clean['negative_timeline_flag'].sum():,}")

# 5. Geographic data checks
# California bounding box: Latitude 32.5-42.0, Longitude -124.5 to -114.0
sold_clean['missing_coords_flag'] = sold_clean['Latitude'].isnull() | sold_clean['Longitude'].isnull()
sold_clean['zero_coords_flag'] = (sold_clean['Latitude'] == 0) | (sold_clean['Longitude'] == 0)
sold_clean['positive_longitude_flag'] = sold_clean['Longitude'] > 0
sold_clean['out_of_state_flag'] = (
    (sold_clean['Latitude'] < 32.5) | (sold_clean['Latitude'] > 42.0) |
    (sold_clean['Longitude'] < -124.5) | (sold_clean['Longitude'] > -114.0)
) & ~sold_clean['missing_coords_flag']

print("\nGeographic data quality summary:")
print(f"  Missing coordinates:        {sold_clean['missing_coords_flag'].sum():,}")
print(f"  Zero coordinates:           {sold_clean['zero_coords_flag'].sum():,}")
print(f"  Positive longitude (error): {sold_clean['positive_longitude_flag'].sum():,}")
print(f"  Out of state/implausible:   {sold_clean['out_of_state_flag'].sum():,}")

# 6. Full data quality summary
print("\nFULL DATA QUALITY SUMMARY:")
print(f"  Total records: {len(sold_clean):,}")
print(f"  Columns: {sold_clean.shape[1]}")
print(f"\n  -- Invalid Numeric Values --")
print(f"  ClosePrice <= 0:            {sold_clean['invalid_close_price'].sum():,}")
print(f"  LivingArea <= 0:            {sold_clean['invalid_living_area'].sum():,}")
print(f"  DaysOnMarket < 0:           {sold_clean['invalid_days_on_market'].sum():,}")
print(f"\n  -- Date Consistency --")
print(f"  listing_after_close_flag:   {sold_clean['listing_after_close_flag'].sum():,}")
print(f"  purchase_after_close_flag:  {sold_clean['purchase_after_close_flag'].sum():,}")
print(f"  negative_timeline_flag:     {sold_clean['negative_timeline_flag'].sum():,}")
print(f"\n  -- Geographic --")
print(f"  Missing coordinates:        {sold_clean['missing_coords_flag'].sum():,}")
print(f"  Zero coordinates:           {sold_clean['zero_coords_flag'].sum():,}")
print(f"  Positive longitude:         {sold_clean['positive_longitude_flag'].sum():,}")
print(f"  Out of state:               {sold_clean['out_of_state_flag'].sum():,}")

# After row count
print(f"\nRow count after cleaning: {len(sold_clean):,}")
print(f"Columns before: {sold.shape[1]} | Columns after: {sold_clean.shape[1]}")


# =============================================================================
# WEEK 6 — FEATURE ENGINEERING 
# =============================================================================
# =============================================================================

# Make explicit copy to avoid SettingWithCopyWarning
sold_clean = sold_clean.copy()

# 1. Price Ratio — measures negotiation strength
sold_clean['price_ratio'] = sold_clean['ClosePrice'] / sold_clean['ListPrice']

# 2. Close to Original List Ratio — captures full price reduction history
sold_clean['close_to_original_list_ratio'] = sold_clean['ClosePrice'] / sold_clean['OriginalListPrice']

# 3. Price Per Square Foot — normalizes price across different home sizes
sold_clean['price_per_sqft'] = sold_clean['ClosePrice'] / sold_clean['LivingArea']

# 4. Listing to Contract Days — time from listing to accepted offer
sold_clean['listing_to_contract_days'] = (
    sold_clean['PurchaseContractDate'] - sold_clean['ListingContractDate']
).dt.days

# 5. Contract to Close Days — escrow and closing period duration
sold_clean['contract_to_close_days'] = (
    sold_clean['CloseDate'] - sold_clean['PurchaseContractDate']
).dt.days

# 6. Time series variables derived from CloseDate
sold_clean['close_year'] = sold_clean['CloseDate'].dt.year
sold_clean['close_month'] = sold_clean['CloseDate'].dt.month
sold_clean['close_yrmo'] = sold_clean['CloseDate'].dt.to_period('M')

# Sample output table showing new columns populated
print("Sample of engineered metrics:")
print(sold_clean[['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
                   'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
                   'listing_to_contract_days', 'contract_to_close_days',
                   'close_year', 'close_month', 'close_yrmo']].head(5).to_string())

# ── SEGMENT ANALYSIS ──────────────────────────────────────────────────────────

# Summary by CountyOrParish
print("\nSummary by CountyOrParish (top 15 by transaction count):")
print(sold_clean.groupby('CountyOrParish').agg(
    transaction_count=('ClosePrice', 'count'),
    median_close_price=('ClosePrice', 'median'),
    median_price_per_sqft=('price_per_sqft', 'median'),
    median_days_on_market=('DaysOnMarket', 'median'),
    avg_price_ratio=('price_ratio', 'mean')
).sort_values('transaction_count', ascending=False).head(15).to_string())

# Summary by PropertySubType
print("\nSummary by PropertySubType:")
print(sold_clean.groupby('PropertySubType').agg(
    transaction_count=('ClosePrice', 'count'),
    median_close_price=('ClosePrice', 'median'),
    median_price_per_sqft=('price_per_sqft', 'median'),
    median_days_on_market=('DaysOnMarket', 'median'),
    avg_price_ratio=('price_ratio', 'mean')
).sort_values('transaction_count', ascending=False).to_string())

# Top 15 Listing Offices by transaction count
print("\nTop 15 Listing Offices by transaction count:")
print(sold_clean.groupby('ListOfficeName').agg(
    transaction_count=('ClosePrice', 'count'),
    total_volume=('ClosePrice', 'sum'),
    median_close_price=('ClosePrice', 'median')
).sort_values('transaction_count', ascending=False).head(15).to_string())

# Top 15 Buyer Offices by transaction count
print("\nTop 15 Buyer Offices by transaction count:")
print(sold_clean.groupby('BuyerOfficeName').agg(
    transaction_count=('ClosePrice', 'count'),
    total_volume=('ClosePrice', 'sum'),
    median_close_price=('ClosePrice', 'median')
).sort_values('transaction_count', ascending=False).head(15).to_string())

# =============================================================================
# WEEK 7 — OUTLIER DETECTION (coming soon)
# =============================================================================


# =============================================================================
# FINAL OUTPUT — Save clean CSV for Tableau
# =============================================================================

sold_clean.to_csv('data/02_intermediate/CRMLSSold_Cleaned.csv', index=False)
print(f"\nSaved CRMLSSold_Cleaned.csv — {len(sold_clean):,} rows, {sold_clean.shape[1]} columns")