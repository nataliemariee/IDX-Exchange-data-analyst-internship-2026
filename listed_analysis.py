import pandas as pd
import requests
import io

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

# Remove duplicate columns caused by duplicate field names in extraction script
duplicate_cols = [col for col in listings.columns if '.1' in col]
listings = listings.drop(columns=duplicate_cols)
print(f"Removed {len(duplicate_cols)} duplicate columns from listings")

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
print("\nNumeric distribution summary (ListPrice, LivingArea, DaysOnMarket):")
print(listings[['ListPrice', 'LivingArea', 'DaysOnMarket']]
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

# # Step 3 – Create a matching year_month key on the listings dataset
# # listings dataset — key off CloseDate
# listings["year_month"] = pd.to_datetime(listings["CloseDate"]).dt.to_period("M")

# # Step 4 – Merge
# listing_with_rates = listings.merge(mortgage_monthly, on="year_month", how="left")

# # Step 5 – Validate the merge
# # Check for any unmatched rows (rate should not be null)
# print(listing_with_rates["rate_30yr_fixed"].isnull().sum())
# =============================================================================
# WEEKS 4-5 — DATA CLEANING AND PREPARATION 
# =============================================================================


# Before row count
print(f"Row count before cleaning: {len(listings):,}")

# 1. Convert date fields to datetime format
# Date fields are read as text (object) by default — must convert for date math
date_fields = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
for field in date_fields:
    listings[field] = pd.to_datetime(listings[field])

print("\nDate field types after conversion:")
print(listings[['CloseDate', 'PurchaseContractDate',
            'ListingContractDate', 'ContractStatusChangeDate']].dtypes)

# 2. Remove unnecessary or redundant columns
# Dropping all columns flagged as >90% missing in Weeks 2-3
# These columns are empty across all records and carry no analytical value
missing_listings = pd.DataFrame({
    'missing_count': listings.isnull().sum(),
    'missing_pct': (listings.isnull().sum() / len(listings) * 100).round(2)
})
high_missing_listings = missing_listings[missing_listings['missing_pct'] > 90]
columns_to_drop = high_missing_listings.index.tolist()

listings_clean = listings.drop(columns=columns_to_drop)
print(f"\nShape before dropping high-missing columns: {listings.shape}")
print(f"Shape after dropping high-missing columns: {listings_clean.shape}")
print(f"Columns dropped: {columns_to_drop}")

# 3. Flag invalid numeric values
# Flagging rather than deleting to preserve raw records for reference
# ListPrice <= 0: a home cannot sell for zero or negative dollars
listings_clean['invalid_close_price'] = listings_clean['ListPrice'] <= 0
# LivingArea <= 0: a home cannot have zero or negative square footage
listings_clean['invalid_living_area'] = listings_clean['LivingArea'] <= 0
# DaysOnMarket < 0: a home cannot sell before it is listed
listings_clean['invalid_days_on_market'] = listings_clean['DaysOnMarket'] < 0
# Negative bedrooms/bathrooms: physically impossible
listings_clean['invalid_bedrooms'] = listings_clean['BedroomsTotal'] < 0
listings_clean['invalid_bathrooms'] = listings_clean['BathroomsTotalInteger'] < 0

print("\nInvalid numeric value counts:")
print(f"  ListPrice <= 0:         {listings_clean['invalid_close_price'].sum():,}")
print(f"  LivingArea <= 0:         {listings_clean['invalid_living_area'].sum():,}")
print(f"  DaysOnMarket < 0:        {listings_clean['invalid_days_on_market'].sum():,}")
print(f"  Negative Bedrooms:       {listings_clean['invalid_bedrooms'].sum():,}")
print(f"  Negative Bathrooms:      {listings_clean['invalid_bathrooms'].sum():,}")

# 4. Date consistency checks
# ListingContractDate should precede PurchaseContractDate which should precede CloseDate
listings_clean['listing_after_close_flag'] = listings_clean['CloseDate'] < listings_clean['ListingContractDate']
listings_clean['purchase_after_close_flag'] = listings_clean['CloseDate'] < listings_clean['PurchaseContractDate']
listings_clean['negative_timeline_flag'] = listings_clean['PurchaseContractDate'] < listings_clean['ListingContractDate']

print("\nDate consistency flag counts:")
print(f"  listing_after_close_flag:   {listings_clean['listing_after_close_flag'].sum():,}")
print(f"  purchase_after_close_flag:  {listings_clean['purchase_after_close_flag'].sum():,}")
print(f"  negative_timeline_flag:     {listings_clean['negative_timeline_flag'].sum():,}")

# 5. Geographic data checks
# California bounding box: Latitude 32.5-42.0, Longitude -124.5 to -114.0
listings_clean['missing_coords_flag'] = listings_clean['Latitude'].isnull() | listings_clean['Longitude'].isnull()
listings_clean['zero_coords_flag'] = (listings_clean['Latitude'] == 0) | (listings_clean['Longitude'] == 0)
listings_clean['positive_longitude_flag'] = listings_clean['Longitude'] > 0
listings_clean['out_of_state_flag'] = (
    (listings_clean['Latitude'] < 32.5) | (listings_clean['Latitude'] > 42.0) |
    (listings_clean['Longitude'] < -124.5) | (listings_clean['Longitude'] > -114.0)
) & ~listings_clean['missing_coords_flag']

print("\nGeographic data quality summary:")
print(f"  Missing coordinates:        {listings_clean['missing_coords_flag'].sum():,}")
print(f"  Zero coordinates:           {listings_clean['zero_coords_flag'].sum():,}")
print(f"  Positive longitude (error): {listings_clean['positive_longitude_flag'].sum():,}")
print(f"  Out of state/implausible:   {listings_clean['out_of_state_flag'].sum():,}")

# 6. Full data quality summary
print("\nFULL DATA QUALITY SUMMARY:")
print(f"  Total records: {len(listings_clean):,}")
print(f"  Columns: {listings_clean.shape[1]}")
print(f"\n  -- Invalid Numeric Values --")
print(f"  ListPrice <= 0:            {listings_clean['invalid_close_price'].sum():,}")
print(f"  LivingArea <= 0:            {listings_clean['invalid_living_area'].sum():,}")
print(f"  DaysOnMarket < 0:           {listings_clean['invalid_days_on_market'].sum():,}")
print(f"\n  -- Date Consistency --")
print(f"  listing_after_close_flag:   {listings_clean['listing_after_close_flag'].sum():,}")
print(f"  purchase_after_close_flag:  {listings_clean['purchase_after_close_flag'].sum():,}")
print(f"  negative_timeline_flag:     {listings_clean['negative_timeline_flag'].sum():,}")
print(f"\n  -- Geographic --")
print(f"  Missing coordinates:        {listings_clean['missing_coords_flag'].sum():,}")
print(f"  Zero coordinates:           {listings_clean['zero_coords_flag'].sum():,}")
print(f"  Positive longitude:         {listings_clean['positive_longitude_flag'].sum():,}")
print(f"  Out of state:               {listings_clean['out_of_state_flag'].sum():,}")

# After row count
print(f"\nRow count after cleaning: {len(listings_clean):,}")
print(f"Columns before: {listings.shape[1]} | Columns after: {listings_clean.shape[1]}")


# =============================================================================
# WEEK 6 — FEATURE ENGINEERING AND MARKET METRICS
# =============================================================================

# Make explicit copy to avoid SettingWithCopyWarning
listings_clean = listings_clean.copy()

# 1. Price Reduction Ratio — list price vs original list price
listings_clean['price_reduction_ratio'] = listings_clean['ListPrice'] / listings_clean['OriginalListPrice']

# 2. Price Per Square Foot — normalizes price across different home sizes
listings_clean['price_per_sqft'] = listings_clean['ListPrice'] / listings_clean['LivingArea']

# 3. Listing to Contract Days — time from listing to accepted offer
listings_clean['listing_to_contract_days'] = (
    listings_clean['PurchaseContractDate'] - listings_clean['ListingContractDate']
).dt.days

# 4. Contract to Close Days — escrow and closing period duration
listings_clean['contract_to_close_days'] = (
    listings_clean['CloseDate'] - listings_clean['PurchaseContractDate']
).dt.days

# 5. Time series variables derived from ListingContractDate
listings_clean['listing_year'] = listings_clean['ListingContractDate'].dt.year
listings_clean['listing_month'] = listings_clean['ListingContractDate'].dt.month
listings_clean['listing_yrmo'] = listings_clean['ListingContractDate'].dt.to_period('M')

# Sample output table showing new columns populated
print("Sample of engineered metrics:")
print(listings_clean[['ListPrice', 'OriginalListPrice', 'LivingArea',
                       'price_reduction_ratio', 'price_per_sqft',
                       'listing_to_contract_days', 'contract_to_close_days',
                       'listing_year', 'listing_month', 'listing_yrmo']].head(5).to_string())

# ── SEGMENT ANALYSIS ──────────────────────────────────────────────────────────

# Summary by CountyOrParish
print("\nSummary by CountyOrParish (top 15 by listing count):")
print(listings_clean.groupby('CountyOrParish').agg(
    listing_count=('ListPrice', 'count'),
    median_list_price=('ListPrice', 'median'),
    median_price_per_sqft=('price_per_sqft', 'median'),
    median_days_on_market=('DaysOnMarket', 'median'),
    avg_price_reduction_ratio=('price_reduction_ratio', 'mean')
).sort_values('listing_count', ascending=False).head(15).to_string())

# Summary by PropertySubType
print("\nSummary by PropertySubType:")
print(listings_clean.groupby('PropertySubType').agg(
    listing_count=('ListPrice', 'count'),
    median_list_price=('ListPrice', 'median'),
    median_price_per_sqft=('price_per_sqft', 'median'),
    median_days_on_market=('DaysOnMarket', 'median'),
    avg_price_reduction_ratio=('price_reduction_ratio', 'mean')
).sort_values('listing_count', ascending=False).to_string())

# Top 15 Listing Offices by listing count
print("\nTop 15 Listing Offices by listing count:")
print(listings_clean.groupby('ListOfficeName').agg(
    listing_count=('ListPrice', 'count'),
    total_volume=('ListPrice', 'sum'),
    median_list_price=('ListPrice', 'median')
).sort_values('listing_count', ascending=False).head(15).to_string())

# =============================================================================
# WEEK 7 — OUTLIER DETECTION (coming soon)
# =============================================================================


# =============================================================================
# FINAL OUTPUT — Save clean CSV for Tableau
# =============================================================================

listings_clean.to_csv('data/02_intermediate/CRMLSListing_Cleaned.csv', index=False)
print(f"\nSaved CRMLSListing_Cleaned.csv — {len(listings_clean):,} rows, {listings_clean.shape[1]} columns")