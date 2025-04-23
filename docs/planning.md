# Planning document


## Structure

### 1. Clean BACI
1. Load incrememntally and convert from CSV into a Parquet
2. Remap country codes. Explode any country codes which refer to regions (only identified EFTA right now)

### 2. Clean WITS MFN 
1. Load into a single polars df. Apply predefined schema.
2. Translate HS codes into HS0
3. Remap country codes 
4. Rename columns

### 3. Clean WITS PREF
1. Load into a single polars df. Apply predefined schema.
2. Translate HS codes into HS0
3. Explode preferential tariff groups into individual country codes
4. Remap country codes
5. Rename columns

### 4. Join into unified dataset
1. Going year-by-year, join the WITS tariffs into the BACI, forming one unified table.

### 5. Validate
1. Check that the joined dataset looks and act as it should
