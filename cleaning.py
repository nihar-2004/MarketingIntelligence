import pandas as pd
import numpy as np

def prepare_marketing_data(business_file, facebook_file, google_file, tiktok_file):
    """
    Loads, cleans, merges marketing and business data, and calculates key KPIs.

    Args:
        business_file (str): Filepath for business.csv.
        facebook_file (str): Filepath for Facebook.csv.
        google_file (str): Filepath for Google.csv.
        tiktok_file (str): Filepath for TikTok.csv.

    Returns:
        pandas.DataFrame: A single, cleaned DataFrame ready for analysis and visualization.
    """
    try:
        # --- 1. Load Data ---
        df_business = pd.read_csv(business_file)
        df_facebook = pd.read_csv(facebook_file)
        df_google = pd.read_csv(google_file)
        df_tiktok = pd.read_csv(tiktok_file)

        # --- 2. Prepare and Combine Marketing Data ---
        # Add a 'platform' column to each marketing dataframe
        df_facebook['platform'] = 'Facebook'
        df_google['platform'] = 'Google'
        df_tiktok['platform'] = 'TikTok'

        # Combine the three marketing dataframes into one
        df_marketing = pd.concat([df_facebook, df_google, df_tiktok], ignore_index=True)

        # --- 3. Clean and Standardize Data ---
        # Convert date columns to datetime objects for accurate merging
        df_business['date'] = pd.to_datetime(df_business['date'])
        df_marketing['date'] = pd.to_datetime(df_marketing['date'])

        # Clean up business column names for easier access
        df_business.columns = [
            'date', 'orders', 'new_orders', 'new_customers', 
            'total_revenue', 'gross_profit', 'cogs'
        ]

        # --- 4. Merge Marketing and Business Data ---
        # Merge the combined marketing data with the business data on the 'date' column
        df_merged = pd.merge(df_marketing, df_business, on='date', how='left')

        # --- 5. Calculate Key Performance Indicators (KPIs) ---
        # Handle potential division by zero by replacing 0 with NaN, then filling resulting NaNs
        
        # Return on Ad Spend (ROAS)
        df_merged['roas'] = (df_merged['attributed revenue'] / df_merged['spend']).replace([np.inf, -np.inf], 0).fillna(0)

        # Cost Per Click (CPC)
        df_merged['cpc'] = (df_merged['spend'] / df_merged['clicks']).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Click-Through Rate (CTR)
        df_merged['ctr'] = ((df_merged['clicks'] / df_merged['impression']) * 100).replace([np.inf, -np.inf], 0).fillna(0)

        # Customer Acquisition Cost (CAC)
        # First, calculate total marketing spend per day
        daily_spend = df_merged.groupby('date')['spend'].sum().reset_index()
        daily_spend.rename(columns={'spend': 'total_daily_spend'}, inplace=True)
        
        # Merge this daily total back into the main dataframe
        df_final = pd.merge(df_merged, daily_spend, on='date', how='left')

        # Now, calculate daily CAC
        df_final['cac'] = (df_final['total_daily_spend'] / df_final['new_customers']).replace([np.inf, -np.inf], 0).fillna(0)

        print("Data processing complete. Final DataFrame is ready.")
        return df_final

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all CSV files are in the correct path.")
        return None

# --- Example of how to use the function ---
if __name__ == '__main__':
    # Define file paths (make sure these files are in the same directory as the script)
    business_csv = 'business.csv'
    facebook_csv = 'Facebook.csv'
    google_csv = 'Google.csv'
    tiktok_csv = 'TikTok.csv'

    # Process the data
    final_df = prepare_marketing_data(business_csv, facebook_csv, google_csv, tiktok_csv)

    if final_df is not None:
        # Display the first 5 rows and some info about the final dataframe
        print("\n--- Final DataFrame Head ---")
        print(final_df.head())
        
        print("\n--- DataFrame Info ---")
        final_df.info()
        
        print("\n--- Newly Calculated KPI Columns ---")
        print(final_df[['date', 'platform', 'spend', 'roas', 'cpc', 'ctr', 'cac']].head())
        final_df.to_csv('final_marketing_data.csv', index=False)
