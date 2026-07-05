import pandas as pd
import numpy as np

class SectorDataEngine:
    def __init__(self):
        """
        Engine to handle memory-conscious data processing with macro-fundamental overlays
        optimized for single-node execution.
        """
        pass

    def downcast_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimizes numeric data types to reduce memory footprint."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                c_min, c_max = df[col].min(), df[col].max()
                if pd.api.types.is_integer_dtype(df[col]):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                elif pd.api.types.is_float_dtype(df[col]):
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates core technical signals sequentially per stock."""
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        
        # SMAs
        df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=10).mean())
        df['sma_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=50).mean())
        
        # Fast Vectorized RSI
        delta = df.groupby('ticker')['close'].diff()
        gain = (delta.where(delta > 0, 0)).groupby(df['ticker'])
        loss = (-delta.where(delta < 0, 0)).groupby(df['ticker'])
        avg_gain = gain.transform(lambda x: x.rolling(window=14).mean())
        avg_loss = loss.transform(lambda x: x.rolling(window=14).mean())
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Target Variable (Next day forward return)
        df['target_return'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1
        return df

    def enrich_macro_layers(self, stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges multi-sector stock sequences with macroscopic fundamentals.
        Uses forward-filling to handle mismatched dates (e.g., policy updates).
        """
        # Ensure date sorting to guarantee structural alignment
        stock_df = stock_df.sort_values(by=['date']).reset_index(drop=True)
        macro_df = macro_df.sort_values(by=['date']).reset_index(drop=True)
        
        # Left join macro variables onto stock rows matching on date
        enriched_df = pd.merge(stock_df, macro_df, on='date', how='left')
        
        # Forward fill macroeconomic attributes if market days don't align with policy logs
        enriched_df[['repo_rate', 'usd_inr']] = enriched_df.groupby('ticker')[['repo_rate', 'usd_inr']].ffill()
        
        # Engineer derivative macro-momentum features
        enriched_df['usd_inr_momentum'] = enriched_df.groupby('ticker')['usd_inr'].transform(lambda x: x.pct_change(periods=5))
        
        return enriched_df

    def get_sector_data(self, complete_df: pd.DataFrame, macro_df: pd.DataFrame, sector_name: str) -> pd.DataFrame:
        """Extracts, updates, blends, and downcasts an isolated sector partition."""
        sector_df = complete_df[complete_df['sector'] == sector_name].copy()
        sector_df = self.calculate_technical_indicators(sector_df)
        sector_df = self.enrich_macro_layers(sector_df, macro_df)
        
        return self.downcast_memory(sector_df.dropna()).reset_index(drop=True)