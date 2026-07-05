import pandas as pd
import numpy as np

class SectorDataEngine:
    def __init__(self, raw_data_path: str = None):
        """
        Engine to handle memory-conscious data processing for 500+ stocks
        optimized for single-node execution.
        """
        self.raw_data_path = raw_data_path

    def downcast_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes data types to reduce memory footprint up to 70%.
        """
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates vectorized technical indicators per stock using pandas/numpy.
        Ensures execution happens purely on fast C-level arrays.
        """
        # Ensure data is sorted sequentially per stock
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        
        # 1. Moving Averages
        df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=10).mean())
        df['sma_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=50).mean())
        
        # 2. Relative Strength Index (RSI - Fast Vectorized implementation)
        delta = df.groupby('ticker')['close'].diff()
        gain = (delta.where(delta > 0, 0)).groupby(df['ticker'])
        loss = (-delta.where(delta < 0, 0)).groupby(df['ticker'])
        
        avg_gain = gain.transform(lambda x: x.rolling(window=14).mean())
        avg_loss = loss.transform(lambda x: x.rolling(window=14).mean())
        
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 3. Target Variable: Next-day forward return (What we want to predict)
        df['target_return'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1
        
        # Drop boundary NaNs caused by indicators and the final row's target shift
        return df.dropna().reset_index(drop=True)

    def get_sector_data(self, complete_df: pd.DataFrame, sector_name: str) -> pd.DataFrame:
        """
        Filters out a single sector partition. 
        This keeps our training loop isolated and light on RAM.
        """
        sector_df = complete_df[complete_df['sector'] == sector_name].copy()
        sector_df = self.calculate_technical_indicators(sector_df)
        return self.downcast_memory(sector_df)