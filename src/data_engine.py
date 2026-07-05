import pandas as pd
import numpy as np

class SectorDataEngine:
    def __init__(self):
        """
        Engine with an expanded 20-feature technical and macro matrix 
        engineered via fast vectorized calculations.
        """
        pass

    def downcast_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimizes numeric data types to reduce memory footprint up to 70%."""
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
        """Calculates scaled, stationary technical features to fix WMAPE variance dampening."""
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        groupby_ticker = df.groupby('ticker')
        
        # Scale prices relative to rolling baselines
        df['sma_10'] = groupby_ticker['close'].transform(lambda x: x.rolling(10).mean()) / df['close'] - 1
        df['sma_50'] = groupby_ticker['close'].transform(lambda x: x.rolling(50).mean()) / df['close'] - 1
        df['sma_20'] = groupby_ticker['close'].transform(lambda x: x.rolling(20).mean()) / df['close'] - 1
        
        df['ema_12'] = groupby_ticker['close'].transform(lambda x: x.ewm(span=12).mean()) / df['close'] - 1
        df['ema_26'] = groupby_ticker['close'].transform(lambda x: x.ewm(span=26).mean()) / df['close'] - 1
        
        # MACD needs to be scaled relative to the asset price
        df['macd'] = (groupby_ticker['close'].transform(lambda x: x.ewm(span=12).mean() - x.ewm(span=26).mean())) / df['close']
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatility features
        rolling_std_20 = groupby_ticker['close'].transform(lambda x: x.rolling(window=20).std())
        df['bollinger_high'] = (df['close'] + (2 * rolling_std_20)) / df['close'] - 1
        df['bollinger_low'] = (df['close'] - (2 * rolling_std_20)) / df['close'] - 1
        df['bollinger_width'] = rolling_std_20 * 4 / (df['close'] + 1e-9)
        
        # ATR scaled by close price
        high_low = df['high'] - df['low']
        high_cp = (df['high'] - groupby_ticker['close'].shift(1)).abs()
        low_cp = (df['low'] - groupby_ticker['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr_14'] = tr.groupby(df['ticker']).transform(lambda x: x.rolling(14).mean()) / df['close']
        
        df['daily_return'] = groupby_ticker['close'].transform(lambda x: x.pct_change())
        df['daily_return_volatility'] = groupby_ticker['daily_return'].transform(lambda x: x.rolling(10).std())
        
        # Bounded Oscillators (Already naturally scaled between 0 and 100 or -100 and 0)
        low_14 = groupby_ticker['low'].transform(lambda x: x.rolling(14).min())
        high_14 = groupby_ticker['high'].transform(lambda x: x.rolling(14).max())
        df['stochastic_k'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-9))
        df['stochastic_d'] = df.groupby('ticker')['stochastic_k'].transform(lambda x: x.rolling(3).mean())
        df['williams_r'] = ((high_14 - df['close']) / (high_14 - low_14 + 1e-9))
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.groupby(df['ticker']).transform(lambda x: x.rolling(20).mean())
        mad_tp = tp.groupby(df['ticker']).transform(lambda x: x.rolling(20).apply(lambda y: np.abs(y - y.mean()).mean(), raw=True))
        df['cci_20'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-9) / 100.0
        
        df['rate_of_change_10'] = groupby_ticker['close'].transform(lambda x: x.pct_change(10))
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.groupby(df['ticker']).transform(lambda x: x.rolling(14).mean())
        avg_loss = loss.groupby(df['ticker']).transform(lambda x: x.rolling(14).mean())
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi_14'] = (100 - (100 / (1 + rs))) / 100.0
        
        # Target Return
        df['target_return'] = groupby_ticker['close'].shift(-1) / df['close'] - 1
        
        # Clean up price columns to prevent scale leakage (except close, open, high, low)
        return df

    def enrich_macro_layers(self, stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Combines stock rows with advanced macro features."""
        stock_df = stock_df.sort_values(by=['date']).reset_index(drop=True)
        macro_df = macro_df.sort_values(by=['date']).reset_index(drop=True)
        
        enriched_df = pd.merge(stock_df, macro_df, on='date', how='left')
        enriched_df[['repo_rate', 'usd_inr']] = enriched_df.groupby('ticker')[['repo_rate', 'usd_inr']].ffill()
        
        # --- 5. Advanced Macro Variables ---
        enriched_df['usd_inr_momentum'] = enriched_df.groupby('ticker')['usd_inr'].transform(lambda x: x.pct_change(periods=5))
        enriched_df['repo_rate_change'] = enriched_df.groupby('ticker')['repo_rate'].transform(lambda x: x.diff(periods=5))
        enriched_df['usd_inr_sma_10'] = enriched_df.groupby('ticker')['usd_inr'].transform(lambda x: x.rolling(window=10).mean())
        enriched_df['usd_inr_deviation'] = enriched_df['usd_inr'] - enriched_df['usd_inr_sma_10']
        
        return enriched_df

    def get_sector_data(self, complete_df: pd.DataFrame, macro_df: pd.DataFrame, sector_name: str) -> pd.DataFrame:
        sector_df = complete_df[complete_df['sector'] == sector_name].copy()
        sector_df = self.calculate_technical_indicators(sector_df)
        sector_df = self.enrich_macro_layers(sector_df, macro_df)
        
        # Drop columns used for intermediate step transformations to keep memory safe
        drop_cols = ['daily_return']
        sector_df = sector_df.drop(columns=[c for c in drop_cols if c in sector_df.columns])
        
        return self.downcast_memory(sector_df.dropna()).reset_index(drop=True)