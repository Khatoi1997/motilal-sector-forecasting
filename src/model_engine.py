import lightgbm as lgb
import pandas as pd
import numpy as np
import shap

class SectorModelEngine:
    def __init__(self, features: list, target: str):
        """
        Engine to handle LightGBM training and advisor-focused SHAP explanations.
        Optimized to prevent Out-Of-Memory crashes on Databricks Free Tier.
        """
        self.features = features
        self.target = target
        self.model = None
        self.explainer = None

    def train_sector_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> lgb.Booster:
        """
        Trains a highly efficient LightGBM model for a specific sector.
        """
        # Convert dataframes into highly compressed native LightGBM Datasets
        dtrain = lgb.Dataset(train_df[self.features], label=train_df[self.target])
        dval = lgb.Dataset(val_df[self.features], label=val_df[self.target], reference=dtrain)
        
        # Lightweight parameters optimized for single-node CPU execution
        params = {
            'objective': 'regression',
            'metric': 'mape',  # Mean Absolute Percentage Error for tracking precision
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1  # Utilize all cores on the single Databricks node
        }
        
        # Train model with early stopping to prevent over-fitting
        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        return self.model

    def generate_advisor_insights(self, latest_market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates next-day predictions and extracts highly targeted TreeSHAP 
        explanations for the top driving factors to assist sectoral advisors.
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating insights.")
            
        # 1. Predict next-day target returns
        preds = self.model.predict(latest_market_data[self.features])
        insights_df = latest_market_data[['date', 'ticker', 'sector', 'close']].copy()
        insights_df['predicted_next_day_return'] = preds
        
        # 2. Compute TreeSHAP Explanations (Optimized for quick execution)
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
            
        shap_values = self.explainer.shap_values(latest_market_data[self.features])
        
        # Extract the top two driving features for each individual stock prediction
        top_feature_1 = []
        top_feature_2 = []
        
        for i in range(len(latest_market_data)):
            # Sort features by absolute contribution to the specific prediction
            row_shap = shap_values[i]
            sorted_indices = np.argsort(np.abs(row_shap))[::-1]
            
            top_feature_1.append(self.features[sorted_indices[0]])
            top_feature_2.append(self.features[sorted_indices[1]])
            
        insights_df['primary_driver'] = top_feature_1
        insights_df['secondary_driver'] = top_feature_2
        
        return insights_df