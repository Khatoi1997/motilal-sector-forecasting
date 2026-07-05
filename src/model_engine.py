import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import shap

class SectorModelEngine:
    def __init__(self, features: list, target: str):
        """
        Ensemble Engine running LightGBM + XGBoost side-by-side.
        Optimized for memory-efficient blending on single-node hardware.
        """
        self.features = features
        self.target = target
        self.lgb_model = None
        self.xgb_model = None

    def train_sector_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains both LightGBM and XGBoost models for a specific sector.
        """
        X_train, y_train = train_df[self.features], train_df[self.target]
        X_val, y_val = val_df[self.features], val_df[self.target]

        # --- 1. Train LightGBM ---
        dtrain_lgb = lgb.Dataset(X_train, label=y_train)
        dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtrain_lgb)
        
        lgb_params = {
            'objective': 'regression', 'metric': 'mape', 'boosting_type': 'gbdt',
            'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'verbose': -1, 'n_jobs': -1
        }
        self.lgb_model = lgb.train(
            lgb_params, dtrain_lgb, num_boost_round=500,
            valid_sets=[dval_lgb], callbacks=[lgb.early_stopping(30, verbose=False)]
        )

        # --- 2. Train XGBoost ---
        # Convert to native DMatrix for extreme speed and low memory
        dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
        dval_xgb = xgb.DMatrix(X_val, label=y_val)
        
        xgb_params = {
            'objective': 'reg:squarederror', 'eval_metric': 'mape',
            'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'nthread': -1
        }
        
        # XGBoost early stopping requires an explicit evaluation list
        evallist = [(dval_xgb, 'eval')]
        self.xgb_model = xgb.train(
            xgb_params, dtrain_xgb, num_boost_round=500,
            evals=evallist, early_stopping_rounds=30, verbose_eval=False
        )

    def generate_ensemble_insights(self, latest_market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Blends LightGBM and XGBoost predictions and uses stable multi-model
        TreeSHAP calculations to flag primary signal drivers for advisors.
        """
        if self.lgb_model is None or self.xgb_model is None:
            raise ValueError("Both models must be trained before generating insights.")

        X_latest = latest_market_data[self.features]
        dlatest_xgb = xgb.DMatrix(X_latest)

        # 1. Generate blended predictions (50/50 Blend)
        lgb_preds = self.lgb_model.predict(X_latest)
        xgb_preds = self.xgb_model.predict(dlatest_xgb)
        ensemble_preds = (lgb_preds + xgb_preds) / 2.0

        insights_df = latest_market_data[['date', 'ticker', 'sector', 'close']].copy()
        insights_df['predicted_next_day_return'] = ensemble_preds

        # 2. Extract blended feature drivers using TreeSHAP
        lgb_explainer = shap.TreeExplainer(self.lgb_model)
        xgb_explainer = shap.TreeExplainer(self.xgb_model)
        
        # Average the SHAP impact matrices across both model perspectives
        shap_blended = (lgb_explainer.shap_values(X_latest) + xgb_explainer.shap_values(X_latest)) / 2.0

        top_feature_1 = []
        top_feature_2 = []
        
        for i in range(len(latest_market_data)):
            row_shap = shap_blended[i]
            sorted_indices = np.argsort(np.abs(row_shap))[::-1]
            top_feature_1.append(self.features[sorted_indices[0]])
            top_feature_2.append(self.features[sorted_indices[1]])
            
        insights_df['primary_driver'] = top_feature_1
        insights_df['secondary_driver'] = top_feature_2
        
        return insights_df