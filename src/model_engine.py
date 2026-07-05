import os
import datetime
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

class SectorModelEngine:
    def __init__(self, features: list, target: str, sector_name: str = "Generic"):
        """
        Institutional Engine with separated plotting tracks and 
        nominal price-basis forecasting metrics.
        """
        self.features = features
        self.target = target
        self.sector_name = sector_name.lower()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        self.best_lgb_params = None
        self.best_xgb_params = None
        self.final_lgb_model = None
        self.final_xgb_model = None
        
        self.val_df_ref = None
        self.val_preds_ensemble = None
        
        os.makedirs("reports/plots", exist_ok=True)

    def _wmape_price_metric(self, y_true, y_pred) -> float:
        """Calculates stable WMAPE on price series."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    def train_and_select_champion(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 15):
        X_train, y_train = train_df[self.features], train_df[self.target]
        X_val, y_val = val_df[self.features], val_df[self.target]
        self.val_df_ref = val_df.copy()

        # --- LightGBM Optimization Pass ---
        default_lgb_params = {'objective': 'regression', 'metric': 'mae', 'verbose': -1, 'n_jobs': -1}
        base_lgb = lgb.train(default_lgb_params, lgb.Dataset(X_train, label=y_train), num_boost_round=100)
        
        def lgb_objective(trial):
            params = {
                'objective': 'regression', 'metric': 'mae', 'verbose': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'num_leaves': trial.suggest_int('num_leaves', 15, 31),
                'max_depth': trial.suggest_int('max_depth', 3, 5)
            }
            model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=100)
            return np.mean(np.abs(y_val - model.predict(X_val)))

        lgb_study = optuna.create_study(direction="minimize")
        lgb_study.optimize(lgb_objective, n_trials=n_trials)
        self.best_lgb_params = lgb_study.best_params if lgb_study.best_value < np.mean(np.abs(y_val - base_lgb.predict(X_val))) else default_lgb_params
        self.best_lgb_params.update({'objective': 'regression', 'metric': 'mae', 'verbose': -1})

        # --- XGBoost Optimization Pass ---
        default_xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'mae', 'nthread': -1}
        base_xgb = xgb.train(default_xgb_params, xgb.DMatrix(X_train, label=y_train), num_boost_round=100)

        def xgb_objective(trial):
            params = {
                'objective': 'reg:squarederror', 'eval_metric': 'mae',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'max_depth': trial.suggest_int('max_depth', 3, 5)
            }
            model = xgb.train(params, xgb.DMatrix(X_train, label=y_train), num_boost_round=100)
            return np.mean(np.abs(y_val - model.predict(xgb.DMatrix(X_val))))

        xgb_study = optuna.create_study(direction="minimize")
        xgb_study.optimize(xgb_objective, n_trials=n_trials)
        self.best_xgb_params = xgb_study.best_params if xgb_study.best_value < np.mean(np.abs(y_val - base_xgb.predict(xgb.DMatrix(X_val)))) else default_xgb_params
        self.best_xgb_params.update({'objective': 'reg:squarederror', 'eval_metric': 'mae'})

        # Cache predictions
        fit_lgb = lgb.train(self.best_lgb_params, lgb.Dataset(X_train, label=y_train), num_boost_round=150)
        fit_xgb = xgb.train(self.best_xgb_params, xgb.DMatrix(X_train, label=y_train), num_boost_round=150)
        self.val_preds_ensemble = (fit_lgb.predict(X_val) + fit_xgb.predict(xgb.DMatrix(X_val))) / 2.0

    def refit_on_full_history(self, full_historical_df: pd.DataFrame):
        X_full = full_historical_df[self.features]
        y_full = full_historical_df[self.target]

        self.final_lgb_model = lgb.train(self.best_lgb_params, lgb.Dataset(X_full, label=y_full), num_boost_round=200)
        self.final_xgb_model = xgb.train(self.best_xgb_params, xgb.DMatrix(X_full, label=y_full), num_boost_round=200)
        
        # Save independent charts separately
        self._generate_separated_training_plots(X_full)

    def _generate_separated_training_plots(self, X_full: pd.DataFrame):
        # 1. Independent Feature Importance Plot
        lgb_explainer = shap.TreeExplainer(self.final_lgb_model)
        xgb_explainer = shap.TreeExplainer(self.final_xgb_model)
        shap_blended = (lgb_explainer.shap_values(X_full) + xgb_explainer.shap_values(X_full)) / 2.0
        importance_series = pd.Series(np.mean(np.abs(shap_blended), axis=0), index=self.features).sort_values()

        plt.figure(figsize=(10, 6))
        plt.barh(importance_series.index, importance_series.values, color='#4A90E2', edgecolor='black')
        plt.title(f"Production Feature Importance Profile - {self.sector_name.upper()}")
        plt.tight_layout()
        plt.savefig(f"reports/plots/{self.sector_name}_lgb_importance_{self.timestamp}.png", dpi=150)
        plt.close()

        # 2. Independent 2025 Price Validation Performance Chart
        if self.val_df_ref is not None:
            plot_val = self.val_df_ref.copy()
            # Reconstruct nominal predicted price from return scalars
            plot_val['predicted_price'] = plot_val['close'] * (1 + self.val_preds_ensemble)
            
            timeline = plot_val.groupby('date')[['close', 'predicted_price']].mean().sort_index()
            
            plt.figure(figsize=(12, 5))
            plt.plot(timeline.index, timeline['close'], label='Actual Close Price', color='#222222', alpha=0.6)
            plt.plot(timeline.index, timeline['predicted_price'], label='Predicted Close Price', color='#D9381E', linestyle='--')
            
            every_nth = max(1, len(timeline) // 7)
            plt.xticks(timeline.index[::every_nth], rotation=15)
            
            val_wmape = self._wmape_price_metric(timeline['close'], timeline['predicted_price'])
            plt.title(f"2025 Gated Validation Price Tracking (WMAPE: {val_wmape*100:.2f}%) - {self.sector_name.upper()}")
            plt.ylabel("Nominal Price (INR)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"reports/plots/{self.sector_name}_validation_tracking_{self.timestamp}.png", dpi=150)
            plt.close()

    def generate_ensemble_insights(self, live_2026_df: pd.DataFrame) -> pd.DataFrame:
        X_live = live_2026_df[self.features]
        
        lgb_preds = self.final_lgb_model.predict(X_live)
        xgb_preds = self.final_xgb_model.predict(xgb.DMatrix(X_live))
        ensemble_return_preds = (lgb_preds + xgb_preds) / 2.0

        # Construct full historical outputs mapping return arrays back to exact prices
        insights_df = live_2026_df[['date', 'ticker', 'sector', 'close']].copy()
        insights_df['actual_price'] = insights_df['close']
        insights_df['predicted_price'] = insights_df['close'] * (1 + ensemble_return_preds)

        # 3. Independent Waterfall Plot for the Snapshot Row
        latest_live_snapshot = live_2026_df.groupby('ticker').last().reset_index()
        X_snapshot = latest_live_snapshot[self.features]
        lgb_explainer = shap.TreeExplainer(self.final_lgb_model)
        xgb_explainer = shap.TreeExplainer(self.final_xgb_model)
        
        shap_values = (lgb_explainer(X_snapshot).values + xgb_explainer(X_snapshot).values) / 2.0
        base_values = (lgb_explainer(X_snapshot).base_values + xgb_explainer(X_snapshot).base_values) / 2.0
        
        single_explanation = shap.Explanation(
            values=shap_values[0], base_values=base_values[0],
            data=X_snapshot.iloc[0].values, feature_names=self.features
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(single_explanation, show=False)
        plt.title(f"Live Attribution ({latest_live_snapshot.iloc[0]['ticker']})")
        plt.tight_layout()
        plt.savefig(f"reports/plots/{self.sector_name}_shap_summary_{self.timestamp}.png", dpi=150)
        plt.close()

        # 4. Independent 2026 Continuous Inference Evaluation Plot
        timeline_infer = insights_df.groupby('date')[['actual_price', 'predicted_price']].mean().sort_index()
        
        plt.figure(figsize=(12, 5))
        plt.plot(timeline_infer.index, timeline_infer['actual_price'], label='Actual Price', color='#222222', alpha=0.6)
        plt.plot(timeline_infer.index, timeline_infer['predicted_price'], label='Ensemble Forecasted Price', color='#008080')
        
        every_nth = max(1, len(timeline_infer) // 7)
        plt.xticks(timeline_infer.index[::every_nth], rotation=15)
        
        infer_wmape = self._wmape_price_metric(timeline_infer['actual_price'], timeline_infer['predicted_price'])
        plt.title(f"Live 2026 Inference Price Evaluation Tracking (Price WMAPE: {infer_wmape*100:.2f}%)")
        plt.ylabel("Nominal Price (INR)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"reports/plots/{self.sector_name}_inference_tracking_{self.timestamp}.png", dpi=150)
        plt.close()

        return insights_df