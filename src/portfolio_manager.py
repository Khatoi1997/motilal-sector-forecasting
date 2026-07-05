import pandas as pd
import numpy as np

class NRIPortfolioManager:
    def __init__(self):
        """
        Compliance and Tax-Optimization layer for Motilal Oswal NRI Client Portfolios.
        Filters raw model predictions through FEMA regulatory caps and capital gains tax rules.
        """
        # Hardcoded statutory boundaries for NRI asset allocation
        self.FEMA_LIMITS = {
            'Banking': 0.20,      # Aggregate NRI cap for public/private banking ceilings
            'Defense': 0.24,      # Standard automatic route cap
            'IT': 1.00,           # 100% automatic route allowed
            'Pharma': 0.74        # Brownfield pharma cap
        }
        
        # Indian Tax Rules (Standard Equity Rates)
        self.STCG_RATE = 0.20     # Short-Term Capital Gains Tax Rate
        self.LTCG_RATE = 0.125    # Long-Term Capital Gains Tax Rate

    def filter_compliance_caps(self, insights_df: pd.DataFrame, aggregate_nri_holdings: dict) -> pd.DataFrame:
        """
        Flags and blocks stock recommendations if the firm's aggregate NRI 
        holding approaches FEMA statutory limits.
        """
        df = insights_df.copy()
        compliance_statuses = []
        regulatory_notes = []
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            sector = row['sector']
            
            # Fetch current aggregate holding of the firm's NRI client pool
            current_holding = aggregate_nri_holdings.get(ticker, 0.0)
            allowed_limit = self.FEMA_LIMITS.get(sector, 0.24) # Default to 24% if unlisted
            
            # Leave a buffer of 2% before the hard statutory ceiling
            if current_holding >= (allowed_limit - 0.02):
                compliance_statuses.append("BLOCKED")
                regulatory_notes.append(f"FEMA Breach Risk: Sector {sector} holding at {current_holding*100}%. Limit is {allowed_limit*100}%.")
            else:
                compliance_statuses.append("APPROVED")
                regulatory_notes.append("FEMA Compliant")
                
        df['fema_status'] = compliance_statuses
        df['compliance_notes'] = regulatory_notes
        return df

    def optimize_tax_exit(self, compliance_df: pd.DataFrame, client_portfolio: dict) -> pd.DataFrame:
        """
        Evaluates whether selling an asset to lock in a model's prediction 
        makes sense after accounting for Short-Term vs Long-Term Capital Gains.
        """
        df = compliance_df.copy()
        action_signals = []
        tax_notes = []
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            predicted_return = row['predicted_next_day_return']
            fema_status = row['fema_status']
            
            # Skip evaluation if already blocked by FEMA regulatory constraints
            if fema_status == "BLOCKED":
                action_signals.append("IGNORE")
                tax_notes.append("Compliance Blocked")
                continue
                
            # If client owns the stock, evaluate tax impact of an immediate exit/rebalance
            if ticker in client_portfolio:
                holding_days = client_portfolio[ticker]['holding_days']
                unrealized_gain_pct = client_portfolio[ticker]['unrealized_gain_pct']
                
                # Determine tax bracket based on the 365-day holding boundary
                if holding_days < 365:
                    tax_hit = unrealized_gain_pct * self.STCG_RATE
                    bracket = "STCG (20%)"
                else:
                    tax_hit = unrealized_gain_pct * self.LTCG_RATE
                    bracket = "LTCG (12.5%)"
                    
                # If the immediate tax liability outweights the model's next-day alpha signal, veto the sell
                if tax_hit > predicted_return and predicted_return < 0:
                    action_signals.append("HOLD")
                    tax_notes.append(f"Tax Shield Active: Selling triggers {bracket} lock-in which outweighs model exit signal.")
                else:
                    action_signals.append("EXECUTE_TRADE")
                    tax_notes.append(f"Tax Optimized. Current bracket: {bracket}.")
            else:
                # Fresh allocation recommendation
                action_signals.append("BUY" if predicted_return > 0.005 else "NEUTRAL")
                tax_notes.append("Fresh Allocation Window")
                
        df['nri_action_signal'] = action_signals
        df['tax_insights'] = tax_notes
        return df