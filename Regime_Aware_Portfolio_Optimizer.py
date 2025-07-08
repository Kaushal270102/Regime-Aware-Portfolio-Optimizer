# =====================================================
# 🏦 ENHANCED REGIME-AWARE PORTFOLIO OPTIMIZER
# =====================================================
# Advanced framework for regime detection, Monte Carlo simulation,
# and multi-objective portfolio optimization with comprehensive risk management

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import norm, rankdata, jarque_bera, kstest
from scipy.stats import t 
import cvxpy as cp
from scipy.optimize import minimize
import itertools
import time
import warnings
import io
import base64
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal

# Configure logging for better error reporting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

# Configure Streamlit
st.set_page_config(
    page_title="Regime-Aware Portfolio Optimizer",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .risk-warning {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #00cc44;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SECTION 1: 📊 Advanced Data Management
# =====================================================

class DataManager:
    """Enhanced data management with multiple data sources and validation"""
    
    def __init__(self):
        self.supported_exchanges = {
            'NSE': '.NS',
            'BSE': '.BO',
            'NASDAQ': '',
            'NYSE': '',
            'LSE': '.L',
            'TSE': '.T'
        }
        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.valid_symbols = []
        
    def get_asset_universe(self, exchange: str) -> List[str]:
        """Get predefined asset universe for different exchanges"""
        universes = {
            'NSE': ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ICICIBANK.NS', 
                   'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS', 'WIPRO.NS'],
            'NASDAQ': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA'],
            'Global_ETFs': ['SPY', 'EFA', 'EEM', 'VNQ', 'GLD', 'TLT', 'VIX'],
            'Crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'BNB-USD']
        }
        return universes.get(exchange, [])
    
    def fetch_single_ticker(self, symbol, start_date, end_date, retries=2):
        """Fetch data for a single ticker with better error handling"""
        for attempt in range(retries):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if hist.empty or len(hist) < 100:
                    if attempt == 0:
                        st.warning(f"⚠️ {symbol}: Insufficient data, retrying...")
                        time.sleep(1)
                        continue
                    else:
                        st.error(f"❌ {symbol}: Failed - insufficient data ({len(hist)} days)")
                        logging.warning(f"Insufficient data for {symbol}: {len(hist)} days")
                        return None
                
                # Clean the data
                hist = hist.dropna()
                if len(hist) < 100:
                    st.error(f"❌ {symbol}: Too few valid data points after cleaning ({len(hist)} days)")
                    logging.warning(f"Too few valid data points for {symbol}: {len(hist)} days")
                    return None
                
                st.success(f"✅ {symbol}: {len(hist)} days loaded")
                logging.info(f"Loaded {len(hist)} days for {symbol}")
                return hist['Close']
                
            except Exception as e:
                if attempt == 0:
                    st.warning(f"⚠️ {symbol}: Error ({str(e)[:50]}), retrying...")
                    logging.warning(f"Error fetching {symbol}: {str(e)[:50]}, retrying...")
                    time.sleep(1)
                else:
                    st.error(f"❌ {symbol}: Failed after retries - {str(e)[:50]}")
                    logging.error(f"Failed to fetch {symbol} after retries: {str(e)}", exc_info=True)
                    return None
        
        return None
    
    def load_data_with_validation(self, tickers: List[str], start_date: str, 
                                end_date: str = None) -> pd.DataFrame:
        """Load and validate price data with comprehensive error handling"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        st.info("🔄 Starting data collection...")
        
        # Try to fetch each symbol
        price_data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(tickers):
            status_text.text(f"Loading {symbol}... ({i+1}/{len(tickers)})")
            data = self.fetch_single_ticker(symbol, start_date, end_date)
            if data is not None:
                price_data[symbol] = data
                self.valid_symbols.append(symbol)
            progress_bar.progress((i + 1) / len(tickers))
            time.sleep(0.5)  # Prevent rate limiting
        
        # If too few symbols loaded, try fallbacks
        fallback_tickers = ['^NSEI', '^BSESN']
        if len(price_data) < 2:
            st.warning("🔄 Too few symbols loaded, trying fallbacks...")
            logging.warning("Too few symbols loaded, attempting fallbacks")
            for symbol in fallback_tickers:
                if symbol not in price_data:
                    data = self.fetch_single_ticker(symbol, start_date, end_date)
                    if data is not None:
                        price_data[symbol] = data
                        self.valid_symbols.append(symbol)
                        if len(price_data) >= 2:
                            break
        
        progress_bar.empty()
        status_text.empty()
        
        if len(price_data) < 2:
            st.error("❌ Failed to load sufficient data even with fallbacks")
            logging.error("Failed to load sufficient data even with fallbacks")
            return pd.DataFrame()
        
        # Create aligned price dataframe
        prices_df = pd.DataFrame(price_data)
        
        # Handle missing values
        initial_rows = len(prices_df)
        prices_df = prices_df.dropna(how='all')  # Drop days with no data
        prices_df = prices_df.fillna(method='ffill', limit=5).fillna(method='bfill')
        
        # ✅ Drop columns only if more than 20% of data is still missing after forward-fill
        threshold = int(0.8 * len(prices_df))
        prices_df = prices_df.dropna(axis=1, thresh=threshold)
        
        if prices_df.empty:
            st.error("❌ No valid price data after alignment")
            logging.error("No valid price data after alignment")
            return pd.DataFrame()
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Clean outliers (more than 5 standard deviations)
        for col in returns_df.columns:
            mean_ret = returns_df[col].mean()
            std_ret = returns_df[col].std()
            if std_ret > 0:
                lower_bound = mean_ret - 5 * std_ret
                upper_bound = mean_ret + 5 * std_ret
                outliers = (returns_df[col] < lower_bound) | (returns_df[col] > upper_bound)
                if outliers.sum() > 0:
                    returns_df[col] = returns_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        self.prices = prices_df
        self.returns = returns_df
        
        st.success(f"✅ Data loaded successfully: {len(prices_df.columns)} assets, {len(prices_df)} days")
        logging.info(f"Loaded {len(prices_df.columns)} assets with {len(prices_df)} days")
        with st.expander("📈 Data Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Assets Loaded", len(prices_df.columns))
            with col2:
                st.metric("Trading Days", len(prices_df))
            with col3:
                st.metric("Date Range", f"{prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")
            st.subheader("Sample Data")
            st.dataframe(prices_df.head())
        
        return prices_df

# =====================================================
# SECTION 2: 🔍 Advanced Regime Detection
# =====================================================

class RegimeDetector:
    """Advanced regime detection with multiple methods and validation"""
    
    def __init__(self):
        self.models = {}
        self.regime_characteristics = {}
        self.regime_label_map = {}
    
    def prepare_features(self, returns: pd.Series, lookback_windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create comprehensive feature set for regime detection"""
        features = pd.DataFrame(index=returns.index)
        # Validate input
        if not isinstance(returns, pd.Series):
            logging.error(f"Expected pandas Series for returns, got {type(returns)}")
            raise TypeError("Input 'returns' must be a pandas Series")
    
        # Ensure returns has a proper index
        if returns.index.isna().all():
            logging.error("Returns Series has no valid index")
            raise ValueError("Returns Series must have a valid index")
    
        features = pd.DataFrame(index=returns.index)
        # Basic return features
        features['returns'] = returns
        features['abs_returns'] = np.abs(returns)
        features['squared_returns'] = returns ** 2
        
        # Volatility features
        for window in lookback_windows:
            features[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            features[f'realized_vol_{window}'] = np.sqrt(returns.rolling(window).apply(lambda x: (x**2).sum())) * np.sqrt(252)
        
        # Momentum and trend features
        for window in lookback_windows:
            features[f'momentum_{window}'] = returns.rolling(window).mean()
            # Trend: slope of a linear regression over the window
            features[f'trend_{window}'] = returns.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False)
        
        # Market microstructure features (approximated from returns)
        features['autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
        features['skewness'] = returns.rolling(60).apply(lambda x: x.skew(), raw=False)
        features['kurtosis'] = returns.rolling(60).apply(lambda x: x.kurtosis(), raw=False)

        
        # Volatility clustering (correlation of squared returns)
        features['vol_clustering'] = returns.rolling(20).apply(
            lambda x: np.corrcoef(x[:-1]**2, x[1:]**2)[0,1] if len(x) > 1 else 0, raw=False
        )
        # Drop rows with NaN values
        features = features.dropna()
        
        if features.empty:
            logging.warning("Features DataFrame is empty after dropping NaNs")
            st.warning("⚠️ No valid features generated. Check data quality or lookback windows.")
    
        return features
    
    def fit_hmm(self, features: pd.DataFrame, n_states: int = 3) -> Tuple[Optional[np.ndarray], Optional[hmm.GaussianHMM]]:
        """Fit Hidden Markov Model with enhanced initialization and BIC/AIC for model selection (conceptual)"""
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        
        # Initialize with K-means
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10) # n_init for robustness
            kmeans.fit(X)
            # initial_states = kmeans.predict(X) # Not directly used for HMM init, but for understanding clusters
        except Exception as e:
            logging.error(f"KMeans initialization failed: {e}")
            st.error("❌ KMeans initialization failed. This might indicate issues with data or n_states.")
            return None, None
        
        # Fit HMM with multiple random initializations
        best_score = -np.inf
        best_model = None
        best_states = None
        
        st.info(f"Attempting to fit HMM with {n_states} states. This may take a moment...")
        
        for seed in range(5):  # Try multiple initializations for robustness
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full", # Can also try 'diag', 'tied', 'spherical'
                n_iter=1000,
                random_state=seed,
                tol=1e-4, # Tolerance for convergence
                init_params="stmc" # Initialize startprob, transmat, means, covars
            )
            
            try:
                # Fit the model
                model.fit(X)
                score = model.score(X) # Log-likelihood
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_states = model.predict(X)
                logging.info(f"HMM fit with seed {seed}, score: {score:.2f}")
            except Exception as e:
                logging.warning(f"HMM fit failed for seed {seed}: {e}")
                continue
        
        if best_model is None:
            st.error("❌ Failed to fit HMM model after multiple attempts. Check data quality or try different parameters.")
            return None, None
        
        self.models['hmm'] = best_model
        
        return best_states, best_model
    
    def fit_gmm_comparison(self, features: pd.DataFrame, n_states: int = 3) -> np.ndarray:
        """Fit Gaussian Mixture Model for comparison"""
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        
        gmm = GaussianMixture(n_components=n_states, random_state=42, n_init=10)
        gmm_states = gmm.fit_predict(X)
        
        self.models['gmm'] = gmm
        return gmm_states
    
    def analyze_regime_characteristics(self, returns: pd.Series, states: np.ndarray, 
                                       features: pd.DataFrame) -> Dict:
        characteristics = {}
        self.regime_label_map = {}
        used_labels = set()  # Track used labels to ensure uniqueness

        # ... (alignment code unchanged)

        for state in np.unique(states):
            mask = states == state
            regime_returns = returns.iloc[mask]
            regime_features = features.iloc[mask]
        
            if regime_returns.empty:
                continue

            # Annualized metrics
            avg_return_annualized = regime_returns.mean() * 252
            volatility_annualized = regime_returns.std() * np.sqrt(252)
            sharpe_ratio = avg_return_annualized / volatility_annualized if volatility_annualized > 0 else 0
        
            stats = {
                'frequency': mask.sum() / len(states),
                'avg_return': avg_return_annualized,
                'volatility': volatility_annualized,
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis(),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self._calculate_max_drawdown(regime_returns),
                'average_duration': self._calculate_average_duration(mask)
            }
        
            # Classify regime type
            base_label = None
            if stats['avg_return'] > 0.10 and stats['volatility'] < 0.15:
                base_label = "Bull Market"
            elif stats['avg_return'] < -0.10 and stats['volatility'] > 0.25:
                base_label = "Bear Market"
            elif stats['volatility'] > 0.30:
                base_label = "Volatile Market"
            else:
                base_label = "Neutral Market"
        
            # Ensure unique label
            regime_type = base_label
            counter = 1
            while regime_type in used_labels:
                regime_type = f"{base_label} {counter}"
                counter += 1
            used_labels.add(regime_type)
        
            stats['regime_type'] = regime_type
            self.regime_label_map[state] = regime_type
            characteristics[regime_type] = stats
    
        return characteristics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if returns.empty:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_average_duration(self, mask: np.ndarray) -> float:
        """Calculate average duration of regime"""
        durations = []
        current_duration = 0
        
        # Convert boolean mask to integer array for easier processing
        mask_int = mask.astype(int)
        
        # Iterate through the mask to find consecutive blocks of the same state
        for i in range(len(mask_int)):
            if mask_int[i] == 1: # If currently in the target regime
                current_duration += 1
            else: # If not in the target regime
                if current_duration > 0: # If we just exited a block of the target regime
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add the last duration if the sequence ends in the target regime
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0

# =====================================================
# SECTION 3: 🎲 Advanced Monte Carlo Engine
# =====================================================

class MonteCarloEngine:
    """Advanced Monte Carlo simulation with multiple distribution models"""
    
    def __init__(self):
        self.simulation_results = {}
    
    def generate_regime_scenarios(self, returns: pd.DataFrame, regime_states: np.ndarray,
                                current_regime: int, n_simulations: int = 1000,
                                horizon: int = 252, method: str = 'gaussian',
                                regime_label_map: Dict = None) -> Dict:
        """Generate comprehensive regime-conditioned scenarios"""
        # Validate simulation method
        valid_methods = ['gaussian', 'copula', 'bootstrap']
        if method not in valid_methods:
            logging.error(f"Invalid simulation method: {method}. Valid options are {valid_methods}")
            st.error(f"Unsupported simulation method: {method}. Please select one of: {', '.join(valid_methods)}")
            return {}

        
        if regime_label_map:
            regime_label = regime_label_map.get(current_regime, f"Regime {current_regime}")
            logging.debug(f"Generating {n_simulations} scenarios for {horizon} days starting in {regime_label}")
        else:
            logging.debug(f"Generating {n_simulations} scenarios for {horizon} days starting in Regime {current_regime}") 
            
        # Ensure returns and regime_states are aligned
        if len(returns) != len(regime_states):
            common_index = returns.index.intersection(pd.Series(regime_states, index=returns.index).index)
            returns = returns.loc[common_index]
            regime_states = regime_states[pd.Series(regime_states, index=returns.index).index.get_indexer(common_index)]
            logging.warning("Returns and regime_states length mismatch in MC engine, aligning to common index.")
            if len(returns) != len(regime_states):
                st.error("Critical alignment error in MC engine. Cannot generate scenarios.")
                return {}

        # Separate returns by regime
        regime_data = {}
        for regime in np.unique(regime_states):
            mask = regime_states == regime
            regime_returns = returns.iloc[mask]
            
            if regime_returns.empty:
                regime_label = regime_label_map.get(regime, f"Regime {regime}") if regime_label_map else f"Regime {regime}"
                logging.warning(f"Regime {regime} has no historical data points. Skipping for MC simulation.")
                continue

            # Ensure covariance matrix is well-conditioned
            cov_matrix = regime_returns.cov()
            # Add regularization to covariance matrix to prevent singularity
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6 
            
            regime_data[regime] = {
                'returns': regime_returns,
                'mean': regime_returns.mean(),
                'cov': cov_matrix,
                'n_obs': len(regime_returns)
            }
        
        if not regime_data:
            st.error("No valid regime data to generate scenarios. Check regime detection results.")
            return {}

        # Generate transition matrix
        transition_matrix = self._estimate_transition_matrix(regime_states)
        
        # Ensure current_regime is a valid key in regime_data
        if current_regime not in regime_data:
            regime_label = regime_label_map.get(current_regime, f"Regime {current_regime}") if regime_label_map else f"Regime {current_regime}"
            st.warning(f"Current regime ({current_regime}) has no historical data. Falling back to a default regime.")
            current_regime = next(iter(regime_data)) # Pick the first available regime

        # Generate scenarios
        scenarios = {}
        
        if method == 'gaussian':
            scenarios = self._generate_gaussian_scenarios(
                regime_data, current_regime, transition_matrix, 
                n_simulations, horizon
            )
        elif method == 'copula':
            scenarios = self._generate_copula_scenarios(
                regime_data, current_regime, transition_matrix,
                n_simulations, horizon
            )
        elif method == 'bootstrap':
            scenarios = self._generate_bootstrap_scenarios(
                regime_data, current_regime, transition_matrix,
                n_simulations, horizon
            )
        else:
            st.error(f"Unsupported simulation method: {method}")
            return {}
        
        # Add regime path information
        scenarios['regime_paths'] = self._simulate_regime_paths(
            current_regime, transition_matrix, n_simulations, horizon
        )
        
        return scenarios
    
    def _estimate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Estimate regime transition matrix"""
        unique_states = np.unique(states)
        n_states = len(unique_states)
        
        # Create a mapping from actual state values to 0-indexed integers
        state_to_idx = {state: i for i, state in enumerate(unique_states)}
        
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            from_state_idx = state_to_idx[states[i]]
            to_state_idx = state_to_idx[states[i + 1]]
            transition_matrix[from_state_idx, to_state_idx] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1)
        # Replace 0s with 1s to avoid division by zero, resulting in 0/1 = 0 for that row
        row_sums[row_sums == 0] = 1 
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    def _generate_gaussian_scenarios(self, regime_data: Dict, current_regime: int,
                                   transition_matrix: np.ndarray, n_sim: int,
                                   horizon: int) -> Dict:
        """Generate Gaussian scenarios"""
        # Get number of assets from the first regime's mean vector
        n_assets = len(regime_data[current_regime]['mean']) 
        scenarios = np.zeros((n_sim, horizon, n_assets))
        
        for sim in range(n_sim):
            regime_path = self._simulate_single_regime_path(
                current_regime, transition_matrix, horizon
            )
            
            for t in range(horizon):
                regime = regime_path[t]
                
                # Fallback if a simulated regime has no historical data
                if regime not in regime_data:
                    logging.warning(f"Simulated regime {regime} not found in historical data. Using current_regime data.")
                    regime = current_regime # Fallback to the initial regime's data
                
                mu = regime_data[regime]['mean'].values
                cov = regime_data[regime]['cov'].values
                
                try:
                    scenarios[sim, t, :] = np.random.multivariate_normal(mu, cov)
                except np.linalg.LinAlgError:
                    logging.warning(f"Singular covariance matrix for regime {regime}. Falling back to diagonal covariance.")
                    # Fallback to diagonal covariance (assuming independence)
                    scenarios[sim, t, :] = np.random.normal(mu, np.sqrt(np.diag(cov)))
        
        return {
            'returns': scenarios,
            'method': 'gaussian',
            'n_simulations': n_sim,
            'horizon': horizon
        }
    
    def _generate_copula_scenarios(self, regime_data: Dict, current_regime: int,
                                 transition_matrix: np.ndarray, n_sim: int,
                                 horizon: int) -> Dict:
        """Generate scenarios using copula approach (t-distribution marginals, Gaussian copula)"""
        n_assets = len(regime_data[current_regime]['mean'])
        scenarios = np.zeros((n_sim, horizon, n_assets))
        
        # Estimate marginal distributions for each regime
        marginal_params = {}
        for regime, data in regime_data.items():
            marginal_params[regime] = {}
            for i, asset in enumerate(data['returns'].columns):
                returns = data['returns'][asset].values
                if len(returns) > 1:
                    try:
                        # Fit t-distribution: df, loc, scale
                        params = t.fit(returns)
                        marginal_params[regime][asset] = params
                    except Exception as e:
                        logging.warning(f"Could not fit t-distribution for {asset} in regime {regime}: {e}. Using normal approximation.")
                        marginal_params[regime][asset] = (np.inf, returns.mean(), returns.std()) # Fallback to normal
                else:
                    logging.warning(f"Not enough data for {asset} in regime {regime} to fit t-distribution. Using normal approximation.")
                    marginal_params[regime][asset] = (np.inf, returns.mean(), returns.std()) # Fallback to normal
        
        for sim in range(n_sim):
            regime_path = self._simulate_single_regime_path(
                current_regime, transition_matrix, horizon
            )
            
            for time_step in range(horizon):
                regime = regime_path[time_step]
                
                if regime not in regime_data:
                    logging.warning(f"Simulated regime {regime} not found in historical data. Using current_regime data.")
                    regime = current_regime
                
                # Generate correlated uniform random variables using Gaussian copula
                corr_matrix = np.corrcoef(regime_data[regime]['returns'].T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0) # Handle NaN values, replace with 0
                
                # Ensure correlation matrix is positive semi-definite
                try:
                    # Add a small diagonal to ensure positive definiteness
                    corr_matrix = corr_matrix + np.eye(n_assets) * 1e-6 
                    normal_samples = np.random.multivariate_normal(
                        np.zeros(n_assets), corr_matrix
                    )
                    uniform_samples = norm.cdf(normal_samples)
                    
                    # Transform to marginal distributions
                    for i, asset in enumerate(regime_data[regime]['returns'].columns):
                        params = marginal_params[regime].get(asset, (np.inf, 0, 1)) # Default to standard normal if params missing
                        # Ensure df is not too small for t.ppf
                        df = max(params[0], 1e-6) # degrees of freedom must be > 0
                        scenarios[sim, time_step, i] = t.ppf(uniform_samples[i], df, loc=params[1], scale=params[2])
                except Exception as e:
                    logging.warning(f"Copula generation failed for regime {regime} at sim {sim}, time {t}: {e}. Falling back to Gaussian.")
                    # Fallback to Gaussian if copula fails
                    mu = regime_data[regime]['mean'].values
                    cov = regime_data[regime]['cov'].values
                    scenarios[sim, time_step, :] = np.random.multivariate_normal(mu, cov)
        
        return {
            'returns': scenarios,
            'method': 'copula',
            'n_simulations': n_sim,
            'horizon': horizon
        }
    
    def _generate_bootstrap_scenarios(self, regime_data: Dict, current_regime: int,
                                    transition_matrix: np.ndarray, n_sim: int,
                                    horizon: int) -> Dict:
        """Generate scenarios using block bootstrap"""
        n_assets = len(regime_data[current_regime]['mean'])
        scenarios = np.zeros((n_sim, horizon, n_assets))
        block_size = 5  # 5-day blocks
        
        for sim in range(n_sim):
            regime_path = self._simulate_single_regime_path(
                current_regime, transition_matrix, horizon
            )
            
            for t in range(horizon):
                regime = regime_path[t]
                
                if regime not in regime_data:
                    logging.warning(f"Simulated regime {regime} not found in historical data. Using current_regime data.")
                    regime = current_regime

                regime_returns = regime_data[regime]['returns']
                
                if len(regime_returns) < block_size:
                    # Fallback to random sampling if not enough data for block bootstrap
                    if not regime_returns.empty:
                        idx = np.random.randint(0, len(regime_returns))
                        scenarios[sim, t, :] = regime_returns.iloc[idx].values
                    else:
                        # If regime_returns is empty, fill with zeros or a small random value
                        scenarios[sim, t, :] = np.zeros(n_assets) # Or np.random.normal(0, 0.001, n_assets)
                        logging.warning(f"Regime {regime} has no data for bootstrap. Filling with zeros.")
                else:
                    # Block bootstrap
                    # Select a random starting point for a block
                    start_idx = np.random.randint(0, len(regime_returns) - block_size + 1)
                    # Select the return from the chosen block based on the current day in the horizon
                    block_idx = t % block_size
                    scenarios[sim, t, :] = regime_returns.iloc[start_idx + block_idx].values
        
        return {
            'returns': scenarios,
            'method': 'bootstrap',
            'n_simulations': n_sim,
            'horizon': horizon
        }
    
    def _simulate_regime_paths(self, current_regime: int, transition_matrix: np.ndarray,
                               n_sim: int, horizon: int) -> np.ndarray:
        """Simulate regime transition paths"""
        n_states = transition_matrix.shape[0]
        paths = np.zeros((n_sim, horizon), dtype=int)

        for sim in range(n_sim):
            paths[sim] = self._simulate_single_regime_path(current_regime, transition_matrix, horizon)

        return paths

    
    def _simulate_single_regime_path(self, current_regime_idx: int, 
                                     transition_matrix: np.ndarray, 
                                     horizon: int) -> np.ndarray:
        """Simulate a single regime path"""
        path = np.zeros(horizon, dtype=int)
        path[0] = current_regime_idx

        for t in range(1, horizon):
            probabilities = transition_matrix[path[t - 1]]
            probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else np.ones_like(probabilities) / len(probabilities)
            path[t] = np.random.choice(len(probabilities), p=probabilities)

        return path


# =====================================================
# SECTION 4: ⚖️ Advanced Portfolio Optimization
# =====================================================

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple objectives and constraints"""
    
    def __init__(self):
        self.optimization_results = {}
    
    def optimize_portfolio(self, scenarios: Dict, method: str = 'cvar',
                         constraints: Dict = None, objectives: List[str] = None,
                         asset_names: List[str] = None, # Added for sector constraints
                         sector_map: Dict[str, str] = None, # Added for sector constraints
                         current_weights: Optional[np.ndarray] = None # Added for turnover
                         ) -> Dict:
        """Comprehensive portfolio optimization"""
        
        if constraints is None:
            constraints = {
                'max_weight': 1.0,
                'min_weight': 0.0,
                'max_sector_weight': 0.6, # Placeholder, needs sector_map
                'turnover_limit': None,   # Placeholder, needs current_weights
                'leverage_limit': 1.0
            }
        
        if objectives is None:
            objectives = ['return', 'risk']
        
        returns_array = scenarios.get('returns')
        if returns_array is None or returns_array.size == 0:
            st.error("Monte Carlo scenarios are empty or invalid. Cannot optimize portfolio.")
            return {}

        n_sim, horizon, n_assets = returns_array.shape
        
        # Calculate portfolio returns for each path
        # (1 + returns_array) gives cumulative product for each asset over horizon
        # .prod(axis=1) gives the cumulative product for each asset across the horizon for each simulation
        # This results in a (n_sim, n_assets) array of total returns over the horizon
        path_returns = (1 + returns_array).prod(axis=1) - 1 
        
        # Clip extreme returns to prevent numerical instability or unrealistic outliers
        path_returns = np.clip(path_returns, -0.99, 10.0) # -99% to +1000%
        
        results = {}
        
        # Ensure asset_names is provided if sector constraints are enabled
        if constraints.get('max_sector_weight') is not None and (asset_names is None or sector_map is None):
            st.warning("Sector constraints enabled but asset_names or sector_map not provided. Ignoring sector constraints.")
            constraints['max_sector_weight'] = None # Disable if data is missing

        if method == 'cvar':
            results = self._optimize_cvar(path_returns, constraints)
        elif method == 'mean_variance':
            results = self._optimize_mean_variance(path_returns, constraints)
        elif method == 'risk_parity':
            results = self._optimize_risk_parity(path_returns, constraints)
        elif method == 'multi_objective':
            results = self._optimize_multi_objective(path_returns, constraints, objectives)
        else:
            st.error(f"Unsupported optimization method: {method}")
            return {}
        
        # Add performance metrics
        if results and 'weights' in results:
            results.update(self._calculate_performance_metrics(path_returns, results['weights']))
        
        return results
    
    def _optimize_cvar(self, path_returns: np.ndarray, constraints: Dict, 
                      alpha: float = 0.05) -> Dict:
        """Optimize portfolio using CVaR (Conditional Value at Risk)"""
        n_sim, n_assets = path_returns.shape
        
        # Decision variables
        weights = cp.Variable(n_assets)
        auxiliary = cp.Variable(n_sim) # For CVaR definition
        var = cp.Variable() # Value at Risk
        
        # Portfolio returns for each simulation path
        portfolio_returns = path_returns @ weights
        
        # CVaR objective: VaR + (1 / (alpha * n_sim)) * sum(max(0, -portfolio_returns - VaR))
        cvar_obj = var + (1 / (alpha * n_sim)) * cp.sum(auxiliary)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= constraints['min_weight'],  # Min weight per asset
            weights <= constraints['max_weight'],  # Max weight per asset
            auxiliary >= 0,  # Auxiliary variable must be non-negative
            auxiliary >= -portfolio_returns - var  # Definition of auxiliary variable for CVaR
        ]
        
        # Add leverage constraint if specified
        if constraints.get('leverage_limit') is not None:
            constraints_list.append(cp.sum(cp.abs(weights)) <= constraints['leverage_limit'])
        
        # Solve optimization
        prob = cp.Problem(cp.Minimize(cvar_obj), constraints_list)
        
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if weights.value is not None:
                return {
                    'weights': weights.value,
                    'cvar': cvar_obj.value,
                    'var': var.value,
                    'status': prob.status,
                    'method': 'cvar'
                }
            else:
                st.warning(f"CVaR optimization did not return weights. Status: {prob.status}")
                return {}
        except Exception as e:
            st.error(f"CVaR optimization failed: {str(e)}")
            logging.error("CVaR optimization error:", exc_info=True)
        
        return {}
    
    def _optimize_mean_variance(self, path_returns: np.ndarray, constraints: Dict) -> Dict:
        """Classic mean-variance optimization"""
        n_sim, n_assets = path_returns.shape
        
        # Calculate statistics
        mean_returns = np.mean(path_returns, axis=0)
        cov_matrix = np.cov(path_returns.T)
        
        # Add regularization to covariance matrix
        cov_matrix += np.eye(n_assets) * 1e-6
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Objective: maximize return - risk penalty (Sharpe-like)
        risk_aversion = 1.0 # Can be tuned or made a parameter
        portfolio_return = mean_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix) # Variance
        
        objective = portfolio_return - risk_aversion * portfolio_risk
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= constraints['min_weight'],
            weights <= constraints['max_weight']
        ]
        
        # Solve
        prob = cp.Problem(cp.Maximize(objective), constraints_list)
        
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if weights.value is not None:
                return {
                    'weights': weights.value,
                    'expected_return': float(mean_returns @ weights.value),
                    'portfolio_risk': float(np.sqrt(weights.value.T @ cov_matrix @ weights.value)), # Standard deviation
                    'status': prob.status,
                    'method': 'mean_variance'
                }
            else:
                st.warning(f"Mean-Variance optimization did not return weights. Status: {prob.status}")
                return {}
        except Exception as e:
            st.error(f"Mean-variance optimization failed: {str(e)}")
            logging.error("Mean-variance optimization error:", exc_info=True)
        
        return {}
    
    def _optimize_risk_parity(self, path_returns: np.ndarray, constraints: Dict) -> Dict:
        """Risk parity optimization"""
        n_sim, n_assets = path_returns.shape
        cov_matrix = np.cov(path_returns.T)
        cov_matrix += np.eye(n_assets) * 1e-6 # Regularization
        
        # Objective function for risk parity: minimize the squared difference between risk contributions
        def risk_parity_objective(weights):
            # Ensure weights are valid (sum to 1, within bounds)
            if not (np.isclose(np.sum(weights), 1.0) and 
                    np.all(weights >= constraints['min_weight']) and 
                    np.all(weights <= constraints['max_weight'])):
                return np.inf # Penalize invalid weights during optimization search

            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            if portfolio_vol == 0: # Avoid division by zero
                return np.inf
            
            # Marginal Risk Contribution (MRC)
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            
            # Total Risk Contribution (TRC) for each asset
            risk_contrib = weights * marginal_risk
            
            # Target risk contribution for each asset (equal contribution)
            target_risk_contrib = portfolio_vol / n_assets
            
            # Minimize the sum of squared differences between actual and target risk contributions
            return np.sum((risk_contrib - target_risk_contrib) ** 2)
        
        # Constraints for scipy.optimize.minimize
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        ]
        
        # Bounds for each weight
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(risk_parity_objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=cons, tol=1e-6)
            
            if result.success:
                weights = result.x
                # Re-normalize weights if they slightly deviate due to numerical precision
                weights = weights / np.sum(weights)
                return {
                    'weights': weights,
                    'risk_parity_error': result.fun,
                    'status': 'optimal',
                    'method': 'risk_parity'
                }
            else:
                st.warning(f"Risk parity optimization failed: {result.message}")
                return {}
        except Exception as e:
            st.error(f"Risk parity optimization failed: {str(e)}")
            logging.error("Risk parity optimization error:", exc_info=True)
        
        return {}
    
    def _calculate_performance_metrics(self, path_returns: np.ndarray, weights: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics"""
        # Ensure weights are valid (sum to 1)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio returns for each simulation path
        portfolio_returns = path_returns @ weights
        
        # Mean and Std Dev of the total returns over the simulation horizon
        mean_horizon_return = np.mean(portfolio_returns)
        std_horizon_return = np.std(portfolio_returns)

        metrics = {
            'expected_return': mean_horizon_return,
            'volatility': std_horizon_return,
            'sharpe_ratio': mean_horizon_return / std_horizon_return if std_horizon_return > 0 else 0,
            'var_95': np.percentile(portfolio_returns, 5), # 5th percentile of horizon returns
            'var_99': np.percentile(portfolio_returns, 1), # 1st percentile of horizon returns
            'cvar_95': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]),
            'cvar_99': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)]),
            'max_drawdown': self._calculate_max_drawdown_from_returns(portfolio_returns), # Max drawdown from simulated paths
            'skewness': pd.Series(portfolio_returns).skew(),
            'kurtosis': pd.Series(portfolio_returns).kurtosis(),
            'downside_deviation': np.std(portfolio_returns[portfolio_returns < 0]) if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0,
            'sortino_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns[portfolio_returns < 0]) 
            if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0,
            'calmar_ratio': np.mean(portfolio_returns) / abs(self._calculate_max_drawdown_from_returns(portfolio_returns)) if self._calculate_max_drawdown_from_returns(portfolio_returns) != 0 else 0
        }
        
        return metrics
    
    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from a series of returns (e.g., simulated portfolio returns)"""
        if returns.size == 0:
            return 0.0
        cumulative_wealth = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_wealth)
        drawdown = (cumulative_wealth - running_max) / running_max
        return np.min(drawdown) if drawdown.size > 0 else 0.0
    
    def _optimize_multi_objective(self, path_returns: np.ndarray, constraints: Dict, 
                                objectives: List[str]) -> Dict:
        """Multi-objective optimization using efficient frontier"""
        results = {}
        
        # Generate efficient frontier
        efficient_portfolios = []
        # Define a range of target risk levels (standard deviation of horizon returns)
        min_risk = np.min(np.std(path_returns, axis=0)) * 0.5 # Heuristic min risk
        max_risk = np.max(np.std(path_returns, axis=0)) * 1.5 # Heuristic max risk
        
        # Ensure min_risk is positive and max_risk > min_risk
        if min_risk <= 0: min_risk = 0.01
        if max_risk <= min_risk: max_risk = min_risk + 0.05

        risk_levels = np.linspace(min_risk, max_risk, 20) # 20 points on the frontier
        
        st.info(f"Generating efficient frontier across risk levels from {min_risk:.2f} to {max_risk:.2f}...")

        for target_risk in risk_levels:
            portfolio = self._optimize_for_target_risk(path_returns, constraints, target_risk)
            if portfolio:
                efficient_portfolios.append(portfolio)
        
        if efficient_portfolios:
            # Select portfolio with best Sharpe ratio from the frontier
            best_portfolio = max(efficient_portfolios, 
                               key=lambda p: p.get('sharpe_ratio', -np.inf)) # Use -inf for comparison
            results = best_portfolio
            results['efficient_frontier'] = efficient_portfolios
            results['method'] = 'multi_objective'
        else:
            st.warning("No efficient portfolios found. Check optimization constraints or data.")
        
        return results
    
    def _optimize_for_target_risk(self, path_returns: np.ndarray, constraints: Dict, 
                                target_risk: float) -> Dict:
        """Optimize for a specific risk target (minimize risk for target return, or maximize return for target risk)"""
        n_sim, n_assets = path_returns.shape
        mean_returns = np.mean(path_returns, axis=0)
        cov_matrix = np.cov(path_returns.T) + np.eye(n_assets) * 1e-6 # Regularization
        
        weights = cp.Variable(n_assets)
        portfolio_return = mean_returns @ weights
        portfolio_risk = cp.sqrt(cp.quad_form(weights, cov_matrix)) # Standard deviation
        
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= constraints['min_weight'],
            weights <= constraints['max_weight'],
            portfolio_risk <= target_risk # Constraint: portfolio risk must be <= target
        ]
        
        # Objective: Maximize return for the given risk target
        prob = cp.Problem(cp.Maximize(portfolio_return), constraints_list)
        
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if weights.value is not None and prob.status in ["optimal", "optimal_near"]:
                actual_return = float(mean_returns @ weights.value)
                actual_risk = float(np.sqrt(weights.value.T @ cov_matrix @ weights.value))
                return {
                    'weights': weights.value,
                    'expected_return': actual_return,
                    'volatility': actual_risk, # Renamed to volatility for consistency
                    'sharpe_ratio': actual_return / actual_risk if actual_risk > 0 else 0
                }
        except Exception as e:
            logging.warning(f"Optimization for target risk {target_risk:.2f} failed: {e}")
            pass # Continue to next target risk
        
        return {}

# =====================================================
# SECTION 5: 📊 Advanced Visualization & Dashboard
# =====================================================

class DashboardGenerator:
    """Generate comprehensive dashboard with advanced visualizations"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_regime_analysis_dashboard(self, returns: pd.DataFrame, regime_states: np.ndarray, 
                                   regime_characteristics: Dict, features: pd.DataFrame, 
                                   regime_label_map: Dict):  # Add regime_label_map parameter
        """Create comprehensive regime analysis dashboard"""
    
        # Ensure returns and regime_states are aligned
        if len(returns) != len(regime_states):
            common_index = returns.index.intersection(pd.Series(regime_states, index=returns.index).index)
            returns_aligned = returns.loc[common_index]
            regime_states_aligned = regime_states[pd.Series(regime_states, index=returns.index).index.get_indexer(common_index)]
            features_aligned = features.loc[common_index]
            logging.warning("Returns/features and regime_states length mismatch for dashboard, aligning.")
        else:
            returns_aligned = returns
            regime_states_aligned = regime_states
            features_aligned = features

        if returns_aligned.empty or regime_states_aligned.size == 0:
            st.warning("No aligned data for regime analysis dashboard. Skipping plots.")
            return

        # Convert numerical regime states to descriptive labels
        descriptive_states = np.array([regime_label_map.get(state, f"Regime {state}") for state in regime_states_aligned])

        # Regime timeline
        fig_timeline = self._create_regime_timeline(
            returns_aligned.index, 
            regime_states_aligned, 
            descriptive_states,  # Pass descriptive states
            returns_aligned.iloc[:, 0]
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
        # Regime characteristics heatmap
        fig_heatmap = self._create_regime_heatmap(regime_characteristics)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
        # Regime distribution plots
        col1, col2 = st.columns(2)
        with col1:
            fig_returns = self._create_regime_return_distribution(
                returns_aligned.iloc[:, 0], 
                descriptive_states  # Use descriptive states
            )
            st.plotly_chart(fig_returns, use_container_width=True)
    
        with col2:
            fig_vol = self._create_regime_volatility_analysis(
                features_aligned, 
                descriptive_states  # Use descriptive states
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    
    def _create_regime_timeline(self, dates: pd.DatetimeIndex, states: np.ndarray, descriptive_states: np.ndarray,
                              asset_returns: pd.Series) -> go.Figure:
        """Create regime timeline visualization"""
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=['Asset Price with Regime Overlay', 'Regime States'],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price chart with regime colors
        # Calculate cumulative returns from the asset_returns series
        cumulative_returns = (1 + asset_returns).cumprod()
        
        # Map states to unique integer indices for consistent coloring
        cumulative_returns = (1 + asset_returns).cumprod()
        unique_states = np.unique(descriptive_states)  # Use descriptive states
        state_color_map = {state: self.colors[i % len(self.colors)] for i, state in enumerate(unique_states)}

        for state in unique_states:
            mask = descriptive_states == state
            regime_dates = dates[mask]
            regime_prices = cumulative_returns.loc[regime_dates] # Use .loc for index alignment
            
            if not regime_dates.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regime_dates,
                        y=regime_prices,
                        mode='lines', # Changed to lines for smoother appearance
                        name=state,
                        line=dict(color=state_color_map[state], width=2),
                        showlegend=True # Show legend for each regime
                    ),
                    row=1, col=1
                )
        
        # Regime state bar
        # Create a discrete color scale for the states
        state_colors = [state_color_map[s] for s in descriptive_states]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=states,
                mode='markers',
                marker=dict(
                    color=state_colors,
                    size=8,
                    symbol='square'
                ),
                name='Regime States',
                showlegend=False, # Legend already shown in price chart
                text=descriptive_states,  # Show descriptive labels on hover
                hovertemplate="Date: %{x}<br>Regime: %{text}<extra></extra>"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Market Regime Analysis",
            height=600,
            showlegend=True,
            hovermode="x unified" # Unified hover for better UX
        )
        
        return fig
    
    def _create_regime_heatmap(self, characteristics: Dict) -> go.Figure:
        """Create regime characteristics heatmap"""
        metrics = ['avg_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'frequency', 'average_duration']
        regime_names = sorted(list(characteristics.keys())) # Sort for consistent order
        
        z_values = []
        for metric in metrics:
            row = []
            for regime in regime_names:
                value = characteristics[regime].get(metric, np.nan) # Use NaN for missing
                row.append(value)
            z_values.append(row)
        
        # Create text annotations for the heatmap cells
        text_values = []
        for i, metric in enumerate(metrics):
            row_text = []
            for j, regime in enumerate(regime_names):
                val = z_values[i][j]
                if np.isnan(val):
                    row_text.append("N/A")
                elif metric in ['avg_return', 'volatility', 'max_drawdown']:
                    row_text.append(f"{val:.2%}")
                elif metric in ['sharpe_ratio']:
                    row_text.append(f"{val:.2f}")
                elif metric in ['frequency']:
                    row_text.append(f"{val:.1%}")
                elif metric in ['average_duration']:
                    row_text.append(f"{val:.1f} days")
                else:
                    row_text.append(f"{val:.3f}")
            text_values.append(row_text)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=regime_names,
            y=metrics,
            colorscale='RdYlBu', # Red-Yellow-Blue for divergence
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverinfo="x+y+z+text"
        ))
        
        fig.update_layout(
            title="Regime Characteristics Heatmap",
            height=450, # Slightly increased height
            xaxis_title="Regime",
            yaxis_title="Metric"
        )
        
        return fig
    
    def _create_regime_return_distribution(self, returns: pd.Series, descriptive_states: np.ndarray) -> go.Figure:
        """Create regime-specific return distributions"""
        fig = go.Figure()
        
        # Ensure returns and states are aligned
        if len(returns) != len(descriptive_states):
            common_index = returns.index.intersection(pd.Series(descriptive_states, index=returns.index).index)
            returns_aligned = returns.loc[common_index]
            descriptive_states_aligned = descriptive_states[pd.Series(descriptive_states, index=returns.index).index.get_indexer(common_index)]
        else:
            returns_aligned = returns
            descriptive_states_aligned = descriptive_states

        unique_states = np.unique(descriptive_states_aligned)
        state_color_map = {state: self.colors[i % len(self.colors)] for i, state in enumerate(unique_states)}

        for state in unique_states:
            mask = descriptive_states_aligned == state
            regime_returns = returns_aligned.iloc[mask]
            
            if not regime_returns.empty:
                fig.add_trace(go.Histogram(
                    x=regime_returns,
                    name=state,
                    opacity=0.6, # Slightly more transparent for overlay
                    nbinsx=50, # More bins for finer distribution
                    marker_color=state_color_map[state]
                ))
        
        fig.update_layout(
            title="Return Distributions by Regime",
            xaxis_title="Daily Returns",
            yaxis_title="Frequency",
            barmode='overlay',
            legend_title_text="Regime"
        )
        
        return fig
    
    def _create_regime_volatility_analysis(self, features: pd.DataFrame, descriptive_states: np.ndarray) -> go.Figure:
        """Create volatility analysis by regime"""
        fig = go.Figure()
    
        # Ensure features and states are aligned
        if len(features) != len(descriptive_states):
            common_index = features.index.intersection(pd.Series(descriptive_states, index=features.index).index)
            features_aligned = features.loc[common_index]
            descriptive_states_aligned = descriptive_states[pd.Series(descriptive_states, index=features.index).index.get_indexer(common_index)]
        else:
            features_aligned = features
            descriptive_states_aligned = descriptive_states

        vol_col = next((col for col in features_aligned.columns if 'volatility' in col), None)
    
        if vol_col is None:
            st.warning("No volatility feature found for regime volatility analysis.")
            return go.Figure().update_layout(title="Volatility Analysis (No Volatility Feature)")

        aligned_vol = features_aligned[vol_col]
    
        unique_states = np.unique(descriptive_states_aligned)
        state_color_map = {state: self.colors[i % len(self.colors)] for i, state in enumerate(unique_states)}

        for state in unique_states:
            mask = descriptive_states_aligned == state
            regime_vol = aligned_vol.iloc[mask]
        
            if not regime_vol.empty:
                fig.add_trace(go.Box(
                    y=regime_vol,
                    name=state,  # Use descriptive label
                    boxpoints='outliers',
                    marker_color=state_color_map[state]
                ))
    
        fig.update_layout(
            title="Volatility Distribution by Regime",
            yaxis_title="Annualized Volatility",
            legend_title_text="Regime"
        )
    
        return fig
    
    def create_portfolio_dashboard(self, optimization_results: Dict, scenarios: Dict, 
                                 asset_names: List[str]):
        """Create portfolio optimization dashboard"""
        
        if not optimization_results or 'weights' not in optimization_results:
            st.error("❌ No optimization results to display for portfolio dashboard.")
            return
        
        # Portfolio composition pie chart
        fig_pie = self._create_portfolio_pie_chart(optimization_results['weights'], asset_names)
        
        # Risk-return scatter (of simulated portfolio returns)
        fig_scatter = self._create_risk_return_scatter(scenarios, optimization_results['weights'])
        
        # Performance metrics table
        self._create_performance_metrics_table(optimization_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_scatter, use_container_width=True)

        # If multi-objective, show efficient frontier
        if optimization_results.get('method') == 'multi_objective' and 'efficient_frontier' in optimization_results:
            st.subheader("📈 Efficient Frontier")
            fig_frontier = self._create_efficient_frontier_plot(optimization_results['efficient_frontier'], optimization_results['weights'])
            st.plotly_chart(fig_frontier, use_container_width=True)
    
    def _create_portfolio_pie_chart(self, weights: np.ndarray, asset_names: List[str]) -> go.Figure:
        """Create portfolio composition pie chart"""
        # Ensure weights sum to 1 for accurate percentages
        weights = weights / np.sum(weights)
        
        # Filter out very small weights for cleaner visualization
        significant_threshold = 0.005 # 0.5%
        significant_mask = weights > significant_threshold
        
        display_weights = weights[significant_mask]
        display_names = [asset_names[i] for i in range(len(asset_names)) if significant_mask[i]]
        
        # Aggregate "Others" if there are small weights
        if len(display_weights) < len(weights):
            other_weight = weights[~significant_mask].sum()
            if other_weight > 0: # Only add if there's a non-zero "other" sum
                display_weights = np.append(display_weights, other_weight)
                display_names.append('Others')
        
        fig = go.Figure(data=[go.Pie(
            labels=display_names,
            values=display_weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            marker_colors=px.colors.qualitative.Pastel # Use a different color scale for pie
        )])
        
        fig.update_layout(
            title="Optimal Portfolio Composition",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20) # Adjust margins
        )
        
        return fig
    
    def _create_risk_return_scatter(self, scenarios: Dict, weights: np.ndarray) -> go.Figure:
        """Create risk-return scatter plot of simulated portfolio returns"""
        returns_array = scenarios['returns']
        # path_returns are total returns over the horizon for each simulation
        path_returns = (1 + returns_array).prod(axis=1) - 1
        portfolio_returns = path_returns @ weights
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram( # Use histogram to show distribution
            x=portfolio_returns,
            nbinsx=50,
            name='Simulated Portfolio Returns',
            marker_color='#1f77b4', # Streamlit blue
            opacity=0.7
        ))
        
        # Add mean and VaR/CVaR lines for context
        mean_return = np.mean(portfolio_returns)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])

        fig.add_vline(x=mean_return, line_dash="dash", line_color="green", annotation_text=f"Mean: {mean_return:.2%}")
        fig.add_vline(x=var_95, line_dash="dot", line_color="red", annotation_text=f"VaR 95%: {var_95:.2%}")
        fig.add_vline(x=cvar_95, line_dash="dot", line_color="darkred", annotation_text=f"CVaR 95%: {cvar_95:.2%}")

        fig.update_layout(
            title="Distribution of Simulated Portfolio Returns",
            xaxis_title="Portfolio Return (Over Horizon)",
            yaxis_title="Frequency of Simulations",
            height=400,
            showlegend=False
        )
        
        return fig

    def _create_efficient_frontier_plot(self, efficient_frontier: List[Dict], optimal_weights: np.ndarray) -> go.Figure:
        """Create a plot of the efficient frontier."""
        returns = [p['expected_return'] for p in efficient_frontier]
        risks = [p['volatility'] for p in efficient_frontier]
        sharpe_ratios = [p['sharpe_ratio'] for p in efficient_frontier]

        fig = go.Figure()

        # Plot efficient frontier points
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='lines+markers',
            name='Efficient Frontier',
            marker=dict(size=8, color=sharpe_ratios, colorscale='Viridis', showscale=True, colorbar=dict(title='Sharpe Ratio')),
            hoverinfo='text',
            text=[f"Return: {r:.2%}<br>Risk: {s:.2%}<br>Sharpe: {sh:.2f}" for r, s, sh in zip(returns, risks, sharpe_ratios)]
        ))

        # Highlight the optimal portfolio
        optimal_portfolio_return = optimization_results.get('expected_return', 0)
        optimal_portfolio_risk = optimization_results.get('volatility', 0)

        fig.add_trace(go.Scatter(
            x=[optimal_portfolio_risk],
            y=[optimal_portfolio_return],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(size=12, color='red', symbol='star'),
            hoverinfo='text',
            text=[f"Optimal Return: {optimal_portfolio_return:.2%}<br>Optimal Risk: {optimal_portfolio_risk:.2%}<br>Optimal Sharpe: {optimal_portfolio_return / optimal_portfolio_risk if optimal_portfolio_risk > 0 else 0:.2f}"]
        ))

        fig.update_layout(
            title="Efficient Frontier: Risk vs. Return",
            xaxis_title="Portfolio Volatility (Over Horizon)",
            yaxis_title="Expected Portfolio Return (Over Horizon)",
            hovermode="closest",
            showlegend=True,
            height=500
        )
        return fig
    
    def _create_performance_metrics_table(self, results: Dict):
        """Create performance metrics table"""
        if not results:
            return
        
        metrics_data = []
        # Define order and formatting for key metrics
        display_order = [
            'expected_return', 'volatility', 'sharpe_ratio', 'max_drawdown',
            'var_95', 'cvar_95', 'var_99', 'cvar_99',
            'skewness', 'kurtosis', 'downside_deviation', 'sortino_ratio', 'calmar_ratio'
        ]

        for key in display_order:
            value = results.get(key)
            if value is not None and isinstance(value, (int, float)):
                formatted_value = ""
                if key.startswith(('expected_return', 'volatility', 'var_', 'cvar_', 'max_drawdown')):
                    formatted_value = f"{value:.2%}"
                elif key.endswith('_ratio'):
                    formatted_value = f"{value:.3f}"
                elif key in ['skewness', 'kurtosis', 'downside_deviation']:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value}" # Fallback for other numeric types
                
                metrics_data.append({
                    'Metric': key.replace('_', ' ').title(),
                    'Value': formatted_value
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.subheader("📊 Portfolio Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


    def perform_stress_testing(self, returns: pd.DataFrame, regime_states: np.ndarray, weights: np.ndarray, regime_label_map: Dict):
        """Perform regime-based stress testing of the optimized portfolio"""
        st.markdown('<h2 class="sub-header">🛡️ Stress Testing Results</h2>', unsafe_allow_html=True)

        # Ensure alignment
        if len(returns) != len(regime_states):
            common_index = returns.index.intersection(pd.Series(regime_states, index=returns.index).index)
            returns = returns.loc[common_index]
            regime_states = regime_states[pd.Series(regime_states, index=returns.index).index.get_indexer(common_index)]

        if len(returns) == 0 or len(regime_states) == 0:
            st.warning("⚠️ No aligned data available for stress testing.")
            return
        # Convert numerical regimes to descriptive labels
        descriptive_states = np.array([regime_label_map.get(state, f"Regime {state}") for state in regime_states])

        portfolio_returns = returns @ weights
        df = pd.DataFrame({
            'Portfolio Return': portfolio_returns,
            'Regime': descriptive_states
        })

        # Calculate regime-wise stats
        stress_summary = df.groupby('Regime').agg(
            Avg_Return=('Portfolio Return', 'mean'),
            Volatility=('Portfolio Return', 'std'),
            Max_Drawdown=('Portfolio Return', lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()),
            Min_Return=('Portfolio Return', 'min'),
            Max_Return=('Portfolio Return', 'max'),         
            Count=('Portfolio Return', 'count')
        )

        st.dataframe(stress_summary.style.format({
            'Avg_Return': "{:.2%}",
            'Volatility': "{:.2%}",
            'Max_Drawdown': "{:.2%}",
            'Min_Return': "{:.2%}",
            'Max_Return': "{:.2%}"
        }))


    def _provide_export_options(self, optimization_results: Dict, regime_characteristics: Dict, 
                              asset_names: List[str]):
        """Provide options to export results"""
        st.markdown('<h2 class="sub-header">💾 Export Results</h2>', unsafe_allow_html=True)
        
        # Export portfolio weights
        if optimization_results.get('weights') is not None:
            weights_df = pd.DataFrame({
                'Asset': asset_names,
                'Weight': optimization_results['weights']
            })
            weights_csv = weights_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Portfolio Weights (CSV)",
                data=weights_csv,
                file_name="portfolio_weights.csv",
                mime="text/csv"
            )
    
        # Export regime characteristics
        regime_data = []
        for regime, stats in regime_characteristics.items():
            regime_data.append({
                'Regime': regime,
                **{k: v for k, v in stats.items()}
            })
        regime_df = pd.DataFrame(regime_data)
        regime_csv = regime_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Regime Characteristics (CSV)",
            data=regime_csv,
            file_name="regime_characteristics.csv",
            mime="text/csv"
        )
    
# =====================================================
# SECTION 6: 🚀 Main Streamlit Application
# =====================================================
# SECTION 6: 🚀 Main Streamlit Application
# =====================================================
def main():
    """Main Streamlit application with enhanced visualizations and error handling"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #4a5568;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .regime-badge {
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Regime-Aware Portfolio Optimizer</h1>', unsafe_allow_html=True)
    
    # Feature overview
    with st.expander("🎯 Application Features", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            #### 🔍 Advanced Regime Detection
            - Hidden Markov Models (HMM)
            - Gaussian Mixture Models (GMM)
            - Regime transition analysis
            - Persistence metrics
            """)
        with col2:
            st.markdown("""
            #### 🎲 Monte Carlo Simulation
            - Multiple distribution types
            - Regime-conditioned scenarios
            - Fat-tailed modeling
            - Bootstrap resampling
            """)
        with col3:
            st.markdown("""
            #### ⚖️ Portfolio Optimization
            - CVaR minimization
            - Multi-objective optimization
            - Risk parity approach
            - Comprehensive stress testing
            """)
    
    # Initialize components with error handling
    try:
        data_manager = DataManager()
        regime_detector = RegimeDetector()
        mc_engine = MonteCarloEngine()
        optimizer = PortfolioOptimizer()
        dashboard = DashboardGenerator()
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {str(e)}")
        st.stop()
    
    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration Panel")
    
    # Data Configuration Section
    st.sidebar.markdown("### 📊 Data Configuration")
    exchange = st.sidebar.selectbox(
        "Select Exchange/Universe", 
        ["NSE", "NASDAQ", "Global_ETFs", "Crypto", "Custom"],
        help="Choose your preferred asset universe"
    )
    
    custom_assets = []
    if exchange == "Custom":
        asset_input = st.sidebar.text_area(
            "Enter asset symbols (one per line)", 
            value="AAPL\nMSFT\nGOOGL\nAMZN\nSPY\n^NSEI",
            height=150
        )
        custom_assets = [sym.strip().upper() for sym in asset_input.split('\n') if sym.strip()]
    else:
        try:
            predefined_assets = data_manager.get_asset_universe(exchange)
            if not predefined_assets:
                st.sidebar.error(f"No assets available for {exchange}")
                custom_assets = []
            else:
                default_selection = predefined_assets[:min(5, len(predefined_assets))]
                selected_assets = st.sidebar.multiselect(
                    f"Select assets from {exchange}", 
                    predefined_assets, 
                    default=default_selection
                )
                custom_assets = selected_assets
        except Exception as e:
            st.sidebar.error(f"Error loading {exchange} assets: {str(e)}")
            custom_assets = []
    
    # Validation
    if not custom_assets:
        st.sidebar.warning("⚠️ Please select at least one asset symbol.")
        st.info("👆 Configure your parameters in the sidebar to begin analysis!")
        return
    
    if len(custom_assets) < 3:
        st.sidebar.warning("⚠️ For optimal results, please select at least 3 assets.")
    
    # Time Period Selection
    st.sidebar.markdown("### 📅 Time Period")
    lookback_period = st.sidebar.selectbox(
        "Historical Data Period", 
        ["1 Year", "2 Years", "3 Years", "5 Years", "10 Years"], 
        index=2
    )
    period_map = {"1 Year": 365, "2 Years": 730, "3 Years": 1095, "5 Years": 1825, "10 Years": 3650}
    start_date = (datetime.now() - timedelta(days=period_map[lookback_period])).strftime('%Y-%m-%d')
    
    # Regime Detection Settings
    st.sidebar.markdown("### 🔍 Regime Detection")
    n_regimes = st.sidebar.slider("Number of Market Regimes", 2, 5, 3)
    regime_method = st.sidebar.selectbox(
        "Detection Method", 
        ["HMM (Recommended)", "HMM + GMM Comparison"],
        help="HMM is generally more robust for time series data"
    )
    
    # Monte Carlo Settings
    st.sidebar.markdown("### 🎲 Monte Carlo Simulation")
    n_simulations = st.sidebar.slider("Number of Simulations", 500, 10000, 2000, step=500)
    simulation_horizon = st.sidebar.slider("Horizon (Trading Days)", 21, 252, 126, step=21)
    simulation_display = st.sidebar.selectbox(
        "Simulation Method", 
        ["Gaussian", "Copula (Fat-tailed)", "Bootstrap"],
        help="Copula captures fat tails better, Bootstrap uses historical patterns"
    )
    
    # Map display labels to internal method names
    simulation_method_map = {
        "Gaussian": "gaussian",
        "Copula (Fat-tailed)": "copula",
        "Bootstrap": "bootstrap"
    }
    simulation_method = simulation_method_map.get(simulation_display)
    
    # Optimization Settings
    st.sidebar.markdown("### ⚖️ Portfolio Optimization")
    optimization_display = st.sidebar.selectbox(
        "Optimization Method", 
        ["CVaR (Recommended)", "Mean-Variance", "Risk Parity", "Multi-Objective"],
        help="CVaR focuses on tail risk, Mean-Variance on risk-return tradeoff"
    )
    
    optimization_method_map = {
        "CVaR (Recommended)": "cvar",
        "Mean-Variance": "mean_variance",
        "Risk Parity": "risk_parity",
        "Multi-Objective": "multi_objective"
    }
    optimization_method = optimization_method_map[optimization_display]
    
    # Risk Management Settings
    st.sidebar.markdown("### 🛡️ Risk Management")
    var_confidence = st.sidebar.slider("VaR/CVaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    # Dynamic position sizing based on number of assets
    num_assets = len(custom_assets)
    if num_assets < 5:
        default_max, default_min = 0.6, 0.0
    elif num_assets <= 10:
        default_max, default_min = 0.3, 0.01
    else:
        default_max, default_min = 0.2, 0.005
    
    max_position_size = st.sidebar.slider("Max Position Size (per asset)", 0.1, 1.0, default_max, 0.05)
    min_position_size = st.sidebar.slider("Min Position Size (per asset)", 0.0, 0.1, default_min, 0.01)
    
    # Advanced Settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        enable_sector_constraints = st.checkbox("Enable Sector Constraints", False)
        enable_turnover_control = st.checkbox("Enable Turnover Control", False)
        enable_stress_testing = st.checkbox("Enable Stress Testing", True)
        use_transaction_costs = st.checkbox("Include Transaction Costs", False)
        show_debug_info = st.checkbox("Show Debug Information", False)
    
    # Run Analysis Button
    run_analysis = st.sidebar.button("🚀 Run Complete Analysis", type="primary")
    
    # Asset name mapping for display
    ticker_to_name = {
        # NSE
        "RELIANCE.NS": "Reliance Industries", "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys",
        "TCS.NS": "Tata Consultancy Services", "ICICIBANK.NS": "ICICI Bank", "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel", "ITC.NS": "ITC Ltd.", "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "WIPRO.NS": "Wipro Ltd.",
        # NASDAQ
        "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.", "NVDA": "NVIDIA Corporation",
        # Global ETFs
        "SPY": "SPDR S&P 500 ETF", "EFA": "iShares MSCI EAFE ETF", "EEM": "iShares MSCI Emerging Markets ETF",
        "VNQ": "Vanguard Real Estate ETF", "GLD": "SPDR Gold Trust", "TLT": "iShares 20+ Year Treasury Bond ETF",
        # Crypto
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "ADA-USD": "Cardano", "BNB-USD": "Binance Coin"
    }
    
    # Main Analysis Pipeline
    if run_analysis:
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Data Loading (20% progress)
            status_text.text("📊 Loading and validating market data...")
            progress_bar.progress(20)
            
            price_data = data_manager.load_data_with_validation(custom_assets, start_date)
            
            if price_data.empty:
                st.error("❌ Failed to load data. Please check your asset symbols and try again.")
                return
            
            returns_data = price_data.pct_change().dropna()
            
            if returns_data.empty:
                st.error("❌ No valid return data after processing. Check data period or assets.")
                return
            
            # Data summary
            st.success(f"✅ Loaded {len(price_data.columns)} assets with {len(price_data)} trading days.")
            
            # Display data sample
            with st.expander("📂 Data Sample", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Price Data Sample:**")
                    st.dataframe(price_data.head())
                with col2:
                    st.write("**Returns Data Sample:**")
                    st.dataframe(returns_data.head())
            
            # Historical price charts
            st.markdown('<h2 class="sub-header">📈 Historical Price Analysis</h2>', unsafe_allow_html=True)
            
            # Create price chart tabs
            if len(custom_assets) <= 6:
                # Show all charts if few assets
                cols = st.columns(min(len(custom_assets), 3))
                for i, ticker in enumerate(custom_assets):
                    with cols[i % 3]:
                        display_name = ticker_to_name.get(ticker, ticker)
                        series = price_data[ticker].dropna()
                        
                        if not series.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode='lines',
                                name=display_name,
                                line=dict(width=2)
                            ))
                            fig.update_layout(
                                title=f"{display_name}",
                                height=300,
                                showlegend=False,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                # Use selectbox for many assets
                selected_ticker = st.selectbox(
                    "Select asset to view price chart:", 
                    custom_assets,
                    format_func=lambda x: ticker_to_name.get(x, x)
                )
                
                display_name = ticker_to_name.get(selected_ticker, selected_ticker)
                series = price_data[selected_ticker].dropna()
                
                if not series.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines',
                        name=display_name,
                        line=dict(width=2, color='#667eea')
                    ))
                    fig.update_layout(
                        title=f"Price Chart - {display_name}",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Step 2: Regime Detection (40% progress)
            status_text.text("🔍 Detecting market regimes...")
            progress_bar.progress(40)
            
            primary_asset_returns = returns_data.iloc[:, 0].copy()
            features = regime_detector.prepare_features(primary_asset_returns)
            
            if regime_method == "HMM (Recommended)":
                regime_states, hmm_model = regime_detector.fit_hmm(features, n_states=n_regimes)
            else:
                regime_states, hmm_model = regime_detector.fit_hmm(features, n_states=n_regimes)
                gmm_states = regime_detector.fit_gmm_comparison(features, n_states=n_regimes)
            
            if regime_states is None or hmm_model is None:
                st.error("❌ Regime detection failed. Check data or reduce number of regimes.")
                return
            
            # Align data
            aligned_index = features.index
            returns_aligned = primary_asset_returns.loc[aligned_index]
            regime_states_aligned = pd.Series(regime_states, index=aligned_index)
            
            regime_characteristics = regime_detector.analyze_regime_characteristics(
                returns_aligned, regime_states_aligned.values, features
            )
            
            if not regime_characteristics:
                st.error("❌ Failed to analyze regime characteristics.")
                return
            
            st.success(f"✅ Detected {len(np.unique(regime_states))} market regimes.")
            
            # Current regime display
            current_regime_state = regime_states_aligned.values[-1]
            regime_name = regime_detector.regime_label_map.get(current_regime_state, f"Regime {current_regime_state}")
            st.markdown(f'<div class="regime-badge">🧭 Current Market Regime: {regime_name}</div>', unsafe_allow_html=True)
            
            # Enhanced Regime Analysis Dashboard
            st.markdown('<h2 class="sub-header">🔍 Market Regime Analysis</h2>', unsafe_allow_html=True)
            
            # Regime visualization tabs
            tab1, tab2, tab3 = st.tabs(["📊 Regime Timeline", "📈 Regime Characteristics", "🔄 Transition Matrix"])
            
            with tab1:
                # Regime timeline chart
                fig = go.Figure()
                
                # Add returns as background
                fig.add_trace(go.Scatter(
                    x=returns_aligned.index,
                    y=returns_aligned.values,
                    mode='lines',
                    name='Returns',
                    line=dict(color='lightgray', width=1),
                    opacity=0.6
                ))
                
                # Add regime coloring
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                for regime in np.unique(regime_states):
                    mask = regime_states_aligned == regime
                    regime_label = regime_detector.regime_label_map.get(regime, f"Regime {regime}")
                    
                    fig.add_trace(go.Scatter(
                        x=returns_aligned.index[mask],
                        y=returns_aligned.values[mask],
                        mode='markers',
                        name=regime_label,
                        marker=dict(
                            color=colors[regime % len(colors)],
                            size=4
                        )
                    ))
                
                fig.update_layout(
                    title="Market Regime Timeline",
                    xaxis_title="Date",
                    yaxis_title="Returns",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                window_size = min(21, len(returns_aligned) // 10)  # Adaptive window size
                if window_size > 1:
                    smoothed_returns = signal.savgol_filter(returns_aligned.values, window_size, 3)
                    fig.add_trace(go.Scatter(
                        x=returns_aligned.index,
                        y=smoothed_returns,
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='black', width=2, dash='dash'),
                        opacity=0.8
                    ))
            
            with tab2:
                # Regime characteristics
                regime_stats = []
                for regime, stats in regime_characteristics.items():
                    regime_stats.append({
                        'Regime': regime,
                        'Mean Return': f"{stats.get('mean_return', 0):.4f}",
                        'Volatility': f"{stats.get('volatility', 0):.4f}",
                        'Persistence': f"{stats.get('persistence', 0):.3f}",
                        'Duration (days)': f"{stats.get('avg_duration', 0):.1f}"
                    })
                
                st.dataframe(pd.DataFrame(regime_stats), use_container_width=True)
                
                # Regime distribution plots
                cols = st.columns(2)
                with cols[0]:
                    # Return distribution by regime
                    fig = go.Figure()
                    for regime in np.unique(regime_states):
                        mask = regime_states_aligned == regime
                        regime_returns = returns_aligned[mask]
                        regime_label = regime_detector.regime_label_map.get(regime, f"Regime {regime}")
                        
                        fig.add_trace(go.Histogram(
                            x=regime_returns,
                            name=regime_label,
                            opacity=0.7,
                            nbinsx=30
                        ))
                    
                    fig.update_layout(
                        title="Return Distribution by Regime",
                        xaxis_title="Returns",
                        yaxis_title="Frequency",
                        barmode='overlay',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with cols[1]:
                    # Regime duration analysis
                    durations = []
                    current_regime = regime_states_aligned.values[0]
                    current_duration = 1
                    
                    for i in range(1, len(regime_states_aligned)):
                        if regime_states_aligned.values[i] == current_regime:
                            current_duration += 1
                        else:
                            durations.append({
                                'Regime': regime_detector.regime_label_map.get(current_regime, f"Regime {current_regime}"),
                                'Duration': current_duration
                            })
                            current_regime = regime_states_aligned.values[i]
                            current_duration = 1
                    
                    # Replace the existing box plot section with:
                    if durations:
                        duration_df = pd.DataFrame(durations)
                        # Calculate regime percentages
                        regime_counts = regime_states_aligned.value_counts()
                        total_periods = len(regime_states_aligned)
    
                        regime_percentages = []
                        for regime in np.unique(regime_states):
                            regime_label = regime_detector.regime_label_map.get(regime, f"Regime {regime}")
                            percentage = (regime_counts.get(regime, 0) / total_periods) * 100
                            regime_percentages.append({
                                'Regime': regime_label,
                                'Percentage': percentage
                            })                    
                        percentage_df = pd.DataFrame(regime_percentages)
    
                        fig = go.Figure([go.Bar(
                            x=percentage_df['Regime'],
                            y=percentage_df['Percentage'],
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(percentage_df)]
                        )])
    
                        fig.update_layout(
                            title="Regime Distribution (%)",
                            xaxis_title="Regime",
                            yaxis_title="Percentage (%)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Transition matrix
                if hasattr(hmm_model, 'transmat_'):
                    transition_matrix = hmm_model.transmat_
                    regime_labels = [regime_detector.regime_label_map.get(i, f"Regime {i}") 
                                   for i in range(len(transition_matrix))]
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=transition_matrix,
                        x=regime_labels,
                        y=regime_labels,
                        colorscale='Viridis',
                        text=np.round(transition_matrix, 3),
                        texttemplate="%{text}",
                        textfont={"size": 12}
                    ))
                    
                    fig.update_layout(
                        title="Regime Transition Matrix",
                        xaxis_title="To Regime",
                        yaxis_title="From Regime",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Transition insights
                    st.write("**Key Insights:**")
                    diagonal_persistence = np.diag(transition_matrix)
                    for i, persistence in enumerate(diagonal_persistence):
                        regime_label = regime_labels[i]
                        st.write(f"- {regime_label}: {persistence:.1%} persistence probability")
            
            # Step 3: Monte Carlo Simulation (60% progress)
            
            status_text.text("🎲 Generating Monte Carlo scenarios...")
            progress_bar.progress(60)
            
            current_regime = regime_states_aligned.values[-1]
            current_regime_label = regime_detector.regime_label_map.get(current_regime, f"Regime {current_regime}")
            
            try:
                scenarios = mc_engine.generate_regime_scenarios(
                    returns=returns_data.loc[aligned_index],
                    regime_states=regime_states_aligned.values,
                    current_regime=current_regime,
                    n_simulations=n_simulations,
                    horizon=simulation_horizon,
                    method=simulation_method.lower(),
                    regime_label_map=regime_detector.regime_label_map
                )
                
                # Validate scenarios output
                if not isinstance(scenarios, dict):
                    st.error(f"❌ Monte Carlo simulation returned unexpected type: {type(scenarios)}. Expected a dictionary.")
                    logging.error(f"Monte Carlo simulation returned {type(scenarios)} instead of dict: {scenarios}")
                    return
                
                if not scenarios:
                    st.error("❌ Monte Carlo simulation failed to generate valid scenarios.")
                    return
                
                st.success(f"✅ Generated {n_simulations} scenarios over {simulation_horizon} days.")
                
            except Exception as e:
                st.error(f"❌ Monte Carlo simulation failed: {str(e)}")
                logging.error("Monte Carlo simulation error:", exc_info=True)
                return
            
            # Enhanced Monte Carlo Analysis
            st.markdown('<h2 class="sub-header">🎲 Monte Carlo Simulation Analysis</h2>', unsafe_allow_html=True)
            
            # Monte Carlo tabs
            mc_tab1, mc_tab2, mc_tab3 = st.tabs(["📊 Scenario Paths", "📈 Distribution Comparison", "🎯 Risk Metrics"])
            
            with mc_tab1:
                if 'returns' in scenarios:
                    sim_returns = scenarios['returns']
            
                    # Use optimized weights if available, otherwise use equal weights
                    portfolio_weights = (optimization_results.get('weights', np.ones(len(custom_assets)) / len(custom_assets)) 
                                        if 'optimization_results' in locals() and optimization_results 
                                        else np.ones(len(custom_assets)) / len(custom_assets))
            
                    portfolio_sim_returns = np.zeros((sim_returns.shape[0], sim_returns.shape[1]))
                    for i in range(sim_returns.shape[0]):
                        portfolio_sim_returns[i] = np.dot(sim_returns[i], portfolio_weights)
            
                    portfolio_paths = np.cumprod(1 + portfolio_sim_returns, axis=1)
                    sample_size = min(50, len(portfolio_paths))
                    sample_paths = portfolio_paths[:sample_size]
            
                    fig = go.Figure()
                    for i in range(min(20, len(sample_paths))):
                        fig.add_trace(go.Scatter(
                            y=sample_paths[i],
                            mode='lines',
                            line=dict(width=1, color='rgba(100,149,237,0.3)'),
                            showlegend=False,
                            hovertemplate='Day: %{x}<br>Portfolio Value: %{y:.3f}<extra></extra>'
                        ))
            
                    percentiles = [10, 25, 50, 75, 90]
                    percentile_paths = {}
                    for p in percentiles:
                        percentile_paths[p] = np.percentile(portfolio_paths, p, axis=0)
            
                    fig.add_trace(go.Scatter(y=percentile_paths[90], mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(y=percentile_paths[10], mode='lines', fill='tonexty', fillcolor='rgba(100,149,237,0.1)', line=dict(width=0), name='10th-90th percentile', showlegend=True))
                    fig.add_trace(go.Scatter(y=percentile_paths[75], mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(y=percentile_paths[25], mode='lines', fill='tonexty', fillcolor='rgba(100,149,237,0.2)', line=dict(width=0), name='25th-75th percentile', showlegend=True))
                    fig.add_trace(go.Scatter(y=percentile_paths[50], mode='lines', line=dict(width=3, color='blue'), name='Median Path'))
            
                    mean_path = np.mean(portfolio_paths, axis=0)
                    fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(width=3, color='red', dash='dash'), name='Mean Path'))
            
                    fig.update_layout(
                        title=f"Portfolio Simulation Paths ({n_simulations:,} simulations)",
                        xaxis_title="Days",
                        yaxis_title="Portfolio Value (Starting Value = 1.0)",
                        height=600,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
                    final_values = portfolio_paths[:, -1]
                    final_returns = final_values - 1
            
                    stats_cols = st.columns(5)
                    with stats_cols[0]:
                        st.metric("Expected Final Value", f"{np.mean(final_values):.3f}")
                    with stats_cols[1]:
                        st.metric("Expected Return", f"{np.mean(final_returns):.2%}")
                    with stats_cols[2]:
                        st.metric("Volatility", f"{np.std(final_returns):.2%}")
                    with stats_cols[3]:
                        st.metric("Best Case (90%)", f"{np.percentile(final_returns, 90):.2%}")
                    with stats_cols[4]:
                        st.metric("Worst Case (10%)", f"{np.percentile(final_returns, 10):.2%}")
            
                    positive_returns = np.sum(final_returns > 0) / len(final_returns)
                    st.metric("Success Rate", f"{positive_returns:.1%}", help="Probability of positive returns")
            
                    weights_label = 'optimized' if 'optimization_results' in locals() and optimization_results and 'weights' in optimization_results else 'equal'
                    st.info(f"📌 Simulations starting from {current_regime_label} regime using {simulation_display} method with {weights_label} weights")
            
                    with st.expander("📊 Detailed Portfolio Statistics"):
                        col1, col2 = st.columns(2)
            
                        with col1:
                            st.write("**Return Distribution:**")
                            quartiles = np.percentile(final_returns, [25, 50, 75])
                            st.write(f"- 25th percentile: {quartiles[0]:.2%}")
                            st.write(f"- Median: {quartiles[1]:.2%}")
                            st.write(f"- 75th percentile: {quartiles[2]:.2%}")
                            st.write(f"- Skewness: {pd.Series(final_returns).skew():.3f}")
                            st.write(f"- Kurtosis: {pd.Series(final_returns).kurtosis():.3f}")
            
                        with col2:
                            st.write("**Path Statistics:**")
                            max_values = np.max(portfolio_paths, axis=1)
                            min_values = np.min(portfolio_paths, axis=1)
                            st.write(f"- Average maximum value: {np.mean(max_values):.3f}")
                            st.write(f"- Average minimum value: {np.mean(min_values):.3f}")
                            st.write(f"- Paths ending above 1.0: {np.sum(final_values > 1.0)}/{len(final_values)}")
                            st.write(f"- Paths ending below 0.8: {np.sum(final_values < 0.8)}/{len(final_values)}")
                else:
                    st.warning("⚠️ No valid simulation data available. Please check Monte Carlo settings.")
            
            

            with mc_tab2:
                # Distribution comparison across methods
                st.subheader("🔄 Simulation Method Comparison")
                
                methods = ["gaussian", "copula", "bootstrap"]
                final_returns = {}
                
                for method in methods:
                    try:
                        temp_scenarios = mc_engine.generate_regime_scenarios(
                            returns=returns_data.loc[aligned_index],
                            regime_states=regime_states_aligned.values,
                            current_regime=current_regime,
                            n_simulations=min(1000, n_simulations),  # Limit for comparison
                            horizon=simulation_horizon,
                            method=method,
                            regime_label_map=regime_detector.regime_label_map
                        )
                        
                        if temp_scenarios and "returns" in temp_scenarios:
                            # Calculate final returns for first asset
                            path_returns = np.cumprod(1 + temp_scenarios["returns"][:, :, 0], axis=1)[:, -1] - 1
                            final_returns[method] = path_returns
                    except Exception as e:
                        st.warning(f"Could not generate {method} scenarios: {str(e)}")
                
                if final_returns:
                    fig = go.Figure()
                    colors = ['blue', 'red', 'green']
    
                    for i, (method, returns) in enumerate(final_returns.items()):
                        # Calculate histogram data manually
                        hist, bin_edges = np.histogram(returns, bins=30, density=True)  
                        # Create x values for the line (bin centers)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
                        # Add line trace
                        fig.add_trace(go.Scatter(
                            x=bin_centers,
                            y=hist,
                            mode='lines',
                            name=method.title(),
                            line=dict(width=2, color=colors[i])
                        ))
    
                    fig.update_layout(
                        title="Final Return Distribution Comparison",
                        xaxis_title="Final Returns",
                        yaxis_title="Frequency",
                        height=400,
                        barmode='overlay',  # Overlay histograms for better comparison
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical comparison
                    comparison_stats = []
                    for method, returns in final_returns.items():
                        comparison_stats.append({
                            'Method': method.title(),
                            'Mean': f"{np.mean(returns):.4f}",
                            'Std': f"{np.std(returns):.4f}",
                            'Skewness': f"{pd.Series(returns).skew():.3f}",
                            'Kurtosis': f"{pd.Series(returns).kurtosis():.3f}"
                        })
                    
                    st.dataframe(pd.DataFrame(comparison_stats), use_container_width=True)
            

            
            with mc_tab3:
                # Risk metrics from simulation
                if 'returns' in scenarios:
                    # Calculate portfolio-level metrics (using equal weights for now)
                    equal_weights = np.ones(len(custom_assets)) / len(custom_assets)
                    portfolio_returns = np.dot(scenarios['returns'], equal_weights)
                    final_portfolio_returns = np.cumprod(1 + portfolio_returns, axis=1)[:, -1] - 1
                    
                    # Risk metrics
                    var_95 = np.percentile(final_portfolio_returns, 5)
                    var_99 = np.percentile(final_portfolio_returns, 1)
                    cvar_95 = np.mean(final_portfolio_returns[final_portfolio_returns <= var_95])
                    cvar_99 = np.mean(final_portfolio_returns[final_portfolio_returns <= var_99])
                    
                    # Display metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("VaR (95%)", f"{var_95:.2%}")
                    with metric_cols[1]:
                        st.metric("VaR (99%)", f"{var_99:.2%}")
                    with metric_cols[2]:
                        st.metric("CVaR (95%)", f"{cvar_95:.2%}")
                    with metric_cols[3]:
                        st.metric("CVaR (99%)", f"{cvar_99:.2%}")
                    
                    # Risk-return scatter
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.std(portfolio_returns, axis=1),
                        y=np.mean(portfolio_returns, axis=1),
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=final_portfolio_returns,
                            colorscale='RdYlBu',
                            showscale=True,
                            colorbar=dict(title="Final Return")
                        ),
                        name='Simulations'
                    ))
                    
                    fig.update_layout(
                        title="Risk-Return Scatter (Equal Weight Portfolio)",
                        xaxis_title="Volatility",
                        yaxis_title="Mean Return",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Step 4: Portfolio Optimization (80% progress)
            status_text.text("⚖️ Optimizing portfolio...")
            progress_bar.progress(80)
            
            # Validation for CVaR
            if optimization_method.lower().startswith("cvar") and len(custom_assets) < 3:
                st.error("❌ CVaR optimization requires at least 3 assets. Please select more assets or use a different method.")
                return
            
            # Set up constraints
            constraints = {
                'max_weight': max_position_size,
                'min_weight': min_position_size,
                'max_sector_weight': None,
                'turnover_limit': None,
                'leverage_limit': 1.0
            }
            
            optimization_results = optimizer.optimize_portfolio(
                scenarios=scenarios,
                method=optimization_method,
                constraints=constraints,
                asset_names=custom_assets
            )
            
            if not optimization_results:
                st.error("❌ Portfolio optimization failed.")
                return
            
            st.success("✅ Portfolio optimized successfully.")
            
            # Enhanced Portfolio Results Dashboard
            st.markdown('<h2 class="sub-header">📊 Portfolio Optimization Results</h2>', unsafe_allow_html=True)
            
            # Portfolio tabs
            port_tab1, port_tab2, port_tab3, port_tab4 = st.tabs(["🎯 Allocation", "📈 Performance", "🛡️ Risk Analysis", "📊 Efficient Frontier"])
            
            with port_tab1:
                # Portfolio allocation visualization
                weights = optimization_results.get('weights', np.ones(len(custom_assets)) / len(custom_assets))
                
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=[ticker_to_name.get(asset, asset) for asset in custom_assets],
                    values=weights,
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title="Portfolio Allocation",
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Allocation table
                allocation_df = pd.DataFrame({
                    'Asset': [ticker_to_name.get(asset, asset) for asset in custom_assets],
                    'Symbol': custom_assets,
                    'Weight': weights,
                    'Weight (%)': [f"{w:.2%}" for w in weights]
                }).sort_values('Weight', ascending=False)
                
                st.dataframe(allocation_df, use_container_width=True)
                
                # Portfolio statistics
                if 'expected_return' in optimization_results:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Return", f"{optimization_results['expected_return']:.2%}")
                    with col2:
                        st.metric("Expected Volatility", f"{optimization_results.get('volatility', 0):.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{optimization_results.get('sharpe_ratio', 0):.3f}")
            
            with port_tab2:
                # Portfolio performance projection
                if 'returns' in scenarios and 'weights' in optimization_results:
                    sim_returns = scenarios['returns']
                    portfolio_paths = np.cumprod(1 + np.dot(sim_returns, weights), axis=1)
                    
                    # Performance metrics
                    mean_path = np.mean(portfolio_paths, axis=0)
                    percentiles = [5, 25, 75, 95]
                    percentile_paths = {p: np.percentile(portfolio_paths, p, axis=0) for p in percentiles}
                    
                    fig = go.Figure()
                    
                    # Add percentile bands
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))),
                        y=percentile_paths[95],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        name='95th percentile'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))),
                        y=percentile_paths[5],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.1)',
                        line=dict(width=0),
                        name='5th-95th percentile',
                        showlegend=True
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))),
                        y=percentile_paths[75],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))),
                        y=percentile_paths[25],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(width=0),
                        name='25th-75th percentile',
                        showlegend=True
                    ))
                    
                    # Add mean path
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))),
                        y=mean_path,
                        mode='lines',
                        line=dict(width=3, color='blue'),
                        name='Expected Path'
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Value Projection",
                        xaxis_title="Days",
                        yaxis_title="Portfolio Value",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance statistics
                    final_returns = portfolio_paths[:, -1] - 1
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Expected Final Return", f"{np.mean(final_returns):.2%}")
                    with col2:
                        st.metric("Volatility", f"{np.std(final_returns):.2%}")
                    with col3:
                        st.metric("Best Case (95%)", f"{np.percentile(final_returns, 95):.2%}")
                    with col4:
                        st.metric("Worst Case (5%)", f"{np.percentile(final_returns, 5):.2%}")
            
            with port_tab3:
                # Risk analysis
                if 'returns' in scenarios and 'weights' in optimization_results:
                    portfolio_returns = np.dot(scenarios['returns'], weights)
                    daily_returns = portfolio_returns.flatten()
                    
                    # Risk metrics
                    var_95 = np.percentile(daily_returns, 5)
                    var_99 = np.percentile(daily_returns, 1)
                    cvar_95 = np.mean(daily_returns[daily_returns <= var_95])
                    cvar_99 = np.mean(daily_returns[daily_returns <= var_99])
                    
                    # Display risk metrics
                    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                    with risk_col1:
                        st.metric("Daily VaR (95%)", f"{var_95:.2%}")
                    with risk_col2:
                        st.metric("Daily VaR (99%)", f"{var_99:.2%}")
                    with risk_col3:
                        st.metric("Daily CVaR (95%)", f"{cvar_95:.2%}")
                    with risk_col4:
                        st.metric("Daily CVaR (99%)", f"{cvar_99:.2%}")
                    
                    # Return distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=daily_returns,
                        nbinsx=50,
                        name='Daily Returns',
                        opacity=0.7
                    ))
                    
                    # Add VaR lines
                    fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                                annotation_text="VaR 95%")
                    fig.add_vline(x=var_99, line_dash="dash", line_color="darkred", 
                                annotation_text="VaR 99%")
                    
                    fig.update_layout(
                        title="Portfolio Daily Return Distribution",
                        xaxis_title="Daily Returns",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Drawdown analysis
                    cumulative_returns = np.cumprod(1 + np.mean(portfolio_returns, axis=0))
                    rolling_max = np.maximum.accumulate(cumulative_returns)
                    drawdowns = (cumulative_returns - rolling_max) / rolling_max
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(drawdowns))),
                        y=drawdowns,
                        mode='lines',
                        fill='tozeroy',
                        name='Drawdown'
                    ))
                    
                    fig.update_layout(
                        title="Expected Portfolio Drawdown",
                        xaxis_title="Days",
                        yaxis_title="Drawdown",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Maximum Expected Drawdown", f"{np.min(drawdowns):.2%}")
            
            with port_tab4:
                # Efficient frontier (if available)
                if optimization_method in ['mean_variance', 'multi_objective']:
                    st.info("Efficient frontier analysis available for Mean-Variance and Multi-Objective optimization")
                    
                    # Generate efficient frontier points
                    risk_levels = np.linspace(0.05, 0.3, 20)
                    frontier_returns = []
                    frontier_risks = []
                    
                    for risk in risk_levels:
                        try:
                            temp_constraints = constraints.copy()
                            temp_constraints['target_risk'] = risk
                            
                            temp_result = optimizer.optimize_portfolio(
                                scenarios=scenarios,
                                method='mean_variance',
                                constraints=temp_constraints,
                                asset_names=custom_assets
                            )
                            
                            if temp_result:
                                frontier_returns.append(temp_result.get('expected_return', 0))
                                frontier_risks.append(temp_result.get('volatility', risk))
                        except:
                            continue
                    
                    if frontier_returns and frontier_risks:
                        fig = go.Figure()
                        
                        # Efficient frontier
                        fig.add_trace(go.Scatter(
                            x=frontier_risks,
                            y=frontier_returns,
                            mode='lines+markers',
                            name='Efficient Frontier',
                            line=dict(width=2, color='blue')
                        ))
                        
                        # Current portfolio
                        if 'volatility' in optimization_results and 'expected_return' in optimization_results:
                            fig.add_trace(go.Scatter(
                                x=[optimization_results['volatility']],
                                y=[optimization_results['expected_return']],
                                mode='markers',
                                marker=dict(size=12, color='red'),
                                name='Current Portfolio'
                            ))
                        
                        fig.update_layout(
                            title="Efficient Frontier",
                            xaxis_title="Risk (Volatility)",
                            yaxis_title="Expected Return",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Efficient frontier not available for {optimization_display} method")
            
            # Step 5: Stress Testing (if enabled)
            if enable_stress_testing:
                progress_bar.progress(90)
                status_text.text("🛡️ Performing stress tests...")
                
                st.markdown('<h2 class="sub-header">🛡️ Stress Testing</h2>', unsafe_allow_html=True)
                
                # Stress test scenarios
                stress_scenarios = {
                    'Market Crash (-30%)': -0.30,
                    'Volatility Spike (2x)': 2.0,
                    'Sector Rotation': 0.15,
                    'Liquidity Crisis': -0.15,
                    'Interest Rate Shock': -0.20
                }
                
                stress_results = []
                for scenario_name, shock_magnitude in stress_scenarios.items():
                    if 'Crash' in scenario_name:
                        # Apply uniform negative shock
                        shock_returns = np.full(len(custom_assets), shock_magnitude)
                        portfolio_impact = np.dot(weights, shock_returns)
                    elif 'Volatility' in scenario_name:
                        # Increase volatility
                        base_vol = np.std(returns_data, axis=0)
                        portfolio_vol = np.sqrt(np.dot(weights, np.dot(np.cov(returns_data.T), weights)))
                        portfolio_impact = -(portfolio_vol * shock_magnitude - portfolio_vol)
                    else:
                        # Random sector-specific shock
                        np.random.seed(42)  # For reproducibility
                        shock_returns = np.random.normal(0, abs(shock_magnitude), len(custom_assets))
                        portfolio_impact = np.dot(weights, shock_returns)
                    
                    stress_results.append({
                        'Stress Scenario': scenario_name,
                        'Portfolio Impact': f"{portfolio_impact:.2%}",
                        'Severity': 'High' if abs(portfolio_impact) > 0.10 else 'Medium' if abs(portfolio_impact) > 0.05 else 'Low'
                    })
                
                # Display stress test results
                stress_df = pd.DataFrame(stress_results)
                
                # Color code by severity
                def color_severity(val):
                    if val == 'High':
                        return 'background-color: #ffcccc'
                    elif val == 'Medium':
                        return 'background-color: #fff3cd'
                    else:
                        return 'background-color: #d4edda'
                
                styled_df = stress_df.style.applymap(color_severity, subset=['Severity'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Stress test visualization
                impacts = [float(result['Portfolio Impact'].strip('%')) / 100 for result in stress_results]
                scenarios = [result['Stress Scenario'] for result in stress_results]
                
                fig = go.Figure([go.Bar(
                    x=scenarios,
                    y=impacts,
                    marker_color=['red' if impact < 0 else 'green' for impact in impacts]
                )])
                
                fig.update_layout(
                    title="Stress Test Results",
                    xaxis_title="Stress Scenario",
                    yaxis_title="Portfolio Impact",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Step 6: Export Options (100% progress)
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.markdown('<h2 class="sub-header">💾 Export Results</h2>', unsafe_allow_html=True)
            
            # Export portfolio weights
            if 'weights' in optimization_results:
                weights_df = pd.DataFrame({
                    'Asset': custom_assets,
                    'Name': [ticker_to_name.get(asset, asset) for asset in custom_assets],
                    'Weight': weights,
                    'Weight (%)': [f"{w:.2%}" for w in weights]
                })
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Portfolio Weights",
                        data=weights_df.to_csv(index=False),
                        file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export regime analysis
                    regime_data = []
                    for regime, stats in regime_characteristics.items():
                        regime_data.append({
                            'Regime': regime,
                            'Mean Return': stats.get('mean_return', 0),
                            'Volatility': stats.get('volatility', 0),
                            'Persistence': stats.get('persistence', 0),
                            'Avg Duration': stats.get('avg_duration', 0)
                        })
                    
                    regime_df = pd.DataFrame(regime_data)
                    st.download_button(
                        label="📥 Download Regime Analysis",
                        data=regime_df.to_csv(index=False),
                        file_name=f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            # Summary report
            with st.expander("📋 Analysis Summary Report"):
                st.markdown(f"""
                **Portfolio Analysis Summary**
                
                **Data:**
                - Assets: {len(custom_assets)} ({', '.join(custom_assets[:3])}{'...' if len(custom_assets) > 3 else ''})
                - Period: {lookback_period} ({len(price_data)} trading days)
                - Exchange: {exchange}
                
                **Regime Analysis:**
                - Method: {regime_method}
                - Regimes Detected: {len(np.unique(regime_states))}
                - Current Regime: {current_regime_label}
                
                **Monte Carlo:**
                - Simulations: {n_simulations:,}
                - Horizon: {simulation_horizon} days
                - Method: {simulation_display}
                
                **Optimization:**
                - Method: {optimization_display}
                - Max Position: {max_position_size:.1%}
                - Min Position: {min_position_size:.1%}
                
                **Risk Metrics:**
                - VaR Confidence: {var_confidence:.1%}
                - Stress Testing: {'Enabled' if enable_stress_testing else 'Disabled'}
                """)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Debug information
            if show_debug_info:
                with st.expander("🔧 Debug Information"):
                    st.write("**Optimization Results Keys:**", list(optimization_results.keys()) if 'optimization_results' in locals() and optimization_results else "None")
                    st.write("**Scenarios Keys:**", list(scenarios.keys()) if isinstance(scenarios, dict) else f"Invalid type: {type(scenarios)}")
                    st.write("**Regime States Shape:**", regime_states_aligned.shape if regime_states_aligned is not None else "None")
                    st.write("**Returns Data Shape:**", returns_data.shape)
                    st.write("**Price Data Shape:**", price_data.shape)
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            if show_debug_info:
                st.exception(e)
            logging.error("Analysis pipeline error:", exc_info=True)
            
    else:
        # Welcome screen when analysis is not running
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   border-radius: 10px; color: white; margin: 2rem 0;">
            <h2>🚀 Ready to Optimize Your Portfolio?</h2>
            <p>Configure your parameters in the sidebar and click 'Run Complete Analysis' to begin!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🔍 Regime Detection</h4>
                <ul>
                    <li>Hidden Markov Models</li>
                    <li>Transition Matrices</li>
                    <li>Regime Characteristics</li>
                    <li>Persistence Analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🎲 Monte Carlo Simulation</h4>
                <ul>
                    <li>Multiple Distribution Types</li>
                    <li>Regime-Conditioned Paths</li>
                    <li>Fat-tailed Modeling</li>
                    <li>Bootstrap Methods</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>⚖️ Portfolio Optimization</h4>
                <ul>
                    <li>CVaR Minimization</li>
                    <li>Multi-Objective Optimization</li>
                    <li>Risk Budgeting</li>
                    <li>Comprehensive Stress Testing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
