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
from scipy.stats import norm, t, rankdata, jarque_bera, kstest
import cvxpy as cp
from scipy.optimize import minimize
import itertools
import time
import warnings
import io
import base64
from typing import Dict, List, Tuple, Optional
import logging

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
            
            for t in range(horizon):
                regime = regime_path[t]
                
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
                        scenarios[sim, t, i] = t.ppf(uniform_samples[i], df, loc=params[1], scale=params[2])
                except Exception as e:
                    logging.warning(f"Copula generation failed for regime {regime} at sim {sim}, time {t}: {e}. Falling back to Gaussian.")
                    # Fallback to Gaussian if copula fails
                    mu = regime_data[regime]['mean'].values
                    cov = regime_data[regime]['cov'].values
                    scenarios[sim, t, :] = np.random.multivariate_normal(mu, cov)
        
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
def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Regime-Aware Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Features
    - **Advanced Regime Detection**: HMM, GMM, and hybrid approaches
    - **Multiple Optimization Methods**: CVaR, Mean-Variance, Risk Parity, Multi-Objective
    - **Comprehensive Risk Management**: VaR, CVaR, Drawdown, Stress Testing
    - **Rich Visualizations**: Interactive dashboards and analytics
    - **Multiple Asset Classes**: Stocks, ETFs, Cryptocurrencies, Global Markets
    """)
    
    # Initialize components
    data_manager = DataManager()
    regime_detector = RegimeDetector()
    mc_engine = MonteCarloEngine()
    optimizer = PortfolioOptimizer()
    dashboard = DashboardGenerator()
    
    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration Panel")
    st.sidebar.subheader("📊 Data Configuration")
    exchange = st.sidebar.selectbox("Select Exchange/Universe", ["NSE", "NASDAQ", "Global_ETFs", "Crypto", "Custom"])
    
    custom_assets = []
    if exchange == "Custom":
        asset_input = st.sidebar.text_area("Enter asset symbols (one per line)", value="AAPL\nMSFT\nGOOGL\nAMZN\nSPY\n^NSEI")
        custom_assets = [sym.strip().upper() for sym in asset_input.split('\n') if sym.strip()]
    else:
        predefined_assets = data_manager.get_asset_universe(exchange)
        default_selection = predefined_assets[:min(5, len(predefined_assets))] if predefined_assets else []
        selected_assets = st.sidebar.multiselect(f"Select assets from {exchange}", predefined_assets, default=default_selection)
        custom_assets = selected_assets
    
    if not custom_assets:
        st.sidebar.warning("Please select or enter at least one asset symbol.")
        run_analysis = False
    else:
        # Time Period Selection
        st.sidebar.subheader("📅 Time Period")
        lookback_period = st.sidebar.selectbox("Historical Data Period", ["1 Year", "2 Years", "3 Years", "5 Years", "10 Years"], index=2)
        period_map = {"1 Year": 365, "2 Years": 730, "3 Years": 1095, "5 Years": 1825, "10 Years": 3650}
        start_date = (datetime.now() - timedelta(days=period_map[lookback_period])).strftime('%Y-%m-%d')
        
        # Regime Detection Settings
        st.sidebar.subheader("🔍 Regime Detection")
        n_regimes = st.sidebar.slider("Number of Market Regimes", 2, 5, 3)
        regime_method = st.sidebar.selectbox("Detection Method", ["HMM (Recommended)", "HMM + GMM Comparison"])
        
        # Monte Carlo Settings
        st.sidebar.subheader("🎲 Monte Carlo Simulation")
        n_simulations = st.sidebar.slider("Number of Simulations", 500, 10000, 2000, step=500)
        simulation_horizon = st.sidebar.slider("Horizon (Trading Days)", 21, 252, 126, step=21)
        simulation_display = st.sidebar.selectbox("Simulation Method", ["Gaussian", "Copula (Fat-tailed)", "Bootstrap"])

        # Map display labels to internal method names
        simulation_method_map = {
            "Gaussian": "gaussian",
            "Copula (Fat-tailed)": "copula",
            "Bootstrap": "bootstrap"
        }
        simulation_method = simulation_method_map.get(simulation_display)
        if simulation_method is None:
            st.error(f"Invalid simulation method selected: {simulation_display}")
            logging.error(f"Invalid simulation method: {simulation_display}")
            return
        
        # Optimization Settings
        st.sidebar.subheader("⚖️ Portfolio Optimization")
        optimization_display = st.sidebar.selectbox(
            "Optimization Method", 
            ["CVaR (Recommended)", "Mean-Variance", "Risk Parity", "Multi-Objective"]
        )
  
        # Map display labels to internal method names
        optimization_method_map = {
            "CVaR (Recommended)": "cvar",
            "Mean-Variance": "mean_variance",
            "Risk Parity": "risk_parity",
            "Multi-Objective": "multi_objective"
        }

        optimization_method = optimization_method_map[optimization_display]

        # Risk Management Settings
        st.sidebar.subheader("🛡️ Risk Management")
        var_confidence = st.sidebar.slider("VaR/CVaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
        max_position_size = st.sidebar.slider("Max Position Size (per asset)", 0.1, 1.0, 0.3, 0.05)
        min_position_size = st.sidebar.slider("Min Position Size (per asset)", 0.0, 0.1, 0.01, 0.01)
        
        # Advanced Settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            enable_sector_constraints = st.checkbox("Enable Sector Constraints (Conceptual)", False)
            enable_turnover_control = st.checkbox("Enable Turnover Control (Conceptual)", False)
            enable_stress_testing = st.checkbox("Enable Stress Testing", True)
            use_transaction_costs = st.checkbox("Include Transaction Costs (Conceptual)", False)
        
        run_analysis = st.sidebar.button("🚀 Run Complete Analysis", type="primary")
    
    # Main Content Area
    if not run_analysis:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            #### 🔍 Regime Detection
            - Hidden Markov Models
            - Transition Matrices
            - Regime Characteristics
            - Persistence Analysis
            """)
        with col2:
            st.markdown("""
            #### 🎲 Scenario Generation
            - Monte Carlo Simulation
            - Fat-tailed Distributions
            - Regime-Conditioned Paths
            - Bootstrap Methods
            """)
        with col3:
            st.markdown("""
            #### ⚖️ Optimization
            - CVaR Minimization
            - Multi-Objective
            - Risk Budgeting
            - Stress Testing
            """)
        st.info("👆 Configure your parameters in the sidebar and click 'Run Complete Analysis' to begin!")
        return


    ticker_to_name = {
        # 🇮🇳 NSE
        "RELIANCE.NS": "Reliance Industries",
        "HDFCBANK.NS": "HDFC Bank",
        "INFY.NS": "Infosys",
        "TCS.NS": "Tata Consultancy Services",
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel",
        "ITC.NS": "ITC Ltd.",
        "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "WIPRO.NS": "Wipro Ltd.",
    
        # 🇺🇸 NASDAQ
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "NVDA": "NVIDIA Corporation",

        # 🌍 Global ETFs
        "SPY": "SPDR S&P 500 ETF",
        "EFA": "iShares MSCI EAFE ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "VNQ": "Vanguard Real Estate ETF",
        "GLD": "SPDR Gold Trust",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "VIX": "CBOE Volatility Index",

        # ₿ Crypto
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "ADA-USD": "Cardano",
        "BNB-USD": "Binance Coin"
    }

 


    # Analysis Pipeline
    if run_analysis and custom_assets:
        try:
            with st.spinner("🚀 Running comprehensive analysis..."):
                
                # Step 1: Data Loading
                st.info("📊 Step 1: Loading and validating market data...")
                price_data = data_manager.load_data_with_validation(custom_assets, start_date)
                                
                if price_data.empty:
                    st.error("❌ Failed to load data. Please check your asset symbols and try again.")
                    return
                
                returns_data = price_data.pct_change().dropna()
                
                if returns_data.empty:
                    st.error("❌ No valid return data after processing. Check data period or assets.")
                    return
                
                st.success(f"✅ Loaded {len(price_data.columns)} assets with {len(price_data)} trading days.")
                # 📉 Display historical price charts
                st.subheader("📉 Historical Price Charts of Selected Stocks")

                with st.expander("📂 Show Price Charts"):
                    for ticker in price_data.columns:
                        series = pd.to_numeric(price_data[ticker], errors='coerce')
                        series.index = pd.to_datetime(series.index, errors='coerce')
                        series = series[series.index.notna()]
                        # ✅ Get display name
                        display_name = ticker_to_name.get(ticker, ticker)
                        if series.isna().all():
                            st.warning(f"⚠️ No plottable data for {ticker}")
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode='lines',
                                name=display_name,
                                line=dict(color='rgba(100, 100, 255, 0.5)', width=2)
                        ))
                            fig.update_layout(
                                title=f"Price Chart - {display_name}",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                
                # Step 2: Regime Detection
                st.info("🔍 Step 2: Detecting market regimes...")
                primary_asset_returns = returns_data.iloc[:, 0].copy()
                if not isinstance(primary_asset_returns, pd.Series):
                    logging.error(f"primary_asset_returns is not a pandas Series, got {type(primary_asset_returns)}")
                    st.error("❌ Internal error: primary_asset_returns is not a pandas Series")
                    return
                
                logging.info(f"primary_asset_returns type: {type(primary_asset_returns)}, shape: {primary_asset_returns.shape}")
                
                features = regime_detector.prepare_features(primary_asset_returns)

                if regime_method == "HMM (Recommended)":
                    regime_states, hmm_model = regime_detector.fit_hmm(features, n_states=n_regimes)
                else:
                    regime_states, hmm_model = regime_detector.fit_hmm(features, n_states=n_regimes)
                    gmm_states = regime_detector.fit_gmm_comparison(features, n_states=n_regimes)

                if regime_states is None or hmm_model is None:
                    st.error("❌ Regime detection failed. Check data or reduce number of regimes.")
                    return

                # Align all data to feature index
                aligned_index = features.index
                returns_aligned = primary_asset_returns.loc[aligned_index]
                features_aligned = features
                regime_states_aligned = pd.Series(regime_states, index=aligned_index)

                regime_characteristics = regime_detector.analyze_regime_characteristics(
                    returns_aligned, regime_states_aligned.values, features_aligned
                )

                # 🧭 Display Current Market Regime
                current_regime_state = regime_states_aligned.values[-1]
                regime_name = regime_detector.regime_label_map.get(current_regime_state, f"Regime {current_regime_state}")
                st.markdown(f"### 🧭 Current Market Regime: `{regime_name}`")


                if not regime_characteristics:
                    st.error("❌ Failed to analyze regime characteristics. Check data alignment.")
                    return

                st.success(f"✅ Detected {len(np.unique(regime_states))} market regimes.")
                
                # Display regime analysis
                st.markdown('<h2 class="sub-header">🔍 Market Regime Analysis</h2>', unsafe_allow_html=True)
                dashboard.create_regime_analysis_dashboard(
                    returns_aligned.to_frame(),
                    regime_states_aligned.values,
                    regime_characteristics,
                    features_aligned,
                    regime_detector.regime_label_map  # Pass regime_label_map
                )

                # Step 3: Monte Carlo Simulation
                st.info("🎲 Step 3: Generating Monte Carlo scenarios...")
                current_regime = regime_states_aligned.values[-1]

                # Log current regime with descriptive label
                if not regime_detector.regime_label_map:
                    logging.warning("regime_label_map is empty. Falling back to numerical regime.")
                    current_regime_label = f"Regime {current_regime}"
                else:
                    current_regime_label = regime_detector.regime_label_map.get(current_regime, f"Regime {current_regime}")

                st.info(f"Generating scenarios starting in {current_regime_label}")

                scenarios = mc_engine.generate_regime_scenarios(
                    returns=returns_data.loc[aligned_index],
                    regime_states=regime_states_aligned.values,
                    current_regime=current_regime,
                    n_simulations=n_simulations,
                    horizon=simulation_horizon,
                    method=simulation_method.lower(),
                    regime_label_map=regime_detector.regime_label_map
                )

                if not scenarios:
                    st.error("❌ Monte Carlo simulation failed. Check regime data or simulation parameters.")
                    return

                st.success(f"✅ Generated {n_simulations} scenarios over {simulation_horizon} days.")

                # Step 4: Portfolio Optimization
                st.info("⚖️ Step 4: Optimizing portfolio...")
                asset_names = returns_data.columns.tolist()
        
                # 🚫 Enforce minimum asset count for CVaR
                if optimization_method.lower().startswith("cvar") and len(asset_names) < 3:
                    st.error("❌ CVaR optimization requires at least 3 assets. Please select more assets or use a different method.")
                    return
                num_assets = len(asset_names)

                if num_assets < 5:
                    min_position_size, max_position_size = 0.0, 0.6
                elif num_assets <= 10:
                    min_position_size, max_position_size = 0.01, 0.3
                else:
                    min_position_size, max_position_size = 0.005, 0.2
                    
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
                    asset_names=asset_names
                )

                if not optimization_results:
                    st.error("❌ Portfolio optimization failed.")
                    return

                st.success("✅ Portfolio optimized successfully.")

                # Display optimization dashboard
                st.markdown('<h2 class="sub-header">📊 Portfolio Optimization Results</h2>', unsafe_allow_html=True)
                dashboard.create_portfolio_dashboard(
                    optimization_results=optimization_results,
                    scenarios=scenarios,
                    asset_names=asset_names
                )
                # ✅ Plot the expected future portfolio path (mean)
                if scenarios and 'returns' in scenarios and 'weights' in optimization_results:
                    sim_returns = scenarios['returns']  # shape: [n_sim, horizon, n_assets]
                    weights = optimization_results['weights']
                    portfolio_paths = np.cumprod(1 + np.dot(sim_returns, weights), axis=1)  # shape: [n_sim, horizon]
                    mean_path = portfolio_paths.mean(axis=0)
                    lower_bound = np.percentile(portfolio_paths, 5, axis=0)
                    upper_bound = np.percentile(portfolio_paths, 99, axis=0)

                    st.subheader("📈 Expected Future Portfolio Value ")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=mean_path,
                        mode='lines',
                        line=dict(width=3, color='blue'),
                        name='Mean Portfolio Path'
                    ))
                    # Upper bound
                    fig.add_trace(go.Scatter(
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        name='95th Percentile',
                        showlegend=False
                    ))
                    # Lower bound + fill between bounds
                    fig.add_trace(go.Scatter(
                        y=lower_bound,
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(0, 100, 250, 0.2)',
                        line=dict(width=0),
                        name='5th–95th Percentile',
                        showlegend=True
                    ))
                    fig.update_layout(
                        title="Projected Portfolio Value Over Time",
                        xaxis_title="Days into the Future",
                        yaxis_title="Portfolio Value",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)



                # Step 5: Stress Testing
                if enable_stress_testing:
                    dashboard.perform_stress_testing(
                        returns_data.loc[aligned_index],
                        regime_states_aligned.values,
                        optimization_results['weights'],
                        regime_detector.regime_label_map  # Pass regime_label_map
                    )

                # Export Options
                dashboard._provide_export_options(
                    optimization_results=optimization_results,
                    regime_characteristics=regime_characteristics,
                    asset_names=asset_names
                )
                
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            logging.error("Analysis pipeline error:", exc_info=True)
    

def perform_stress_testing(self, scenarios: Dict, weights: np.ndarray, asset_names: List[str]):
    """Perform stress testing on the portfolio"""
    if weights is None or scenarios.get('returns') is None:
        st.warning("Cannot perform stress testing: Missing weights or scenarios.")
        return
    
    returns_array = scenarios['returns']
    n_sim, horizon, n_assets = returns_array.shape
    
    # Calculate portfolio returns
    portfolio_returns = (1 + returns_array).prod(axis=1) @ weights - 1
    
    # Stress scenarios
    stress_scenarios = {
        'Market Crash (-30%)': np.full(n_assets, -0.30),
        'Volatility Spike (2x)': np.std(returns_array, axis=1) * 2,
        'Liquidity Shock': np.random.uniform(-0.1, -0.05, n_assets)
    }
    
    stress_results = []
    for name, shock in stress_scenarios.items():
        if 'Crash' in name or 'Liquidity' in name:
            stress_return = shock @ weights
        else:  # Volatility spike
            stress_return = np.mean(portfolio_returns) - shock @ weights
        
        stress_results.append({
            'Scenario': name,
            'Portfolio Impact': f"{stress_return:.2%}"
        })
    
    st.subheader("Stress Test Outcomes")
    st.dataframe(pd.DataFrame(stress_results), use_container_width=True)

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

if __name__ == "__main__":
    main()
