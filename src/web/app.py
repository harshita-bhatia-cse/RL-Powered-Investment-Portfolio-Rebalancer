"""
FinRL Trading Dashboard
======================

Main Streamlit application for the FinRL Trading platform.
Provides interactive visualization and control of trading strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Import project modules
from src.config.settings import get_config
from src.data.data_store import get_data_store
from src.strategies.base_strategy import create_strategy, StrategyConfig
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig
from src.trading.alpaca_manager import create_alpaca_account_from_env
from src.trading.trade_executor import TradeExecutor, ExecutionConfig

# Setup logging
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="FinRL Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = get_config()
if 'data_store' not in st.session_state:
    st.session_state.data_store = get_data_store()


def main():
    """Main application function."""
    st.title("📈 FinRL Trading Dashboard")
    st.markdown("AI-powered quantitative trading platform")

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Overview", "Data Management", "Strategy Backtesting",
             "Live Trading", "Portfolio Analysis", "Settings"]
        )

        st.divider()

        # Quick stats
        display_quick_stats()

    # Main content
    if page == "Overview":
        show_overview()
    elif page == "Data Management":
        show_data_management()
    elif page == "Strategy Backtesting":
        show_strategy_backtesting()
    elif page == "Live Trading":
        show_live_trading()
    elif page == "Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "Settings":
        show_settings()


def display_quick_stats():
    """Display quick statistics in sidebar."""
    st.subheader("Quick Stats")

    try:
        # Get data store stats
        stats = st.session_state.data_store.get_storage_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Price Records", stats.get('price_records', 0))
        with col2:
            st.metric("Stored Files", stats.get('total_files', 0))

        st.metric("Storage Used", f"{stats.get('total_size_mb', 0):.1f} MB")

    except Exception as e:
        st.error(f"Could not load stats: {e}")


def _load_latest_backtest_equity():
    """Load the most recently generated adaptive-rotation equity curve."""
    project_root = Path(__file__).resolve().parents[2]
    reports = list(
        (project_root / "src" / "strategies" / "output" / "weights" / "adaptive_rotation")
        .glob("backtest_equity_*.csv")
    )
    if not reports:
        return None, None

    report_path = max(reports, key=lambda path: path.stat().st_mtime)
    equity = pd.read_csv(report_path, parse_dates=["date"])
    required_columns = {"date", "portfolio"}
    missing_columns = required_columns - set(equity.columns)
    if missing_columns:
        raise ValueError(
            f"Backtest report {report_path.name} is missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    numeric_columns = [
        column for column in ("portfolio", "SPY", "QQQ") if column in equity.columns
    ]
    for column in numeric_columns:
        equity[column] = pd.to_numeric(equity[column], errors="coerce")

    equity = (
        equity.dropna(subset=["date", "portfolio"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    if len(equity) < 2 or (equity["portfolio"] <= 0).any():
        raise ValueError(f"Backtest report {report_path.name} has insufficient valid equity data")

    return equity, report_path


def _calculate_backtest_metrics(equity):
    """Calculate summary statistics from a daily growth-of-one equity curve."""
    portfolio = equity["portfolio"]
    daily_returns = portfolio.pct_change().dropna()
    years = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25
    annualized_return = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / years) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (
        daily_returns.mean() * 252 / annualized_volatility
        if annualized_volatility > 0 else 0.0
    )
    max_drawdown = (portfolio / portfolio.cummax() - 1).min()
    return {
        "cumulative_return": portfolio.iloc[-1] / portfolio.iloc[0] - 1,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def show_overview():
    """Show overview dashboard."""
    st.header("Trading Overview")

    try:
        equity, report_path = _load_latest_backtest_equity()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load the saved backtest report: {error}")
        return

    if equity is None:
        st.warning(
            "No backtest equity report found. Run the adaptive-rotation backtest to "
            "populate this dashboard."
        )
        return

    metrics = _calculate_backtest_metrics(equity)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cumulative Return", f"{metrics['cumulative_return']:.2%}")
    col2.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    col3.metric("Sharpe Ratio (0% risk-free)", f"{metrics['sharpe']:.2f}")
    col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    st.caption(
        f"Adaptive Rotation backtest · {equity['date'].iloc[0]:%Y-%m-%d} to "
        f"{equity['date'].iloc[-1]:%Y-%m-%d} · {report_path.name}. "
        "Historical simulation, not live portfolio performance."
    )

    chart_columns = ["portfolio"] + [
        benchmark for benchmark in ("SPY", "QQQ") if benchmark in equity.columns
    ]
    chart = equity.melt(
        id_vars="date",
        value_vars=chart_columns,
        var_name="Series",
        value_name="Growth of $1",
    )
    chart["Series"] = chart["Series"].replace({"portfolio": "Adaptive Rotation"})
    fig = px.line(
        chart,
        x="date",
        y="Growth of $1",
        color="Series",
        title="Backtest Performance vs. Benchmarks",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)


def show_data_management():
    """Show data management interface."""
    st.header("Data Management")

    tab1, tab2, tab3, tab4 = st.tabs(["Data Sources", "Data Processing", "Data Storage", "Data Quality"])

    with tab1:
        st.subheader("Data Sources")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("WRDS Data")
            if st.button("Fetch S&P 500 Components"):
                with st.spinner("Fetching data..."):
                    try:
                        from src.data.data_fetcher import fetch_sp500_tickers
                        tickers = fetch_sp500_tickers()
                        st.success(f"Successfully fetched {len(tickers)} tickers")
                        st.info(f"Sample tickers: {tickers[:10]}")
                    except Exception as e:
                        st.error(f"Failed to fetch data: {e}")

            if st.button("Fetch Fundamental Data"):
                with st.spinner("Fetching fundamental data..."):
                    try:
                        from src.data.data_fetcher import fetch_fundamental_data
                        fundamentals = fetch_fundamental_data(
                            ['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2023-12-31'
                        )
                        st.success(f"Successfully fetched {len(fundamentals)} records")
                    except Exception as e:
                        st.error(f"Failed to fetch data: {e}")

        with col2:
            st.subheader("Local Data")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(df)} rows")
                st.dataframe(df.head())

    with tab2:
        st.subheader("Data Processing")

        if st.button("Process Raw Data"):
            with st.spinner("Processing data..."):
                try:
                    from src.data.data_processor import process_fundamentals, process_prices

                    # Process sample data
                    fundamentals = process_fundamentals("./data/fundamentals.csv")
                    prices = process_prices("./data/prices.csv")

                    st.success("Data processing completed")
                    st.write(f"Processed {len(fundamentals)} fundamental records")
                    st.write(f"Processed {len(prices)} price records")

                except Exception as e:
                    st.error(f"Data processing failed: {e}")

        if st.button("Generate ML Dataset"):
            with st.spinner("Creating ML dataset..."):
                try:
                    from src.data.data_processor import create_ml_dataset

                    X, y = create_ml_dataset("./data/fundamentals.csv", "./data/prices.csv")
                    st.success("ML dataset created")
                    st.write(f"Features shape: {X.shape}")
                    st.write(f"Target shape: {y.shape}")

                except Exception as e:
                    st.error(f"ML dataset creation failed: {e}")

    with tab3:
        st.subheader("Data Storage")

        # Display storage stats
        stats = st.session_state.data_store.get_storage_stats()
        st.json(stats)

        if st.button("Cleanup Expired Cache"):
            with st.spinner("Cleaning up cache..."):
                try:
                    st.session_state.data_store.cleanup_expired_cache()
                    st.success("Cache cleanup completed")
                except Exception as e:
                    st.error(f"Cache cleanup failed: {e}")

    with tab4:
        st.subheader("Data Quality")

        # Data quality checks
        st.subheader("Data Quality Metrics")

        # Sample quality metrics
        quality_data = pd.DataFrame({
            'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
            'Score': [95.2, 98.1, 92.3, 99.8],
            'Status': ['Good', 'Excellent', 'Good', 'Excellent']
        })

        st.dataframe(quality_data, use_container_width=True)


def show_strategy_backtesting():
    """Show strategy backtesting interface."""
    st.header("Strategy Backtesting")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Backtest Configuration")

        # Strategy selection
        strategy_type = st.selectbox(
            "Strategy Type",
            ["equal_weight", "market_cap_weight", "ml_strategy"]
        )

        # Backtest parameters
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
        end_date = st.date_input("End Date", datetime(2023, 12, 31))
        initial_capital = st.number_input("Initial Capital", value=1000000, step=100000)

        # ML strategy parameters
        if strategy_type == "ml_strategy":
            top_quantile = st.slider("Top Quantile", 0.5, 1.0, 0.75, 0.05)

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    run_backtest(
                        strategy_type=strategy_type,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        top_quantile=top_quantile if strategy_type == "ml_strategy" else 0.75
                    )
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

    with col2:
        st.subheader("Backtest Results")

        # Display results if available
        if 'backtest_result' in st.session_state:
            result = st.session_state.backtest_result

            # Key metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Final Value", ".2f")
            with metrics_cols[1]:
                st.metric("Total Return", ".2%")
            with metrics_cols[2]:
                st.metric("Annual Return", ".2%")
            with metrics_cols[3]:
                st.metric("Sharpe Ratio", ".2f")

            # Performance chart
            if hasattr(result, 'portfolio_values'):
                fig = px.line(
                    x=result.portfolio_values.index,
                    y=result.portfolio_values.values,
                    title="Portfolio Value"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics
            st.subheader("Detailed Metrics")
            metrics_df = pd.DataFrame({
                'Metric': list(result.metrics.keys()),
                'Value': [".4f" for v in result.metrics.values()]
            })
            st.dataframe(metrics_df)


def run_backtest(strategy_type, start_date, end_date, initial_capital, top_quantile):
    """Run backtest with given parameters."""
    # Create strategy
    config = StrategyConfig(name=f"{strategy_type} Backtest")
    strategy = create_strategy(strategy_type, config)

    # Create backtest configuration
    backtest_config = BacktestConfig(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=initial_capital
    )

    # Load sample data (in practice, load real data)
    dates = pd.date_range(start_date, end_date, freq='d')
    price_data = pd.DataFrame({
        'datadate': dates,
        'adj_close': 100 + np.cumsum(np.random.normal(0, 0.02, len(dates)))
    })

    # Sample weight signals
    weight_signals = pd.DataFrame({
        'date': pd.date_range(start_date, end_date, freq='QE'),
        'AAPL': 0.5,
        'MSFT': 0.3,
        'GOOGL': 0.2
    })

    # Run backtest
    engine = BacktestEngine(backtest_config)
    result = engine.run_backtest(strategy, price_data, weight_signals)

    # Store result
    st.session_state.backtest_result = result

    st.success("Backtest completed successfully!")


def show_live_trading():
    """Show live trading interface."""
    st.header("Live Trading")

    # Check if trading is configured
    try:
        account = create_alpaca_account_from_env()
        st.success(f"Connected to Alpaca account (Paper: {account.is_paper})")

        tab1, tab2, tab3 = st.tabs(["Portfolio", "Order Management", "Strategy Execution"])

        with tab1:
            st.subheader("Current Portfolio")

            if st.button("Refresh Portfolio"):
                with st.spinner("Loading portfolio..."):
                    try:
                        from src.trading.alpaca_manager import AlpacaManager
                        manager = AlpacaManager([account])

                        # Get account info
                        account_info = manager.get_account_info()
                        positions = manager.get_positions()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Portfolio Value", ".2f")
                        with col2:
                            st.metric("Cash", ".2f")
                        with col3:
                            st.metric("Buying Power", ".2f")

                        # Positions table
                        if positions:
                            positions_df = pd.DataFrame(positions)
                            st.dataframe(positions_df[['symbol', 'qty', 'avg_entry_price', 'market_value', 'unrealized_pl']], use_container_width=True)
                        else:
                            st.info("No open positions")

                    except Exception as e:
                        st.error(f"Failed to load portfolio: {e}")

        with tab2:
            st.subheader("Order Management")

            # Place order form
            with st.form("place_order"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    symbol = st.text_input("Symbol", "AAPL").upper()
                with col2:
                    quantity = st.number_input("Quantity", min_value=1, value=10)
                with col3:
                    side = st.selectbox("Side", ["buy", "sell"])

                order_type = st.selectbox("Order Type", ["market", "limit"])
                limit_price = st.number_input("Limit Price", min_value=0.01, step=0.01) if order_type == "limit" else None

                submitted = st.form_submit_button("Place Order")
                if submitted:
                    try:
                        from src.trading.alpaca_manager import AlpacaManager, OrderRequest
                        manager = AlpacaManager([account])

                        order = OrderRequest(
                            symbol=symbol,
                            quantity=quantity,
                            side=side,
                            order_type=order_type,
                            limit_price=limit_price
                        )

                        response = manager.place_order(order)
                        st.success(f"Order placed: {response.order_id}")

                    except Exception as e:
                        st.error(f"Failed to place order: {e}")

        with tab3:
            st.subheader("Strategy Execution")

            # Strategy execution
            if st.button("Execute Sample Strategy"):
                with st.spinner("Executing strategy..."):
                    try:
                        from src.trading.trade_executor import TradeExecutor
                        from src.strategies.base_strategy import StrategyConfig, EqualWeightStrategy

                        manager = AlpacaManager([account])
                        executor = TradeExecutor(manager)

                        # Create sample strategy
                        config = StrategyConfig(name="Sample Equal Weight")
                        strategy = EqualWeightStrategy(config)

                        # Sample data
                        sample_data = {
                            'fundamentals': pd.DataFrame({
                                'gvkey': ['AAPL', 'MSFT', 'GOOGL'],
                                'datadate': ['2024-01-01'] * 3
                            })
                        }

                        result = executor.execute_strategy(strategy, sample_data)
                        st.success(f"Strategy executed: {len(result.orders_placed)} orders placed")

                    except Exception as e:
                        st.error(f"Strategy execution failed: {e}")

    except Exception as e:
        st.error(f"Trading not configured: {e}")
        st.info("Please set up Alpaca API credentials in environment variables")


def show_portfolio_analysis():
    """Show portfolio analysis interface."""
    st.header("Portfolio Analysis")

    # Sample portfolio data
    dates = pd.date_range('2024-01-01', periods=100, freq='d')
    portfolio_values = 1000000 + np.cumsum(np.random.normal(2000, 8000, 100))

    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Analysis", "Attribution", "Benchmarking"])

    with tab1:
        st.subheader("Performance Analysis")

        # Performance chart
        fig = px.line(x=dates, y=portfolio_values, title="Portfolio Performance")
        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{total_return:.2%}")
        with col2:
            st.metric("Annual Return", f"{annual_return:.2%}")
        with col3:
            st.metric("Volatility", f"{volatility:.2%}")
        with col4:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    with tab2:
        st.subheader("Risk Analysis")

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig = px.line(x=dates[1:], y=drawdown, title="Portfolio Drawdown")
        fig.update_layout(yaxis_title="Drawdown", yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

        # Risk metrics
        max_drawdown = drawdown.min()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        with col2:
            st.metric("VaR (95%)", f"{var_95:.2%}")
        with col3:
            st.metric("CVaR (95%)", f"{cvar_95:.2%}")

    with tab3:
        st.subheader("Attribution Analysis")

        # Sample attribution data
        attribution_data = pd.DataFrame({
            'Asset': ['AAPL', 'MSFT', 'GOOGL', 'Bonds', 'Cash'],
            'Weight': [0.3, 0.25, 0.2, 0.15, 0.1],
            'Return': [0.15, 0.12, 0.18, 0.03, 0.02],
            'Contribution': [0.045, 0.03, 0.036, 0.0045, 0.002]
        })

        fig = px.bar(attribution_data, x='Asset', y='Contribution',
                    title="Return Attribution by Asset")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(attribution_data, use_container_width=True)

    with tab4:
        st.subheader("Benchmarking")

        # Sample benchmark comparison
        benchmark_data = pd.DataFrame({
            'Date': dates,
            'Portfolio': portfolio_values,
            'SPY': 1000000 + np.cumsum(np.random.normal(1500, 6000, 100)),
            'QQQ': 1000000 + np.cumsum(np.random.normal(1800, 7000, 100))
        })

        fig = px.line(benchmark_data, x='Date', y=['Portfolio', 'SPY', 'QQQ'],
                     title="Portfolio vs Benchmarks")
        st.plotly_chart(fig, use_container_width=True)


def show_settings():
    """Show settings interface."""
    st.header("Settings")

    tab1, tab2, tab3 = st.tabs(["General", "Trading", "Data"])

    with tab1:
        st.subheader("General Settings")

        # Logging level
        log_level = st.selectbox("Logging Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        if st.button("Apply Logging Level"):
            logging.getLogger().setLevel(getattr(logging, log_level))
            st.success(f"Logging level set to {log_level}")

        # Theme
        theme = st.selectbox("Theme", ["Light", "Dark"])
        if st.button("Apply Theme"):
            st.success(f"Theme set to {theme}")

    with tab2:
        st.subheader("Trading Settings")

        # Risk limits
        max_order_value = st.number_input("Max Order Value ($)", value=100000, step=10000)
        max_portfolio_turnover = st.slider("Max Portfolio Turnover (%)", 0.0, 1.0, 0.5, 0.05)

        if st.button("Save Trading Settings"):
            st.success("Trading settings saved")

        # API Configuration
        st.subheader("API Configuration")
        api_key = st.text_input("Alpaca API Key", type="password")
        api_secret = st.text_input("Alpaca API Secret", type="password")
        use_paper = st.checkbox("Use Paper Trading", value=True)

        if st.button("Save API Settings"):
            st.success("API settings saved")

    with tab3:
        st.subheader("Data Settings")

        # Data paths
        data_dir = st.text_input("Data Directory", value="./data")
        cache_dir = st.text_input("Cache Directory", value="./data/cache")

        if st.button("Save Data Settings"):
            st.success("Data settings saved")

        # Data sources
        st.subheader("Data Sources")
        enable_wrds = st.checkbox("Enable WRDS", value=True)
        enable_alpha_vantage = st.checkbox("Enable Alpha Vantage", value=False)

        if st.button("Save Data Source Settings"):
            st.success("Data source settings saved")


if __name__ == "__main__":
    main()
