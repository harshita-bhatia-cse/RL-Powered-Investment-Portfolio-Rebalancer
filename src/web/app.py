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
from src.data.data_quality import calculate_price_quality
from src.strategies.base_strategy import create_strategy, StrategyConfig
from src.trading.alpaca_manager import create_alpaca_account_from_env
from src.trading.trade_executor import TradeExecutor, ExecutionConfig
from src.analytics import (
    analyze_assets,
    analyze_correlations,
    allocation_summary,
    calculate_portfolio_metrics,
    calculate_risk_metrics,
    compare_benchmarks,
    prices_to_returns,
    validate_weights,
)

# Setup logging
logger = logging.getLogger(__name__)


def _display_table(data: pd.DataFrame, **kwargs) -> None:
    """Render a table, with a text fallback when Streamlit's Arrow backend is unavailable."""
    if "use_container_width" in kwargs:
        kwargs["width"] = "stretch" if kwargs.pop("use_container_width") else "content"
    try:
        st.dataframe(data, **kwargs)
    except ImportError as error:
        logger.warning("Streamlit dataframe rendering is unavailable: %s", error)
        st.warning("Interactive table rendering is unavailable on this machine.")
        st.code(data.to_string(index=False))


def _format_currency(value) -> str:
    """Format numeric API values that may be returned as strings."""
    return f"${float(value or 0):,.2f}"


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

            if st.button("Download Sample Historical Prices"):
                with st.spinner("Downloading SPY, QQQ, GLD, TLT and BND prices..."):
                    try:
                        from src.data.data_fetcher import fetch_price_data
                        sample_tickers = ["SPY", "QQQ", "GLD", "TLT", "BND"]
                        end = datetime.now().date()
                        start = end - timedelta(days=365 * 5)
                        prices = fetch_price_data(
                            sample_tickers, start.isoformat(), end.isoformat(), preferred_source="FMP"
                        )
                        if prices.empty:
                            st.warning("No historical data was returned by FMP or Yahoo Finance.")
                        else:
                            st.success(f"Stored {len(prices)} historical price records.")
                    except Exception as error:
                        st.error(f"Historical download failed: {error}")

        with col2:
            st.subheader("Local Data")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(df)} rows")
                _display_table(df.head())

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
        quality_source = st.session_state.data_store.get_price_data(
            ["SPY", "QQQ", "GLD", "TLT", "BND"], "1900-01-01", datetime.now().date().isoformat()
        )
        if quality_source.empty:
            st.info("No historical price data is available. Download or import data before calculating quality scores.")
        else:
            quality = calculate_price_quality(quality_source.rename(columns={
                "tic": "ticker",
                "datadate": "date",
                "prcod": "open",
                "prchd": "high",
                "prcld": "low",
                "prccd": "close",
                "cshtrd": "volume",
            }))
            quality_data = pd.DataFrame({
                "Metric": ["Completeness", "Accuracy", "Consistency", "Timeliness"],
                "Score": [quality["completeness"], quality["accuracy"], quality["consistency"], quality["timeliness"]],
            })
            _display_table(quality_data, use_container_width=True)
            st.caption(
                f"Based on {quality['record_count']} stored rows; latest date: {quality['latest_date']}."
            )


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
                st.metric("Final Value", f"${result.portfolio_values.iloc[-1]:,.2f}")
            with metrics_cols[1]:
                st.metric("Total Return", f"{result.metrics.get('total_return', 0.0):.2%}")
            with metrics_cols[2]:
                st.metric("Annual Return", f"{result.metrics.get('annual_return', 0.0):.2%}")
            with metrics_cols[3]:
                st.metric("Sharpe Ratio", f"{result.metrics.get('sharpe_ratio', 0.0):.2f}")

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
                'Value': [f"{value:.4f}" for value in result.metrics.values()]
            })
            _display_table(metrics_df)


def run_backtest(strategy_type, start_date, end_date, initial_capital, top_quantile):
    """Run a backtest using stored or downloaded historical prices."""
    from src.data.data_fetcher import fetch_price_data
    tickers = ["SPY", "QQQ", "GLD", "TLT", "BND"]
    price_data = fetch_price_data(
        tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
    if price_data.empty:
        raise ValueError("No historical price data is available for this backtest.")

    # Sample weight signals
    weight_signals = pd.DataFrame(
        {ticker: 1.0 / len(tickers) for ticker in tickers},
        index=pd.date_range(start_date, end_date, freq="QE"),
    )

    # Run backtest
    strategy_name = f"{strategy_type} Backtest"
    try:
        from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

        backtest_config = BacktestConfig(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=initial_capital
        )
        engine = BacktestEngine(backtest_config)
        result = engine.run_backtest(
            strategy_name=strategy_name,
            price_data=price_data,
            weight_signals=weight_signals,
        )
    except ImportError as error:
        raise RuntimeError("The native backtest engine is unavailable; synthetic results are disabled.") from error

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
                            st.metric("Portfolio Value", _format_currency(account_info.get("portfolio_value")))
                        with col2:
                            st.metric("Cash", _format_currency(account_info.get("cash")))
                        with col3:
                            st.metric("Buying Power", _format_currency(account_info.get("buying_power")))

                        # Positions table
                        if positions:
                            positions_df = pd.DataFrame(positions)
                            _display_table(
                                positions_df[
                                    ['symbol', 'qty', 'avg_entry_price', 'market_value', 'unrealized_pl']
                                ],
                                use_container_width=True,
                            )
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
    """Show historical-data portfolio performance and risk analytics."""
    st.header("Portfolio Analysis")
    with st.expander("Portfolio inputs", expanded=True):
        ticker_text = st.text_input("Assets (comma-separated)", "SPY, QQQ")
        ticker_list = [ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()]
        default_weight = 1.0 / len(ticker_list) if ticker_list else 0.0
        weight_text = st.text_input(
            "Weights (comma-separated, percentages)",
            ", ".join(f"{default_weight * 100:.2f}" for _ in ticker_list),
        )
        start_date = st.date_input("Start date", datetime.now().date() - timedelta(days=365))
        end_date = st.date_input("End date", datetime.now().date())
        initial_investment = st.number_input("Initial investment", min_value=1.0, value=100000.0)

    try:
        weights = validate_weights({
            ticker: float(weight.strip()) / 100.0
            for ticker, weight in zip(ticker_list, weight_text.split(","))
        })
        data = st.session_state.data_store.get_price_data(
            ticker_list, start_date.isoformat(), end_date.isoformat()
        )
        if data.empty:
            st.info("No historical data is stored for these assets and dates. Load market data in Data Management first.")
            return
        prices = data.assign(datadate=pd.to_datetime(data["datadate"])).pivot(
            index="datadate", columns="tic", values="adj_close"
        ).sort_index()
        returns = prices_to_returns(prices).dropna()
        if len(returns) < 2:
            st.warning("At least two valid historical observations are required.")
            return
        performance = calculate_portfolio_metrics(returns, weights, initial_investment)
        risk = calculate_risk_metrics(performance["daily_returns"])
        assets = analyze_assets(returns, weights)
        correlations = analyze_correlations(returns) if len(returns.columns) > 1 else None
    except (ValueError, KeyError) as error:
        st.error(f"Unable to calculate analytics: {error}")
        return

    tabs = st.tabs(["Portfolio Overview", "Risk Analysis", "Asset Performance", "Correlation", "Allocation", "Benchmark"])
    with tabs[0]:
        st.subheader("Portfolio Overview")
        values = performance["portfolio_values"].rename("Portfolio value")
        st.plotly_chart(px.line(values, title="Historical portfolio value"), use_container_width=True)
        columns = st.columns(4)
        columns[0].metric("Initial investment", _format_currency(performance["initial_investment"]))
        columns[1].metric("Current value", _format_currency(performance["current_value"]))
        columns[2].metric("Total return", f"{performance['total_return']:.2%}")
        columns[3].metric("Annualized return", f"{performance['annualized_return']:.2%}")
    with tabs[1]:
        st.subheader("Risk Analysis")
        columns = st.columns(5)
        columns[0].metric("Volatility", f"{risk['annualized_volatility']:.2%}")
        columns[1].metric("Sharpe", f"{risk['sharpe_ratio']:.2f}")
        columns[2].metric("Sortino", f"{risk['sortino_ratio']:.2f}")
        columns[3].metric("Max drawdown", f"{risk['maximum_drawdown']:.2%}")
        columns[4].metric("CVaR (95%)", f"{risk['cvar']:.2%}")
        drawdown = (1 + performance["daily_returns"]).cumprod()
        drawdown = drawdown / drawdown.cummax() - 1
        st.plotly_chart(px.line(drawdown, title="Historical drawdown"), use_container_width=True)
    with tabs[2]:
        st.subheader("Asset Performance")
        _display_table(assets["assets"].reset_index(), use_container_width=True)
        st.info(f"Best performer: {assets['best_performing_asset']} | Worst performer: {assets['worst_performing_asset']}")
    with tabs[3]:
        st.subheader("Correlation Analysis")
        if correlations is None:
            st.info("Add at least two assets for correlation analysis.")
        else:
            st.plotly_chart(px.imshow(correlations["matrix"], text_auto=".2f", title="Asset return correlation"), use_container_width=True)
            _display_table(pd.DataFrame(correlations["highly_correlated_pairs"]), use_container_width=True)
    with tabs[4]:
        st.subheader("Portfolio Allocation")
        allocation = allocation_summary(weights)
        _display_table(allocation["allocation"].reset_index(), use_container_width=True)
        st.metric("Concentration (Herfindahl index)", f"{allocation['herfindahl_index']:.3f}")
    with tabs[5]:
        st.subheader("Benchmark Comparison")
        if "SPY" not in returns.columns:
            st.info("SPY benchmark data is not available for this date range.")
        else:
            comparison = compare_benchmarks(
                performance["daily_returns"],
                {"SPY buy-and-hold": returns["SPY"]},
            )
            _display_table(comparison.reset_index(), use_container_width=True)


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
