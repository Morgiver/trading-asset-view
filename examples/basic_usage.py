"""
Real-time QQQ trading example with yfinance and mplfinance visualization.

This example demonstrates:
1. Fetching real market data for QQQ from Yahoo Finance
2. Creating multi-timeframe analysis (1m, 5m, 15m, 1h)
3. Adding multiple technical indicators (RSI, SMA, EMA, MACD, Bollinger Bands)
4. Visualizing candlestick charts with indicators using mplfinance
"""

import yfinance as yf
import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta
from trading_frame import Candle
from trading_indicators import RSI, SMA, EMA, MACD, BollingerBands
from trading_asset_view import AssetView


def fetch_qqq_data(period="1d", interval="1m"):
    """Fetch QQQ data from Yahoo Finance."""
    print(f"Fetching QQQ data (period={period}, interval={interval})...")
    ticker = yf.Ticker("QQQ")
    data = ticker.history(period=period, interval=interval)

    if data.empty:
        raise ValueError("No data fetched from Yahoo Finance")

    print(f"Fetched {len(data)} candles")
    return data


def convert_to_candles(yf_data):
    """Convert yfinance DataFrame to Candle objects."""
    candles = []
    for timestamp, row in yf_data.iterrows():
        candle = Candle(
            date=timestamp.to_pydatetime(),
            open=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=float(row['Volume'])
        )
        candles.append(candle)
    return candles


def main():
    print("="*80)
    print("QQQ Multi-Timeframe Analysis with Technical Indicators")
    print("="*80)

    # Fetch QQQ data (last 5 days, 1-minute candles)
    yf_data = fetch_qqq_data(period="5d", interval="1m")
    candles = convert_to_candles(yf_data)

    # Create AssetView with multiple timeframes
    print("\nCreating AssetView for QQQ...")
    asset_view = AssetView("QQQ", timeframes=["1T", "5T", "15T", "1H"], max_periods=500)

    # Add indicators BEFORE prefill
    print("\nAdding technical indicators...")

    # 1-minute timeframe indicators
    rsi_1m = RSI(frame=asset_view["1T"], length=14, column_name="RSI_14")
    sma_20_1m = SMA(frame=asset_view["1T"], period=20, column_name="SMA_20")
    ema_9_1m = EMA(frame=asset_view["1T"], period=9, column_name="EMA_9")

    # 5-minute timeframe indicators
    rsi_5m = RSI(frame=asset_view["5T"], length=14, column_name="RSI_14")
    sma_50_5m = SMA(frame=asset_view["5T"], period=50, column_name="SMA_50")
    bb_5m = BollingerBands(
        frame=asset_view["5T"],
        period=20,
        nbdevup=2.0,
        nbdevdn=2.0,
        column_names=["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
    )

    # 15-minute timeframe indicators
    macd_15m = MACD(
        frame=asset_view["15T"],
        fast=12,
        slow=26,
        signal=9,
        column_names=["MACD", "MACD_SIGNAL", "MACD_HIST"]
    )
    sma_200_15m = SMA(frame=asset_view["15T"], period=200, column_name="SMA_200")

    # 1-hour timeframe indicators
    bb_1h = BollingerBands(
        frame=asset_view["1H"],
        period=20,
        nbdevup=2.0,
        nbdevdn=2.0,
        column_names=["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
    )
    sma_20_1h = SMA(frame=asset_view["1H"], period=20, column_name="SMA_20")

    # Prefill frames with historical data AFTER adding indicators
    print(f"\nPrefilling frames with {len(candles)} historical candles...")
    for i, candle in enumerate(candles):
        prefill_success = asset_view.prefill(candle)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(candles)} candles...")
        if prefill_success:
            print(f"Prefill completed at candle {i + 1}/{len(candles)}")
            break

    print(f"Prefill done - indicators calculated automatically")

    # Display statistics
    print("\n" + "="*80)
    print("MULTI-TIMEFRAME STATISTICS")
    print("="*80)

    for tf in asset_view.timeframes:
        frame = asset_view[tf]
        print(f"\n{tf} Timeframe:")
        print(f"  Total periods: {len(frame.periods)}")

        if frame.periods:
            last_period = frame.periods[-1]
            print(f"  Last close: ${last_period.close_price:.2f}")
            print(f"  Last volume: {last_period.volume:,.0f}")

    # Display indicator values
    print("\n" + "="*80)
    print("TECHNICAL INDICATORS (Latest Values)")
    print("="*80)

    # 1-minute indicators
    if rsi_1m.periods:
        print("\n1-Minute Timeframe:")
        last_rsi = rsi_1m.periods[-1]
        last_sma = sma_20_1m.periods[-1] if sma_20_1m.periods else None
        last_ema = ema_9_1m.periods[-1] if ema_9_1m.periods else None

        if hasattr(last_rsi, 'RSI_14'):
            print(f"  RSI(14): {last_rsi.RSI_14:.2f}")
        if last_sma and hasattr(last_sma, 'SMA_20'):
            print(f"  SMA(20): ${last_sma.SMA_20:.2f}")
        if last_ema and hasattr(last_ema, 'EMA_9'):
            print(f"  EMA(9): ${last_ema.EMA_9:.2f}")

    # 5-minute indicators
    if rsi_5m.periods:
        print("\n5-Minute Timeframe:")
        last_rsi = rsi_5m.periods[-1]
        last_bb = bb_5m.periods[-1] if bb_5m.periods else None

        if hasattr(last_rsi, 'RSI_14'):
            print(f"  RSI(14): {last_rsi.RSI_14:.2f}")
        if last_bb and hasattr(last_bb, 'BB_UPPER'):
            print(f"  Bollinger Bands:")
            print(f"    Upper: ${last_bb.BB_UPPER:.2f}")
            print(f"    Middle: ${last_bb.BB_MIDDLE:.2f}")
            print(f"    Lower: ${last_bb.BB_LOWER:.2f}")

    # 15-minute indicators
    if macd_15m.periods:
        print("\n15-Minute Timeframe:")
        last_macd = macd_15m.periods[-1]

        if hasattr(last_macd, 'MACD'):
            print(f"  MACD:")
            print(f"    MACD: {last_macd.MACD:.4f}")
            print(f"    Signal: {last_macd.MACD_SIGNAL:.4f}")
            print(f"    Histogram: {last_macd.MACD_HIST:.4f}")

    # 1-hour indicators
    if bb_1h.periods:
        print("\n1-Hour Timeframe:")
        last_bb = bb_1h.periods[-1]

        if hasattr(last_bb, 'BB_UPPER'):
            print(f"  Bollinger Bands:")
            print(f"    Upper: ${last_bb.BB_UPPER:.2f}")
            print(f"    Middle: ${last_bb.BB_MIDDLE:.2f}")
            print(f"    Lower: ${last_bb.BB_LOWER:.2f}")

    # Create visualization with single window
    print("\n" + "="*80)
    print("GENERATING CHART")
    print("="*80)

    # Prepare data for 5-minute timeframe with Bollinger Bands and RSI
    frame_5m = asset_view["5T"]
    if len(frame_5m.periods) > 0:
        df_5m = frame_5m.to_pandas()

        # Ensure index is DatetimeIndex
        if not isinstance(df_5m.index, pd.DatetimeIndex):
            df_5m.index = pd.to_datetime(df_5m.index)

        # Rename columns to match mplfinance requirements (capitalized OHLCV)
        df_5m = df_5m.rename(columns={
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'close_price': 'Close',
            'volume': 'Volume'
        })

        # Prepare additional plots for indicators
        apds = []

        # Add Bollinger Bands if available
        if len(bb_5m.periods) > 0:
            try:
                df_bb = bb_5m.to_pandas()

                # Ensure datetime index
                if not isinstance(df_bb.index, pd.DatetimeIndex):
                    df_bb.index = pd.to_datetime(df_bb.index)

                # Align with main dataframe
                df_bb = df_bb.reindex(df_5m.index)

                valid_count = df_bb['BB_UPPER'].notna().sum()
                print(f"Bollinger Bands: {valid_count} valid values out of {len(df_5m)}")

                if valid_count > 0:
                    bb_plot = [
                        mpf.make_addplot(df_bb['BB_UPPER'], color='gray', linestyle='--', width=0.7, alpha=0.7),
                        mpf.make_addplot(df_bb['BB_MIDDLE'], color='blue', linestyle='-', width=1.0),
                        mpf.make_addplot(df_bb['BB_LOWER'], color='gray', linestyle='--', width=0.7, alpha=0.7)
                    ]
                    apds.extend(bb_plot)
            except Exception as e:
                print(f"Warning: Could not add Bollinger Bands to chart: {e}")

        # Add RSI as separate panel if available
        if len(rsi_5m.periods) > 0:
            try:
                df_rsi = rsi_5m.to_pandas()

                # Ensure datetime index
                if not isinstance(df_rsi.index, pd.DatetimeIndex):
                    df_rsi.index = pd.to_datetime(df_rsi.index)

                # Get RSI values
                if isinstance(df_rsi, pd.DataFrame) and 'RSI_14' in df_rsi.columns:
                    rsi_values = df_rsi['RSI_14']
                else:
                    rsi_values = df_rsi

                # Align with main dataframe
                rsi_values = rsi_values.reindex(df_5m.index)

                valid_count = rsi_values.notna().sum()
                print(f"RSI: {valid_count} valid values out of {len(df_5m)}")

                if valid_count > 0:
                    # Panel 0 = price (with BB), Panel 1 = volume, Panel 2 = RSI
                    rsi_plot = mpf.make_addplot(
                        rsi_values,
                        panel=2,
                        color='purple',
                        ylabel='RSI',
                        ylim=(0, 100),
                        width=1.2
                    )
                    apds.append(rsi_plot)
            except Exception as e:
                print(f"Warning: Could not add RSI to chart: {e}")

        # Create the chart
        print("\nGenerating 5-minute candlestick chart with indicators...")

        style = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 8})

        # Determine panel ratios based on number of panels
        # Panel 0 = price, Panel 1 = volume, Panel 2 = RSI (if added)
        has_rsi_panel = False
        for ap in apds:
            if hasattr(ap, 'panel') and ap.panel == 2:
                has_rsi_panel = True
                break

        # Ratios: (price_panel, volume_panel, rsi_panel)
        panel_ratios = (5, 1, 1.5) if has_rsi_panel else (5, 1)

        plot_kwargs = {
            'type': 'candle',
            'style': style,
            'title': 'QQQ - 5-Minute Chart with Bollinger Bands & RSI',
            'ylabel': 'Price ($)',
            'volume': True,
            'figsize': (18, 10),
            'panel_ratios': panel_ratios
        }

        # Only add addplot if we have indicators to plot
        if apds:
            plot_kwargs['addplot'] = apds

        print(f"Displaying chart with {len(apds)} indicator plots...")
        mpf.plot(df_5m, **plot_kwargs)
        print("Chart window opened. Close the window to continue...")

    # Display summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSuccessfully processed {len(candles)} QQQ candles")
    print(f"Generated {len(asset_view.timeframes)} timeframes: {', '.join(asset_view.timeframes)}")
    print(f"Added {sum(1 for _ in [rsi_1m, sma_20_1m, ema_9_1m, rsi_5m, bb_5m, macd_15m, bb_1h, sma_20_1h])} indicators")
    print(f"Displayed 5-minute chart with Bollinger Bands and RSI")
    print("\nClose the chart window to exit...")
    print("\nYou can now use this AssetView object for further analysis or trading strategies!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
