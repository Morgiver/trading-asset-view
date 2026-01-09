"""
Complex multi-timeframe analysis example with multiple indicators.

This example demonstrates:
1. Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
2. Multiple indicators per timeframe (RSI, SMA, EMA, MACD, Bollinger Bands, ATR)
3. Console output showing last 5 periods for each timeframe and indicator
"""

import yfinance as yf
from datetime import datetime
from trading_frame import Candle
from trading_indicators import RSI, SMA, EMA, MACD, BollingerBands, ATR
from trading_asset_view import AssetView


def fetch_qqq_data(period="30d", interval="1m"):
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


def print_separator(char="=", length=100):
    """Print a separator line."""
    print(char * length)


def print_header(text):
    """Print a section header."""
    print_separator()
    print(f" {text}")
    print_separator()


def print_frame_data(frame, timeframe_name):
    """Print last 5 periods of a frame."""
    print(f"\n{timeframe_name} Frame - Last 5 Periods:")
    print(f"  Total periods: {len(frame.periods)}")

    if len(frame.periods) >= 5:
        periods = frame.periods[-5:]
    else:
        periods = frame.periods

    print(f"\n  {'Timestamp':<25} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for period in periods:
        ts_str = period.open_date.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {ts_str:<25} {period.open_price:>10.2f} {period.high_price:>10.2f} "
              f"{period.low_price:>10.2f} {period.close_price:>10.2f} {period.volume:>12,.0f}")


def print_indicator_data(indicator, indicator_name, column_names):
    """Print last 5 periods of an indicator."""
    print(f"\n  {indicator_name} - Last 5 Periods:")
    print(f"    Total periods: {len(indicator.periods)}")

    if len(indicator.periods) >= 5:
        periods = indicator.periods[-5:]
    else:
        periods = indicator.periods

    # Build header
    header_parts = [f"{'Timestamp':<25}"]
    for col_name in column_names:
        header_parts.append(f"{col_name:>12}")
    print(f"    {' '.join(header_parts)}")

    separator_parts = [f"{'-'*25}"]
    for _ in column_names:
        separator_parts.append(f"{'-'*12}")
    print(f"    {' '.join(separator_parts)}")

    # Print data
    for period in periods:
        ts_str = period.open_date.strftime('%Y-%m-%d %H:%M:%S')
        row_parts = [f"{ts_str:<25}"]

        for col_name in column_names:
            if hasattr(period, col_name):
                value = getattr(period, col_name)
                if value is None or (isinstance(value, float) and value != value):  # Check for NaN
                    row_parts.append(f"{'NaN':>12}")
                else:
                    row_parts.append(f"{value:>12.4f}")
            else:
                row_parts.append(f"{'N/A':>12}")

        print(f"    {' '.join(row_parts)}")


def main():
    print_header("QQQ MULTI-TIMEFRAME ANALYSIS WITH MULTIPLE INDICATORS")

    # Fetch data (30 days of 1-minute data)
    yf_data = fetch_qqq_data(period="5d", interval="1m")
    candles = convert_to_candles(yf_data)

    # Create AssetView with multiple timeframes
    print("\nCreating AssetView for QQQ...")
    timeframes = ["1T", "5T", "15T", "1H"]
    asset_view = AssetView("QQQ", timeframes=timeframes, max_periods=500)

    print(f"Timeframes: {', '.join(timeframes)}")

    # Dictionary to store all indicators
    indicators = {}

    # Add indicators BEFORE prefill
    print_header("ADDING INDICATORS")

    # 1-minute timeframe indicators
    print("\n1-Minute Timeframe:")
    indicators['1T'] = {
        'RSI': RSI(frame=asset_view["1T"], length=14, column_name="RSI_14"),
        'SMA_20': SMA(frame=asset_view["1T"], period=20, column_name="SMA_20"),
        'EMA_9': EMA(frame=asset_view["1T"], period=9, column_name="EMA_9"),
        'ATR': ATR(frame=asset_view["1T"], period=14, column_name="ATR_14")
    }
    print(f"  Added: RSI(14), SMA(20), EMA(9), ATR(14)")

    # 5-minute timeframe indicators
    print("\n5-Minute Timeframe:")
    indicators['5T'] = {
        'RSI': RSI(frame=asset_view["5T"], length=14, column_name="RSI_14"),
        'SMA_50': SMA(frame=asset_view["5T"], period=50, column_name="SMA_50"),
        'EMA_21': EMA(frame=asset_view["5T"], period=21, column_name="EMA_21"),
        'BB': BollingerBands(
            frame=asset_view["5T"],
            period=20,
            nbdevup=2.0,
            nbdevdn=2.0,
            column_names=["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
        ),
        'ATR': ATR(frame=asset_view["5T"], period=14, column_name="ATR_14")
    }
    print(f"  Added: RSI(14), SMA(50), EMA(21), Bollinger Bands(20), ATR(14)")

    # 15-minute timeframe indicators
    print("\n15-Minute Timeframe:")
    indicators['15T'] = {
        'RSI': RSI(frame=asset_view["15T"], length=14, column_name="RSI_14"),
        'SMA_200': SMA(frame=asset_view["15T"], period=200, column_name="SMA_200"),
        'EMA_50': EMA(frame=asset_view["15T"], period=50, column_name="EMA_50"),
        'MACD': MACD(
            frame=asset_view["15T"],
            fast=12,
            slow=26,
            signal=9,
            column_names=["MACD", "MACD_SIGNAL", "MACD_HIST"]
        ),
        'BB': BollingerBands(
            frame=asset_view["15T"],
            period=20,
            nbdevup=2.0,
            nbdevdn=2.0,
            column_names=["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
        )
    }
    print(f"  Added: RSI(14), SMA(200), EMA(50), MACD(12,26,9), Bollinger Bands(20)")

    # 1-hour timeframe indicators
    print("\n1-Hour Timeframe:")
    indicators['1H'] = {
        'RSI': RSI(frame=asset_view["1H"], length=14, column_name="RSI_14"),
        'SMA_20': SMA(frame=asset_view["1H"], period=20, column_name="SMA_20"),
        'SMA_50': SMA(frame=asset_view["1H"], period=50, column_name="SMA_50"),
        'EMA_12': EMA(frame=asset_view["1H"], period=12, column_name="EMA_12"),
        'BB': BollingerBands(
            frame=asset_view["1H"],
            period=20,
            nbdevup=2.0,
            nbdevdn=2.0,
            column_names=["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
        ),
        'ATR': ATR(frame=asset_view["1H"], period=14, column_name="ATR_14")
    }
    print(f"  Added: RSI(14), SMA(20), SMA(50), EMA(12), Bollinger Bands(20), ATR(14)")

    # Prefill frames with historical data
    print_header("PREFILLING FRAMES")
    print(f"\nProcessing {len(candles)} candles...")

    for i, candle in enumerate(candles):
        asset_view.prefill(candle)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(candles)} candles...")

    print(f"\nPrefill completed - all indicators calculated automatically")

    # Display results for each timeframe
    print_header("TIMEFRAME ANALYSIS")

    for tf in timeframes:
        print_header(f"TIMEFRAME: {tf}")

        # Print frame data
        frame = asset_view[tf]
        print_frame_data(frame, tf)

        # Print indicators for this timeframe
        print(f"\n{tf} Indicators:")

        tf_indicators = indicators[tf]
        for ind_name, ind_obj in tf_indicators.items():
            if ind_name == 'RSI':
                print_indicator_data(ind_obj, "RSI", ["RSI_14"])
            elif ind_name == 'SMA_20':
                print_indicator_data(ind_obj, "SMA(20)", ["SMA_20"])
            elif ind_name == 'SMA_50':
                print_indicator_data(ind_obj, "SMA(50)", ["SMA_50"])
            elif ind_name == 'SMA_200':
                print_indicator_data(ind_obj, "SMA(200)", ["SMA_200"])
            elif ind_name == 'EMA_9':
                print_indicator_data(ind_obj, "EMA(9)", ["EMA_9"])
            elif ind_name == 'EMA_12':
                print_indicator_data(ind_obj, "EMA(12)", ["EMA_12"])
            elif ind_name == 'EMA_21':
                print_indicator_data(ind_obj, "EMA(21)", ["EMA_21"])
            elif ind_name == 'EMA_50':
                print_indicator_data(ind_obj, "EMA(50)", ["EMA_50"])
            elif ind_name == 'BB':
                print_indicator_data(ind_obj, "Bollinger Bands", ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"])
            elif ind_name == 'MACD':
                print_indicator_data(ind_obj, "MACD", ["MACD", "MACD_SIGNAL", "MACD_HIST"])
            elif ind_name == 'ATR':
                print_indicator_data(ind_obj, "ATR", ["ATR_14"])

    # Summary
    print_header("SUMMARY")

    total_indicators = sum(len(inds) for inds in indicators.values())

    print(f"\nProcessed {len(candles)} QQQ candles")
    print(f"Generated {len(timeframes)} timeframes: {', '.join(timeframes)}")
    print(f"Added {total_indicators} indicators across all timeframes")
    print(f"\nIndicator breakdown:")
    for tf, inds in indicators.items():
        print(f"  {tf}: {len(inds)} indicators ({', '.join(inds.keys())})")

    print(f"\nAll data synchronized and ready for analysis!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
