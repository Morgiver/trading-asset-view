"""Example usage of trading-asset-view with indicators."""

from datetime import datetime, timedelta
from trading_frame import Candle
from trading_frame.indicators import RSI, SMA, BollingerBands
from trading_asset_view import AssetView


def main():
    # Create AssetView with multiple timeframes
    print("Creating AssetView for BTC/USDT...")
    asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "15T", "1H"])

    # Add indicators
    print("\nAdding indicators...")

    # RSI on 1-minute timeframe
    asset_view.add_indicator("1T", RSI(length=14), "RSI_14")

    # SMA on all timeframes
    asset_view.add_indicator_to_all(SMA(period=20), "SMA_20")

    # Bollinger Bands on 1-hour timeframe
    bb = BollingerBands(period=20, std_dev=2.0)
    asset_view.add_indicator("1H", bb, ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"])

    # Feed candles (simulate 2 hours of 1-minute candles)
    print("\nFeeding candles...")
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    base_price = 50000.0

    for i in range(120):
        # Simulate some price variation
        price_change = (i % 20 - 10) * 50  # Oscillating price

        candle = Candle(
            date=base_time + timedelta(minutes=i),
            open=base_price + price_change,
            high=base_price + price_change + 100,
            low=base_price + price_change - 100,
            close=base_price + price_change + 50,
            volume=100.0 + (i % 10)
        )
        asset_view.feed(candle)

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # 1-minute timeframe
    frame_1m = asset_view["1T"]
    print(f"\n1-Minute Timeframe:")
    print(f"  Total periods: {len(frame_1m.periods)}")
    print(f"  Indicators: {asset_view.get_indicator_columns('1T')}")

    last_period_1m = frame_1m.periods[-1]
    print(f"\n  Last period:")
    print(f"    Close: ${last_period_1m.close_price:,.2f}")
    print(f"    RSI(14): {last_period_1m.RSI_14:.2f}")
    print(f"    SMA(20): ${last_period_1m.SMA_20:,.2f}" if last_period_1m.SMA_20 else "    SMA(20): N/A")

    # 5-minute timeframe
    frame_5m = asset_view["5T"]
    print(f"\n5-Minute Timeframe:")
    print(f"  Total periods: {len(frame_5m.periods)}")
    print(f"  Indicators: {asset_view.get_indicator_columns('5T')}")

    last_period_5m = frame_5m.periods[-1]
    print(f"\n  Last period:")
    print(f"    Close: ${last_period_5m.close_price:,.2f}")
    print(f"    SMA(20): ${last_period_5m.SMA_20:,.2f}" if last_period_5m.SMA_20 else "    SMA(20): N/A")

    # 1-hour timeframe
    frame_1h = asset_view["1H"]
    print(f"\n1-Hour Timeframe:")
    print(f"  Total periods: {len(frame_1h.periods)}")
    print(f"  Indicators: {asset_view.get_indicator_columns('1H')}")

    last_period_1h = frame_1h.periods[-1]
    print(f"\n  Last period:")
    print(f"    Close: ${last_period_1h.close_price:,.2f}")
    print(f"    SMA(20): ${last_period_1h.SMA_20:,.2f}" if last_period_1h.SMA_20 else "    SMA(20): N/A")
    if last_period_1h.BB_UPPER:
        print(f"    Bollinger Bands:")
        print(f"      Upper: ${last_period_1h.BB_UPPER:,.2f}")
        print(f"      Middle: ${last_period_1h.BB_MIDDLE:,.2f}")
        print(f"      Lower: ${last_period_1h.BB_LOWER:,.2f}")

    # Convert to pandas for analysis
    print(f"\n{'='*70}")
    print("DATAFRAME CONVERSION")
    print("="*70)

    df_1m = frame_1m.to_pandas()
    print(f"\n1-Minute DataFrame shape: {df_1m.shape}")
    print("\nLast 5 periods:")
    print(df_1m[['close_price', 'RSI_14', 'SMA_20']].tail())


if __name__ == "__main__":
    main()
