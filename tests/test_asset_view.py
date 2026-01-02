"""Tests for AssetView class."""

import pytest
from datetime import datetime
from trading_frame import Candle, TimeFrame
from trading_frame.indicators import RSI, SMA, BollingerBands
from trading_asset_view import AssetView


class TestAssetViewInitialization:
    """Test AssetView initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H"])

        assert asset_view.symbol == "BTC/USDT"
        assert asset_view.timeframes == ["1T", "5T", "1H"]
        assert len(asset_view.frames) == 3
        assert "1T" in asset_view.frames
        assert "5T" in asset_view.frames
        assert "1H" in asset_view.frames

    def test_init_with_custom_max_periods(self):
        """Test initialization with custom max_periods."""
        asset_view = AssetView("ETH/USDT", timeframes=["1T"], max_periods=500)

        assert asset_view.frames["1T"].max_periods == 500

    def test_init_frames_are_timeframe_instances(self):
        """Test that frames are TimeFrame instances."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        for frame in asset_view.frames.values():
            assert isinstance(frame, TimeFrame)

    def test_init_empty_timeframes_raises_error(self):
        """Test that empty timeframes list raises ValueError."""
        with pytest.raises(ValueError, match="At least one timeframe must be specified"):
            AssetView("BTC/USDT", timeframes=[])

    def test_init_duplicate_timeframes_raises_error(self):
        """Test that duplicate timeframes raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate timeframes are not allowed"):
            AssetView("BTC/USDT", timeframes=["1T", "5T", "1T"])

    def test_init_invalid_max_periods_raises_error(self):
        """Test that invalid max_periods raises ValueError."""
        with pytest.raises(ValueError, match="max_periods must be at least 1"):
            AssetView("BTC/USDT", timeframes=["1T"], max_periods=0)


class TestAssetViewFeeding:
    """Test AssetView candle feeding."""

    def test_feed_single_candle(self):
        """Test feeding a single candle to all timeframes."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        candle = Candle(
            date=datetime(2024, 1, 1, 12, 0, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=123.45
        )

        asset_view.feed(candle)

        # Check that all frames received the candle
        assert len(asset_view.frames["1T"].periods) == 1
        assert len(asset_view.frames["5T"].periods) == 1

    def test_feed_multiple_candles(self):
        """Test feeding multiple candles."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])

        # Feed 3 candles, 1 minute apart
        for i in range(3):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0 + i,
                high=51000.0 + i,
                low=49000.0 + i,
                close=50500.0 + i,
                volume=100.0
            )
            asset_view.feed(candle)

        # Should have 3 periods for 1T
        assert len(asset_view.frames["1T"].periods) == 3

    def test_feed_routes_to_all_timeframes(self):
        """Test that feed routes to all configured timeframes."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H"])

        # Feed 10 candles, 1 minute apart
        for i in range(10):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.0
            )
            asset_view.feed(candle)

        # 1T should have 10 periods
        assert len(asset_view.frames["1T"].periods) == 10
        # 5T should have 2 periods (0-4min, 5-9min)
        assert len(asset_view.frames["5T"].periods) == 2
        # 1H should have 1 period (all in same hour)
        assert len(asset_view.frames["1H"].periods) == 1

    def test_feed_invalid_type_raises_error(self):
        """Test that feeding non-Candle raises TypeError."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])

        with pytest.raises(TypeError, match="Expected Candle instance"):
            asset_view.feed("not a candle")


class TestAssetViewAccess:
    """Test AssetView frame access methods."""

    def test_get_frame_method(self):
        """Test get_frame method."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        frame_1t = asset_view.get_frame("1T")
        frame_5t = asset_view.get_frame("5T")

        assert isinstance(frame_1t, TimeFrame)
        assert isinstance(frame_5t, TimeFrame)
        assert frame_1t is asset_view.frames["1T"]
        assert frame_5t is asset_view.frames["5T"]

    def test_get_frame_invalid_timeframe_raises_error(self):
        """Test that getting invalid timeframe raises KeyError."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])

        with pytest.raises(KeyError, match="Timeframe '1H' not found"):
            asset_view.get_frame("1H")

    def test_dict_like_access(self):
        """Test dict-like access with []."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        frame_1t = asset_view["1T"]
        frame_5t = asset_view["5T"]

        assert isinstance(frame_1t, TimeFrame)
        assert isinstance(frame_5t, TimeFrame)
        assert frame_1t is asset_view.frames["1T"]
        assert frame_5t is asset_view.frames["5T"]

    def test_dict_like_access_invalid_timeframe_raises_error(self):
        """Test that dict-like access with invalid timeframe raises KeyError."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])

        with pytest.raises(KeyError, match="Timeframe '1H' not found"):
            _ = asset_view["1H"]

    def test_timeframes_property_returns_copy(self):
        """Test that timeframes property returns a copy."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        timeframes = asset_view.timeframes
        timeframes.append("1H")

        # Original should not be modified
        assert asset_view.timeframes == ["1T", "5T"]


class TestAssetViewIntegration:
    """Integration tests for AssetView."""

    def test_realistic_trading_scenario(self):
        """Test a realistic trading scenario with multiple timeframes."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "15T", "1H"])

        # Simulate 1 hour of 1-minute candles
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(60):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0 + i * 10,
                high=50100.0 + i * 10,
                low=49900.0 + i * 10,
                close=50050.0 + i * 10,
                volume=100.0 + i
            )
            asset_view.feed(candle)

        # Verify period counts
        assert len(asset_view["1T"].periods) == 60  # 60 x 1-min periods
        assert len(asset_view["5T"].periods) == 12  # 12 x 5-min periods
        assert len(asset_view["15T"].periods) == 4  # 4 x 15-min periods
        assert len(asset_view["1H"].periods) == 1   # 1 x 1-hour period

        # Verify 1H period aggregation
        hour_period = asset_view["1H"].periods[0]
        assert hour_period.open_price == 50000.0  # First candle's open
        assert hour_period.close_price == 50050.0 + 59 * 10  # Last candle's close

    def test_repr(self):
        """Test string representation."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"], max_periods=100)

        repr_str = repr(asset_view)
        assert "BTC/USDT" in repr_str
        assert "['1T', '5T']" in repr_str
        assert "100" in repr_str


class TestAssetViewIndicators:
    """Test AssetView indicator management."""

    def setup_method(self):
        """Setup method with sample data."""
        self.asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        # Feed some candles for indicators to work with
        for i in range(20):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0 + i * 10,
                high=50100.0 + i * 10,
                low=49900.0 + i * 10,
                close=50050.0 + i * 10,
                volume=100.0
            )
            self.asset_view.feed(candle)

    def test_add_indicator_to_single_timeframe(self):
        """Test adding indicator to a specific timeframe."""
        rsi = RSI(length=14)
        self.asset_view.add_indicator("1T", rsi, "RSI_14")

        # Check that indicator exists in 1T
        frame_1t = self.asset_view["1T"]
        assert "RSI_14" in frame_1t.indicators

        # Check that last period has RSI value
        assert frame_1t.periods[-1].RSI_14 is not None

        # Check that 5T doesn't have it
        frame_5t = self.asset_view["5T"]
        assert "RSI_14" not in frame_5t.indicators

    def test_add_indicator_to_all_timeframes(self):
        """Test adding indicator to all timeframes."""
        sma = SMA(period=3)  # Use small period to work with both timeframes
        self.asset_view.add_indicator_to_all(sma, "SMA_3")

        # Check that both timeframes have the indicator
        for tf in ["1T", "5T"]:
            frame = self.asset_view[tf]
            assert "SMA_3" in frame.indicators
            assert frame.periods[-1].SMA_3 is not None

    def test_add_multi_column_indicator(self):
        """Test adding multi-column indicator (Bollinger Bands)."""
        bb = BollingerBands(period=10, std_dev=2.0)
        self.asset_view.add_indicator("1T", bb, ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"])

        frame = self.asset_view["1T"]
        assert ("BB_UPPER", "BB_MIDDLE", "BB_LOWER") in frame.indicators

        # Check that period has all columns
        last_period = frame.periods[-1]
        assert last_period.BB_UPPER is not None
        assert last_period.BB_MIDDLE is not None
        assert last_period.BB_LOWER is not None

    def test_add_indicator_invalid_timeframe_raises_error(self):
        """Test that adding to invalid timeframe raises KeyError."""
        rsi = RSI(length=14)

        with pytest.raises(KeyError, match="Timeframe '1H' not found"):
            self.asset_view.add_indicator("1H", rsi, "RSI_14")

    def test_remove_indicator_from_timeframe(self):
        """Test removing indicator from specific timeframe."""
        rsi = RSI(length=14)
        self.asset_view.add_indicator("1T", rsi, "RSI_14")

        # Verify it exists
        assert "RSI_14" in self.asset_view["1T"].indicators

        # Remove it
        self.asset_view.remove_indicator("1T", "RSI_14")

        # Verify it's gone
        assert "RSI_14" not in self.asset_view["1T"].indicators
        assert not hasattr(self.asset_view["1T"].periods[-1], "RSI_14")

    def test_remove_indicator_from_all_timeframes(self):
        """Test removing indicator from all timeframes."""
        sma = SMA(period=3)
        self.asset_view.add_indicator_to_all(sma, "SMA_3")

        # Remove from all
        self.asset_view.remove_indicator_from_all("SMA_3")

        # Verify it's gone from both
        for tf in ["1T", "5T"]:
            assert "SMA_3" not in self.asset_view[tf].indicators

    def test_remove_indicator_from_all_silently_skips_missing(self):
        """Test that remove_from_all doesn't error if indicator missing."""
        # Add indicator only to 1T
        sma = SMA(period=10)
        self.asset_view.add_indicator("1T", sma, "SMA_10")

        # Remove from all (should skip 5T silently)
        self.asset_view.remove_indicator_from_all("SMA_10")

        # Should not raise error
        assert "SMA_10" not in self.asset_view["1T"].indicators
        assert "SMA_10" not in self.asset_view["5T"].indicators

    def test_get_indicator_columns(self):
        """Test getting list of indicator columns."""
        rsi = RSI(length=14)
        sma = SMA(period=20)

        self.asset_view.add_indicator("1T", rsi, "RSI_14")
        self.asset_view.add_indicator("1T", sma, "SMA_20")

        columns = self.asset_view.get_indicator_columns("1T")

        assert "RSI_14" in columns
        assert "SMA_20" in columns

    def test_get_indicator_columns_multi_column(self):
        """Test getting indicator columns with multi-column indicators."""
        bb = BollingerBands(period=10, std_dev=2.0)
        self.asset_view.add_indicator("1T", bb, ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"])

        columns = self.asset_view.get_indicator_columns("1T")

        assert "BB_UPPER" in columns
        assert "BB_MIDDLE" in columns
        assert "BB_LOWER" in columns

    def test_indicators_calculated_on_feed(self):
        """Test that indicators are recalculated when feeding new candles."""
        sma = SMA(period=5)
        self.asset_view.add_indicator("1T", sma, "SMA_5")

        # Get initial SMA value
        initial_sma = self.asset_view["1T"].periods[-1].SMA_5

        # Feed new candle with higher price
        candle = Candle(
            date=datetime(2024, 1, 1, 12, 20, 0),
            open=55000.0,
            high=56000.0,
            low=54500.0,
            close=55500.0,
            volume=200.0
        )
        self.asset_view.feed(candle)

        # SMA should be recalculated and higher
        new_sma = self.asset_view["1T"].periods[-1].SMA_5

        # New SMA should be higher due to higher price
        assert new_sma > initial_sma
