"""Tests for AssetView class."""

import pytest
from datetime import datetime
from trading_frame import Candle, TimeFrame
from trading_indicators import RSI, SMA, BollingerBands
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
    """Test AssetView with new indicator system (trading-indicators)."""

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
        """Test adding indicator to a specific timeframe using new API."""
        # Create RSI indicator bound to specific frame
        frame_1t = self.asset_view["1T"]
        rsi = RSI(frame=frame_1t, length=14, column_name="RSI_14")

        # Check that indicator has periods
        assert len(rsi.periods) > 0

        # Check that last period has RSI value
        assert hasattr(rsi.periods[-1], "RSI_14")
        assert rsi.periods[-1].RSI_14 is not None

    def test_add_indicator_to_all_timeframes(self):
        """Test adding same indicator type to all timeframes."""
        # Create separate SMA instances for each timeframe
        indicators = {}
        for tf in ["1T", "5T"]:
            frame = self.asset_view[tf]
            indicators[tf] = SMA(frame=frame, period=3, column_name="SMA_3")

        # Check that both indicators have periods
        for tf in ["1T", "5T"]:
            assert len(indicators[tf].periods) > 0
            assert hasattr(indicators[tf].periods[-1], "SMA_3")
            assert indicators[tf].periods[-1].SMA_3 is not None

    def test_add_multi_column_indicator(self):
        """Test adding multi-column indicator (Bollinger Bands)."""
        frame = self.asset_view["1T"]
        bb = BollingerBands(
            frame=frame,
            period=10,
            nbdevup=2.0,
            nbdevdn=2.0,
            column_names=["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
        )

        # Check that indicator has periods with all columns
        assert len(bb.periods) > 0
        last_period = bb.periods[-1]
        assert hasattr(last_period, "BB_UPPER")
        assert hasattr(last_period, "BB_MIDDLE")
        assert hasattr(last_period, "BB_LOWER")
        assert last_period.BB_UPPER is not None
        assert last_period.BB_MIDDLE is not None
        assert last_period.BB_LOWER is not None

    def test_indicator_synchronizes_with_frame(self):
        """Test that indicator automatically synchronizes with frame events."""
        frame = self.asset_view["1T"]
        initial_periods = len(frame.periods)

        # Create indicator
        sma = SMA(frame=frame, period=5, column_name="SMA_5")

        # Feed new candle
        candle = Candle(
            date=datetime(2024, 1, 1, 12, 20, 0),
            open=55000.0,
            high=56000.0,
            low=54500.0,
            close=55500.0,
            volume=200.0
        )
        self.asset_view.feed(candle)

        # Check that indicator added new period automatically
        assert len(sma.periods) == initial_periods + 1

    def test_indicators_recalculate_on_update(self):
        """Test that indicators recalculate when frames update."""
        frame = self.asset_view["1T"]
        sma = SMA(frame=frame, period=5, column_name="SMA_5")

        # Get initial SMA value
        initial_sma = sma.periods[-1].SMA_5 if hasattr(sma.periods[-1], "SMA_5") else None

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

        # SMA should be recalculated for the new period
        new_sma = sma.periods[-1].SMA_5
        assert new_sma is not None

        # New SMA should be higher due to higher price
        if initial_sma is not None:
            assert new_sma > initial_sma
