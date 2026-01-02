"""Tests for advanced AssetView features."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from trading_frame import Candle
from trading_frame.indicators import SMA
from trading_asset_view import AssetView


class TestEventManagement:
    """Test cross-timeframe event management."""

    def test_candle_fed_event(self):
        """Test candle_fed event is triggered."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])
        events_received = []

        def on_candle_fed(av, candle):
            events_received.append(('candle_fed', candle.date))

        asset_view.on('candle_fed', on_candle_fed)

        candle = Candle(
            date=datetime(2024, 1, 1, 12, 0, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.0
        )
        asset_view.feed(candle)

        assert len(events_received) == 1
        assert events_received[0][0] == 'candle_fed'

    def test_period_created_event(self):
        """Test period_created event is triggered."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])
        events_received = []

        def on_period_created(av, timeframe, frame):
            events_received.append(('period_created', timeframe))

        asset_view.on('period_created', on_period_created)

        candle = Candle(
            date=datetime(2024, 1, 1, 12, 0, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.0
        )
        asset_view.feed(candle)

        assert len(events_received) == 1
        assert events_received[0][1] == '1T'

    def test_all_periods_closed_event(self):
        """Test all_periods_closed event when multiple timeframes close."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])
        events_received = []

        def on_all_closed(av, timeframes):
            events_received.append(('all_closed', timeframes))

        asset_view.on('all_periods_closed', on_all_closed)

        # Feed candles to trigger period closes
        for i in range(6):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.0
            )
            asset_view.feed(candle)

        # Should have triggered when both 1T and 5T closed
        assert len(events_received) > 0


class TestHistoricalDataLoading:
    """Test historical data loading features."""

    def test_load_historical_data(self):
        """Test loading historical candles."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])

        candles = [
            Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0 + i,
                high=51000.0 + i,
                low=49000.0 + i,
                close=50500.0 + i,
                volume=100.0
            )
            for i in range(10)
        ]

        asset_view.load_historical_data(candles)

        assert len(asset_view["1T"].periods) == 10

    def test_load_historical_data_empty_raises_error(self):
        """Test that empty candles list raises error."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T"])

        with pytest.raises(ValueError, match="Candles list cannot be empty"):
            asset_view.load_historical_data([])

    def test_load_from_dataframe(self):
        """Test loading from pandas DataFrame."""
        import pandas as pd

        asset_view = AssetView("BTC/USDT", timeframes=["1T"])

        df = pd.DataFrame({
            'date': [datetime(2024, 1, 1, 12, i, 0) for i in range(5)],
            'open': [50000.0 + i for i in range(5)],
            'high': [51000.0 + i for i in range(5)],
            'low': [49000.0 + i for i in range(5)],
            'close': [50500.0 + i for i in range(5)],
            'volume': [100.0 for _ in range(5)]
        })

        asset_view.load_from_dataframe(df)

        assert len(asset_view["1T"].periods) == 5


class TestUnifiedExport:
    """Test unified multi-timeframe export features."""

    def setup_method(self):
        """Setup method with sample data."""
        self.asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        for i in range(10):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0 + i * 10,
                high=50100.0 + i * 10,
                low=49900.0 + i * 10,
                close=50050.0 + i * 10,
                volume=100.0
            )
            self.asset_view.feed(candle)

    def test_to_dict(self):
        """Test to_dict export."""
        data = self.asset_view.to_dict()

        assert isinstance(data, dict)
        assert '1T' in data
        assert '5T' in data
        assert len(data['1T']) == 10
        assert len(data['5T']) == 2

    def test_to_pandas_multi(self):
        """Test to_pandas_multi export."""
        df = self.asset_view.to_pandas_multi()

        assert 'timeframe' in df.index.names
        assert 'open_date' in df.index.names
        assert len(df.loc['1T']) == 10
        assert len(df.loc['5T']) == 2

    def test_to_numpy_all(self):
        """Test to_numpy_all export."""
        arrays = self.asset_view.to_numpy_all()

        assert isinstance(arrays, dict)
        assert '1T' in arrays
        assert '5T' in arrays
        assert arrays['1T'].shape[0] == 10
        assert arrays['5T'].shape[0] == 2

    def test_to_normalize_all(self):
        """Test to_normalize_all export."""
        normalized = self.asset_view.to_normalize_all()

        assert isinstance(normalized, dict)
        assert '1T' in normalized
        assert '5T' in normalized

        # Check values are in [0, 1] range (allowing some numerical tolerance)
        import numpy as np
        assert np.all((normalized['1T'] >= -0.01) & (normalized['1T'] <= 1.01) | np.isnan(normalized['1T']))


class TestStateManagement:
    """Test state management and snapshot features."""

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        asset_view1 = AssetView("BTC/USDT", timeframes=["1T"])

        # Feed some data
        for i in range(5):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.0
            )
            asset_view1.feed(candle)

        # Save state
        state = asset_view1.save_state()

        # Create new instance and load state
        asset_view2 = AssetView("BTC/USDT", timeframes=["1T"])
        asset_view2.load_state(state)

        assert asset_view2.symbol == "BTC/USDT"
        assert asset_view2.timeframes == ["1T"]

    def test_to_pickle_and_from_pickle(self):
        """Test pickle save/load."""
        asset_view1 = AssetView("BTC/USDT", timeframes=["1T"])

        # Feed some data
        for i in range(5):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.0
            )
            asset_view1.feed(candle)

        # Save to pickle
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name

        try:
            asset_view1.to_pickle(filepath)

            # Load from pickle
            asset_view2 = AssetView.from_pickle(filepath)

            assert asset_view2.symbol == "BTC/USDT"
            assert asset_view2.timeframes == ["1T"]
        finally:
            os.unlink(filepath)


class TestStatistics:
    """Test statistics and metrics gathering."""

    def test_get_statistics(self):
        """Test get_statistics method."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

        for i in range(10):
            candle = Candle(
                date=datetime(2024, 1, 1, 12, i, 0),
                open=50000.0 + i * 10,
                high=50100.0 + i * 10,
                low=49900.0 + i * 10,
                close=50050.0 + i * 10,
                volume=100.0 + i
            )
            asset_view.feed(candle)

        stats = asset_view.get_statistics()

        assert stats['symbol'] == "BTC/USDT"
        assert stats['total_periods']['1T'] == 10
        assert stats['total_periods']['5T'] == 2
        assert stats['price_range']['min'] == 49900.0
        assert stats['price_range']['max'] == 50190.0
        assert stats['volume_stats']['total'] > 0


class TestTemporalAlignment:
    """Test temporal alignment features."""

    def test_get_aligned_periods(self):
        """Test get_aligned_periods method."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

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

        # Get aligned periods for a specific timestamp
        aligned = asset_view.get_aligned_periods(datetime(2024, 1, 1, 12, 3, 30))

        assert aligned['1T'] is not None
        assert aligned['5T'] is not None


class TestConsistencyValidation:
    """Test consistency validation."""

    def test_validate_consistency(self):
        """Test validate_consistency method."""
        asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T"])

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

        # Should be consistent
        assert asset_view.validate_consistency() is True
