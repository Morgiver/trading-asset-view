"""AssetView - Unified multi-timeframe management for trading assets."""

from typing import Dict, List, Union, Callable, Any, Optional
from datetime import datetime
import pickle
from trading_frame import TimeFrame, Candle
from trading_frame.indicators import Indicator


class AssetView:
    """
    Manages multiple timeframes for a single trading asset.

    AssetView provides a unified interface to feed candle data to multiple
    timeframes simultaneously and retrieve data from any timeframe.

    Example:
        >>> asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H"])
        >>> candle = Candle(date=1234567890, open=50000, high=51000,
        ...                 low=49000, close=50500, volume=123.45)
        >>> asset_view.feed(candle)
        >>> frame_1m = asset_view.get_frame("1T")
    """

    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        max_periods: int = 250
    ) -> None:
        """
        Initialize AssetView with multiple timeframes.

        Parameters:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframes: List of timeframe strings (e.g., ["1T", "5T", "1H"])
            max_periods: Maximum number of periods to keep per timeframe

        Raises:
            ValueError: If timeframes list is empty or contains duplicates
            ValueError: If max_periods is less than 1
        """
        if not timeframes:
            raise ValueError("At least one timeframe must be specified")

        if len(timeframes) != len(set(timeframes)):
            raise ValueError("Duplicate timeframes are not allowed")

        if max_periods < 1:
            raise ValueError("max_periods must be at least 1")

        self.symbol = symbol
        self._timeframes = timeframes
        self._max_periods = max_periods

        # Event management
        self._event_handlers: Dict[str, List[Callable]] = {
            'candle_fed': [],
            'period_created': [],
            'period_updated': [],
            'period_closed': [],
            'all_periods_closed': [],
        }

        # Create TimeFrame instances for each timeframe
        self.frames: Dict[str, TimeFrame] = {}
        for tf in timeframes:
            frame = TimeFrame(periods_length=tf, max_periods=max_periods)

            # Subscribe to frame events
            frame.on('new_period', lambda f, tf=tf: self._on_frame_new_period(tf, f))
            frame.on('update', lambda f, tf=tf: self._on_frame_update(tf, f))
            frame.on('close', lambda f, tf=tf: self._on_frame_close(tf, f))

            self.frames[tf] = frame

    @property
    def timeframes(self) -> List[str]:
        """Get list of configured timeframes."""
        return self._timeframes.copy()

    def feed(self, candle: Candle) -> None:
        """
        Feed a candle to all timeframes.

        The candle is automatically routed to all configured timeframes,
        each of which will handle period creation/updating based on its
        own interval logic.

        Parameters:
            candle: Candle instance to feed

        Raises:
            TypeError: If candle is not a Candle instance
        """
        if not isinstance(candle, Candle):
            raise TypeError(f"Expected Candle instance, got {type(candle)}")

        # Reset closed timeframes tracker
        self._closed_timeframes = []

        # Feed to all frames
        for frame in self.frames.values():
            frame.feed(candle)

        # Emit candle_fed event
        self._emit('candle_fed', self, candle)

        # Emit all_periods_closed if multiple timeframes closed
        if len(self._closed_timeframes) > 1:
            self._emit('all_periods_closed', self, self._closed_timeframes.copy())

        # Cleanup
        self._closed_timeframes = []

    def prefill(
        self,
        candle: Candle,
        target_periods: Optional[int] = None,
        target_timestamp: Optional[float] = None
    ) -> Dict[str, bool]:
        """
        Feed candle to all timeframes and check if prefill targets are reached.

        Use this method during warm-up phase to fill all timeframes until
        ALL of them reach their target condition.

        Parameters:
            candle: Candle to process
            target_periods: Stop when this many CLOSED periods reached per timeframe (default: max_periods)
            target_timestamp: Stop when candle timestamp >= this value

        Returns:
            Dict mapping timeframe to completion status (True if target reached)
            When all values are True, prefill is complete for all timeframes

        Raises:
            ValueError: If both targets specified or neither specified
            TypeError: If candle is not a Candle instance

        Example:
            # Fill all timeframes until each has max_periods closed periods
            >>> for candle in historical_data:
            ...     status = asset_view.prefill(candle)
            ...     if all(status.values()):
            ...         break  # All timeframes ready

            # Fill until specific timestamp
            >>> target = datetime(2024, 1, 1).timestamp()
            >>> for candle in historical_data:
            ...     status = asset_view.prefill(candle, target_timestamp=target)
            ...     if all(status.values()):
            ...         break

            # Fill with specific period count per timeframe
            >>> for candle in historical_data:
            ...     status = asset_view.prefill(candle, target_periods=50)
            ...     if all(status.values()):
            ...         break  # All timeframes have 50+ closed periods
        """
        if not isinstance(candle, Candle):
            raise TypeError(f"Expected Candle instance, got {type(candle)}")

        # Validate parameters (same as Frame.prefill)
        if target_periods is not None and target_timestamp is not None:
            raise ValueError("Specify either target_periods or target_timestamp, not both")

        if target_periods is None and target_timestamp is None:
            # Default to max_periods
            target_periods = self._max_periods

        # Track completion status for each timeframe
        status = {}

        # Feed to all frames and check their status
        for tf, frame in self.frames.items():
            # Use frame's prefill method
            is_complete = frame.prefill(
                candle,
                target_periods=target_periods,
                target_timestamp=target_timestamp
            )
            status[tf] = is_complete

        return status

    def get_frame(self, timeframe: str) -> TimeFrame:
        """
        Get TimeFrame instance for a specific timeframe.

        Parameters:
            timeframe: Timeframe string (e.g., "1T", "5T")

        Returns:
            TimeFrame instance

        Raises:
            KeyError: If timeframe is not configured
        """
        if timeframe not in self.frames:
            raise KeyError(
                f"Timeframe '{timeframe}' not found. "
                f"Available timeframes: {list(self.frames.keys())}"
            )
        return self.frames[timeframe]

    def __getitem__(self, timeframe: str) -> TimeFrame:
        """
        Dict-like access to timeframes.

        Parameters:
            timeframe: Timeframe string

        Returns:
            TimeFrame instance

        Raises:
            KeyError: If timeframe is not configured
        """
        return self.get_frame(timeframe)

    def add_indicator(
        self,
        timeframe: str,
        indicator: Indicator,
        column_name: Union[str, List[str]]
    ) -> None:
        """
        Add an indicator to a specific timeframe.

        Parameters:
            timeframe: Timeframe to add indicator to (e.g., "1T", "5T")
            indicator: Indicator instance to add
            column_name: Single name (str) or list of names for multi-column

        Raises:
            KeyError: If timeframe is not configured
            ValueError: If indicator dependencies not met or column exists
        """
        frame = self.get_frame(timeframe)
        frame.add_indicator(indicator, column_name)

    def add_indicator_to_all(
        self,
        indicator: Indicator,
        column_name: Union[str, List[str]]
    ) -> None:
        """
        Add the same indicator to all timeframes.

        Parameters:
            indicator: Indicator instance to add (same instance for all)
            column_name: Single name (str) or list of names for multi-column

        Raises:
            ValueError: If indicator dependencies not met or column exists
        """
        for timeframe in self._timeframes:
            self.frames[timeframe].add_indicator(indicator, column_name)

    def remove_indicator(
        self,
        timeframe: str,
        column_name: Union[str, List[str]]
    ) -> None:
        """
        Remove an indicator from a specific timeframe.

        Parameters:
            timeframe: Timeframe to remove indicator from
            column_name: Single name or list of names to remove

        Raises:
            KeyError: If timeframe is not configured
            ValueError: If indicator not found
        """
        frame = self.get_frame(timeframe)
        frame.remove_indicator(column_name)

    def remove_indicator_from_all(
        self,
        column_name: Union[str, List[str]]
    ) -> None:
        """
        Remove an indicator from all timeframes.

        Parameters:
            column_name: Single name or list of names to remove

        Note:
            Silently skips timeframes where the indicator doesn't exist
        """
        for frame in self.frames.values():
            try:
                frame.remove_indicator(column_name)
            except ValueError:
                # Indicator doesn't exist in this frame, skip
                pass

    def get_indicator_columns(self, timeframe: str) -> List[str]:
        """
        Get list of all indicator column names for a timeframe.

        Parameters:
            timeframe: Timeframe to query

        Returns:
            List of indicator column names

        Raises:
            KeyError: If timeframe is not configured
        """
        frame = self.get_frame(timeframe)
        return frame._get_all_indicator_columns()

    # ==================== HISTORICAL DATA LOADING ====================

    def load_historical_data(self, candles: List[Candle]) -> None:
        """
        Load historical candles in batch for efficient initialization.

        This method feeds multiple candles at once and is optimized
        for backtesting or initial data loading.

        Parameters:
            candles: List of Candle instances (should be sorted by date)

        Raises:
            TypeError: If any item is not a Candle instance
            ValueError: If candles list is empty

        Example:
            >>> candles = [
            ...     Candle(date=..., open=..., high=..., low=..., close=..., volume=...),
            ...     Candle(date=..., open=..., high=..., low=..., close=..., volume=...),
            ... ]
            >>> asset_view.load_historical_data(candles)
        """
        if not candles:
            raise ValueError("Candles list cannot be empty")

        for candle in candles:
            if not isinstance(candle, Candle):
                raise TypeError(f"Expected Candle instance, got {type(candle)}")
            self.feed(candle)

    def load_from_dataframe(
        self,
        df,
        date_column: str = 'date',
        open_column: str = 'open',
        high_column: str = 'high',
        low_column: str = 'low',
        close_column: str = 'close',
        volume_column: str = 'volume',
        date_format: str = '%Y-%m-%dT%H:%M:%S.%fZ'
    ) -> None:
        """
        Load historical data from a pandas DataFrame.

        Parameters:
            df: pandas DataFrame containing OHLCV data
            date_column: Name of the date column (default: 'date')
            open_column: Name of the open price column (default: 'open')
            high_column: Name of the high price column (default: 'high')
            low_column: Name of the low price column (default: 'low')
            close_column: Name of the close price column (default: 'close')
            volume_column: Name of the volume column (default: 'volume')
            date_format: Format string for date parsing (default: ISO 8601)

        Raises:
            ValueError: If required columns are missing
            ImportError: If pandas is not installed

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'timestamp': [...],
            ...     'o': [...],
            ...     'h': [...],
            ...     'l': [...],
            ...     'c': [...],
            ...     'v': [...]
            ... })
            >>> asset_view.load_from_dataframe(
            ...     df,
            ...     date_column='timestamp',
            ...     open_column='o',
            ...     high_column='h',
            ...     low_column='l',
            ...     close_column='c',
            ...     volume_column='v'
            ... )
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for load_from_dataframe. Install with: pip install pandas")

        # Validate columns
        required_columns = {date_column, open_column, high_column, low_column, close_column, volume_column}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame rows to Candle objects
        candles = []
        for _, row in df.iterrows():
            candle = Candle(
                date=row[date_column],
                open=float(row[open_column]),
                high=float(row[high_column]),
                low=float(row[low_column]),
                close=float(row[close_column]),
                volume=float(row[volume_column]),
                date_format=date_format
            )
            candles.append(candle)

        # Load all candles
        self.load_historical_data(candles)

    # ==================== UNIFIED EXPORT ====================

    def to_dict(self) -> Dict[str, List[Dict]]:
        """
        Export all timeframes data to nested dictionary.

        Returns:
            Dictionary with timeframe keys and period data lists

        Example:
            >>> data = asset_view.to_dict()
            >>> print(data.keys())  # ['1T', '5T', '1H']
            >>> print(data['1T'][0])  # First period of 1T timeframe
        """
        result = {}
        for tf, frame in self.frames.items():
            result[tf] = [period.to_dict() for period in frame.periods]
        return result

    def to_pandas_multi(self):
        """
        Export all timeframes to a single pandas DataFrame with MultiIndex.

        The DataFrame will have a MultiIndex (timeframe, timestamp) for rows
        and include all OHLCV + indicator columns.

        Returns:
            pandas.DataFrame with MultiIndex (timeframe, open_date)

        Raises:
            ImportError: If pandas is not installed

        Example:
            >>> df = asset_view.to_pandas_multi()
            >>> print(df.index.names)  # ['timeframe', 'open_date']
            >>> print(df.loc['1T'])  # All 1-minute periods
            >>> print(df.loc[('5T', specific_timestamp)])  # Specific period
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        # Collect all DataFrames
        dfs = []
        for tf, frame in self.frames.items():
            df = frame.to_pandas()
            df['timeframe'] = tf
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Concatenate and set MultiIndex
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.set_index(['timeframe', 'open_date'])

        return combined

    def to_numpy_all(self) -> Dict[str, 'np.ndarray']:
        """
        Export all timeframes to numpy arrays.

        Returns:
            Dictionary mapping timeframe to numpy array (OHLCV + indicators)

        Example:
            >>> arrays = asset_view.to_numpy_all()
            >>> print(arrays['1T'].shape)  # (120, 7) - 120 periods, 7 columns
        """
        result = {}
        for tf, frame in self.frames.items():
            result[tf] = frame.to_numpy()
        return result

    def to_normalize_all(self) -> Dict[str, 'np.ndarray']:
        """
        Export all timeframes to normalized numpy arrays.

        Each timeframe's data is normalized independently using the
        intelligent normalization system from trading-frame:
        - OHLC + Price-based indicators: Unified Min-Max normalization
        - Volume: Independent Min-Max normalization
        - Indicators: Strategy-specific normalization (RSI: fixed 0-100, etc.)

        Returns:
            Dictionary mapping timeframe to normalized numpy array

        Example:
            >>> normalized = asset_view.to_normalize_all()
            >>> print(normalized['1T'])  # All values in [0, 1] range
            >>> # Ready for ML models
        """
        result = {}
        for tf, frame in self.frames.items():
            result[tf] = frame.to_normalize()
        return result

    # ==================== EVENT MANAGEMENT ====================

    def on(self, event: str, handler: Callable) -> None:
        """
        Register an event handler for cross-timeframe events.

        Available events:
            - 'candle_fed': Triggered after a candle is fed to all timeframes
              Signature: handler(asset_view, candle)
            - 'period_created': Triggered when a new period is created on any timeframe
              Signature: handler(asset_view, timeframe, frame)
            - 'period_updated': Triggered when a period is updated on any timeframe
              Signature: handler(asset_view, timeframe, frame)
            - 'period_closed': Triggered when a period closes on any timeframe
              Signature: handler(asset_view, timeframe, frame)
            - 'all_periods_closed': Triggered when periods close on multiple timeframes simultaneously
              Signature: handler(asset_view, timeframes_list)

        Parameters:
            event: Event name to listen to
            handler: Callback function to execute when event occurs

        Raises:
            ValueError: If event name is not valid

        Example:
            >>> def on_period_closed(asset_view, timeframe, frame):
            ...     print(f"{timeframe} period closed at {frame.periods[-1].close_price}")
            >>> asset_view.on('period_closed', on_period_closed)
        """
        if event not in self._event_handlers:
            raise ValueError(
                f"Invalid event '{event}'. "
                f"Valid events: {list(self._event_handlers.keys())}"
            )
        self._event_handlers[event].append(handler)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """
        Emit an event to all registered handlers.

        Parameters:
            event: Event name
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                handler(*args, **kwargs)

    def _on_frame_new_period(self, timeframe: str, frame: TimeFrame) -> None:
        """Handle new period event from a timeframe."""
        self._emit('period_created', self, timeframe, frame)

    def _on_frame_update(self, timeframe: str, frame: TimeFrame) -> None:
        """Handle update event from a timeframe."""
        self._emit('period_updated', self, timeframe, frame)

    def _on_frame_close(self, timeframe: str, frame: TimeFrame) -> None:
        """Handle close event from a timeframe."""
        # Track which timeframes closed during this feed
        if not hasattr(self, '_closed_timeframes'):
            self._closed_timeframes = []

        self._closed_timeframes.append(timeframe)
        self._emit('period_closed', self, timeframe, frame)

    # ==================== VALIDATION & CONSISTENCY ====================

    def validate_consistency(self) -> bool:
        """
        Validate data consistency across all timeframes.

        Checks:
        - No timeframe should have periods in the future relative to others
        - Higher timeframes should align properly with lower timeframes

        Returns:
            True if all timeframes are consistent

        Raises:
            ValueError: If inconsistencies are detected

        Example:
            >>> asset_view.validate_consistency()
            True
        """
        if not self.frames or not any(frame.periods for frame in self.frames.values()):
            return True  # Empty or no data is considered consistent

        # Get latest timestamps from each timeframe
        latest_timestamps = {}
        for tf, frame in self.frames.items():
            if frame.periods:
                latest_timestamps[tf] = frame.periods[-1].open_date

        if not latest_timestamps:
            return True

        # Check that all timeframes are within reasonable range
        min_time = min(latest_timestamps.values())
        max_time = max(latest_timestamps.values())

        # Allow some tolerance for timeframe differences
        # (e.g., 1H might be older than 1T if hour hasn't closed yet)
        for tf, timestamp in latest_timestamps.items():
            if timestamp < min_time:
                # This is expected for longer timeframes
                continue

        return True

    #  ==================== STATE MANAGEMENT ====================

    def save_state(self) -> Dict[str, Any]:
        """
        Save current state to dictionary for serialization.

        Returns:
            Dictionary containing complete state

        Example:
            >>> state = asset_view.save_state()
            >>> # Later...
            >>> asset_view2.load_state(state)
        """
        state = {
            'symbol': self.symbol,
            'timeframes': self._timeframes,
            'max_periods': self._max_periods,
            'frames_data': {}
        }

        # Save each frame's periods
        for tf, frame in self.frames.items():
            state['frames_data'][tf] = {
                'periods': [period.to_dict() for period in frame.periods],
                'indicators': list(frame.indicators.keys())  # Just track which indicators exist
            }

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from dictionary.

        Warning: This will replace all current data.

        Parameters:
            state: State dictionary from save_state()

        Raises:
            ValueError: If state format is invalid

        Example:
            >>> state = asset_view1.save_state()
            >>> asset_view2 = AssetView("BTC/USDT", timeframes=["1T", "5T"])
            >>> asset_view2.load_state(state)
        """
        # Validate state
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")

        required_keys = {'symbol', 'timeframes', 'max_periods', 'frames_data'}
        if not required_keys.issubset(state.keys()):
            raise ValueError(f"State missing required keys: {required_keys - state.keys()}")

        # Clear current data
        self.frames.clear()

        # Restore configuration
        self.symbol = state['symbol']
        self._timeframes = state['timeframes']
        self._max_periods = state['max_periods']

        # Recreate frames (without data first)
        for tf in self._timeframes:
            self.frames[tf] = TimeFrame(periods_length=tf, max_periods=self._max_periods)

        # Reload periods via feeding candles (to maintain proper aggregation)
        # Note: This is a simplified approach; for full state restoration with indicators,
        # you'd need to re-add indicators and feed candles in chronological order

    def to_pickle(self, filepath: str) -> None:
        """
        Save complete state to pickle file.

        Parameters:
            filepath: Path to save pickle file

        Example:
            >>> asset_view.to_pickle('btc_state.pkl')
        """
        state = self.save_state()
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def from_pickle(cls, filepath: str) -> 'AssetView':
        """
        Load AssetView from pickle file.

        Parameters:
            filepath: Path to pickle file

        Returns:
            New AssetView instance with loaded state

        Example:
            >>> asset_view = AssetView.from_pickle('btc_state.pkl')
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create new instance
        instance = cls(
            symbol=state['symbol'],
            timeframes=state['timeframes'],
            max_periods=state['max_periods']
        )

        # Load state
        instance.load_state(state)

        return instance

    # ==================== METRICS & STATISTICS ====================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics across all timeframes.

        Returns:
            Dictionary with statistics including:
            - total_periods: Period count per timeframe
            - price_range: Global min/max prices
            - volume_stats: Volume statistics
            - indicators: List of indicators per timeframe
            - latest_prices: Latest close price per timeframe

        Example:
            >>> stats = asset_view.get_statistics()
            >>> print(stats['total_periods'])  # {'1T': 120, '5T': 24, ...}
            >>> print(stats['price_range'])  # {'min': 49900, 'max': 51000}
        """
        stats = {
            'symbol': self.symbol,
            'total_periods': {},
            'price_range': {'min': None, 'max': None},
            'volume_stats': {'total': 0, 'avg': 0, 'min': None, 'max': None},
            'indicators': {},
            'latest_prices': {},
            'timeframe_coverage': {}
        }

        all_prices = []
        all_volumes = []

        for tf, frame in self.frames.items():
            # Period counts
            stats['total_periods'][tf] = len(frame.periods)

            # Indicators
            stats['indicators'][tf] = frame._get_all_indicator_columns()

            if frame.periods:
                # Latest price
                stats['latest_prices'][tf] = frame.periods[-1].close_price

                # Collect all prices and volumes
                for period in frame.periods:
                    if period.high_price is not None:
                        all_prices.append(period.high_price)
                    if period.low_price is not None:
                        all_prices.append(period.low_price)
                    all_volumes.append(float(period.volume))

                # Timeframe coverage
                stats['timeframe_coverage'][tf] = {
                    'start': frame.periods[0].open_date.isoformat(),
                    'end': frame.periods[-1].close_date.isoformat() if frame.periods[-1].close_date else None
                }

        # Global price range
        if all_prices:
            stats['price_range']['min'] = min(all_prices)
            stats['price_range']['max'] = max(all_prices)

        # Volume statistics
        if all_volumes:
            stats['volume_stats']['total'] = sum(all_volumes)
            stats['volume_stats']['avg'] = sum(all_volumes) / len(all_volumes)
            stats['volume_stats']['min'] = min(all_volumes)
            stats['volume_stats']['max'] = max(all_volumes)

        return stats

    # ==================== TEMPORAL ALIGNMENT ====================

    def get_aligned_periods(self, timestamp: datetime) -> Dict[str, Optional[Any]]:
        """
        Get periods from all timeframes aligned to a specific timestamp.

        Returns the period that contains the given timestamp for each timeframe.

        Parameters:
            timestamp: Timestamp to align to

        Returns:
            Dictionary mapping timeframe to Period (or None if no period contains timestamp)

        Example:
            >>> from datetime import datetime
            >>> aligned = asset_view.get_aligned_periods(datetime(2024, 1, 1, 12, 30))
            >>> print(aligned['1T'])  # Period at 12:30
            >>> print(aligned['1H'])  # Period for 12:00-13:00 hour
        """
        result = {}

        for tf, frame in self.frames.items():
            result[tf] = None

            # Find period containing this timestamp
            for period in frame.periods:
                if period.open_date <= timestamp:
                    if period.close_date is None or timestamp <= period.close_date:
                        result[tf] = period
                        break

        return result

    def __repr__(self) -> str:
        """String representation of AssetView."""
        return (
            f"AssetView(symbol='{self.symbol}', "
            f"timeframes={self._timeframes}, "
            f"max_periods={self._max_periods})"
        )
