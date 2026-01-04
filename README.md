# Trading Asset View

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-50%20passed-success.svg)](https://github.com/Morgiver/trading-asset-view)

**Unified multi-timeframe orchestration layer for trading applications** built on top of [trading-frame](https://github.com/Morgiver/trading-frame).

## Overview

`trading-asset-view` provides a powerful abstraction for managing multiple timeframes of a single trading asset. Feed candle data once, and automatically synchronize all configured timeframes with event-driven architecture, advanced export capabilities, and ML-ready normalized data.

## Features

### Core Capabilities
- 📊 **Multi-timeframe management**: Manage 1m, 5m, 1h, 1d timeframes for a single asset
- 🔄 **Synchronized feeding**: Feed once, update all timeframes automatically
- 📈 **Indicator support**: Add technical indicators to individual or all timeframes
- 🎯 **Unified API**: Simple interface with dict-like timeframe access

### Advanced Features (New!)
- 🔔 **Cross-timeframe events**: React to period creation, updates, and closes
- 📥 **Historical data loading**: Batch-load from lists or pandas DataFrames
- 📤 **Unified exports**: Export all timeframes to dict, pandas, numpy, or **normalized arrays**
- ✅ **Consistency validation**: Verify data integrity across timeframes
- 💾 **State management**: Save/load complete state with pickle support
- 📊 **Global statistics**: Aggregate metrics across all timeframes
- ⏰ **Temporal alignment**: Get synchronized periods at specific timestamps

### Machine Learning Ready
- 🤖 **Normalized data export**: Get ML-ready normalized data via `to_normalize_all()`
- 🎯 **Intelligent normalization**: Leverages trading-frame's smart normalization system
  - OHLC + price indicators: Unified Min-Max normalization
  - Volume: Independent normalization
  - RSI: Fixed 0-100 → [0,1] range
  - Indicators: Strategy-specific normalization

## Installation

```bash
pip install git+https://github.com/Morgiver/trading-asset-view.git
```

## Quick Start

```python
from trading_frame import Candle
from trading_frame.indicators import RSI, SMA
from trading_asset_view import AssetView

# Initialize with multiple timeframes
asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H", "1D"])

# Feed a candle (automatically routed to all timeframes)
candle = Candle(
    date=1234567890,
    open=50000.0,
    high=51000.0,
    low=49000.0,
    close=50500.0,
    volume=123.45
)
asset_view.feed(candle)

# Access specific timeframe
frame_1m = asset_view["1T"]  # Dict-like access
periods = frame_1m.periods

# Add indicators
rsi = RSI(length=14)
asset_view.add_indicator("1T", rsi, "RSI_14")

sma = SMA(period=20)
asset_view.add_indicator_to_all(sma, "SMA_20")  # All timeframes
```

## Complete Documentation

### Table of Contents
- [Warm-Up with Prefill](#warm-up-with-prefill)
- [Event Management](#event-management)
- [Historical Data Loading](#historical-data-loading)
- [Unified Exports](#unified-exports)
- [State Management](#state-management)
- [Statistics & Metrics](#statistics--metrics)
- [Temporal Alignment](#temporal-alignment)
- [Indicator Management](#indicator-management)

---

## Warm-Up with Prefill

Efficiently fill all timeframes during initialization or backtesting warm-up phase.

### Why Prefill?

When starting live trading or backtesting, you need historical data to calculate indicators (e.g., RSI needs 14 periods). Instead of manually tracking when each timeframe is ready, use `prefill()` to automate this process.

### Basic Usage

```python
from datetime import datetime
from trading_frame import Candle
from trading_asset_view import AssetView

asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H"])

# Prefill until all timeframes have enough data (default: max_periods)
for candle in historical_data:
    if asset_view.prefill(candle):
        print("All timeframes ready!")
        break  # Warm-up complete

# Now ready for live trading
for candle in live_data:
    asset_view.feed(candle)  # Use normal feed
```

### Prefill with Target Timestamp

```python
# Option 1: Fill until timestamp (validated mode - RECOMMENDED)
# Raises InsufficientDataError if frames not full at target timestamp
from trading_frame.exceptions import InsufficientDataError

target_date = datetime(2024, 1, 1, 12, 0, 0)
target_ts = target_date.timestamp()

try:
    for candle in historical_data:
        if asset_view.prefill(candle, target_timestamp=target_ts, require_full=True):
            break  # All frames full at target date
except InsufficientDataError as e:
    print(f"Not enough historical data before target date: {e}")

# Option 2: Fill until timestamp (relaxed mode)
# Stops at timestamp regardless of period count
for candle in historical_data:
    if asset_view.prefill(candle, target_timestamp=target_ts, require_full=False):
        break  # Reached target date
```

### Monitor Progress

```python
# Track progress during prefill
for i, candle in enumerate(historical_data):
    is_complete = asset_view.prefill(candle)

    # Print progress every 10 candles
    if i % 10 == 0:
        print(f"Prefill in progress... ({i+1} candles processed)")

    if is_complete:
        print(f"Prefill complete after {i+1} candles")
        break
```

### Backtest Example

```python
from trading_frame.indicators import RSI, SMA

asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H"])

# Add indicators
asset_view.add_indicator("1T", RSI(length=14), "RSI_14")
asset_view.add_indicator_to_all(SMA(period=20), "SMA_20")

# Phase 1: Warm-up (silent, no events, no trading signals)
warmup_complete = False
for candle in all_data:
    if not warmup_complete:
        if asset_view.prefill(candle):  # Fill to max_periods capacity
            print("Warm-up complete, starting backtest...")
            warmup_complete = True
    else:
        # Phase 2: Live backtest (events enabled, generate signals)
        asset_view.feed(candle)

        # Your trading logic here
        rsi = asset_view["1T"].periods[-1].RSI_14
        if rsi and rsi < 30:
            print(f"BUY signal at {candle.date}")
```

### Key Features

- ✅ **Silent operation**: No events emitted during prefill (efficient warm-up)
- ✅ **Simple boolean return**: Returns `True` when ALL timeframes reach target
- ✅ **Three modes**: Default (fill to capacity), timestamp validated, timestamp relaxed
- ✅ **Automatic**: Handles different timeframe completion rates (1T fills faster than 1H)
- ✅ **Error handling**: Raises `InsufficientDataError` in validated mode if not enough data

---

## Event Management

React to cross-timeframe events with a flexible event system.

### Available Events

- `candle_fed`: After candle fed to all timeframes
- `period_created`: When new period created on any timeframe
- `period_updated`: When period updated on any timeframe
- `period_closed`: When period closes on any timeframe
- `all_periods_closed`: When multiple timeframes close simultaneously

### Example

```python
from trading_asset_view import AssetView

asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H"])

# Register event handlers
def on_period_closed(asset_view, timeframe, frame):
    period = frame.periods[-1]
    print(f"{timeframe} closed: ${period.close_price:,.2f}")

asset_view.on('period_closed', on_period_closed)

# Detect multi-timeframe closes
def on_multiple_closes(asset_view, timeframes):
    print(f"Multiple closes: {timeframes}")
    # Example: ['1T', '5T'] closed at same time

asset_view.on('all_periods_closed', on_multiple_closes)

# Feed candles - events will trigger automatically
for candle in candles:
    asset_view.feed(candle)
```

---

## Historical Data Loading

Efficiently load large datasets for backtesting or initialization.

### From Candle List

```python
from datetime import datetime, timedelta
from trading_frame import Candle

# Generate historical candles
base_time = datetime(2024, 1, 1, 12, 0, 0)
candles = [
    Candle(
        date=base_time + timedelta(minutes=i),
        open=50000.0 + i * 10,
        high=50100.0 + i * 10,
        low=49900.0 + i * 10,
        close=50050.0 + i * 10,
        volume=100.0
    )
    for i in range(1000)
]

# Load all at once
asset_view.load_historical_data(candles)
```

### From pandas DataFrame

```python
import pandas as pd

# Load from DataFrame (common format from exchanges)
df = pd.DataFrame({
    'timestamp': [...],  # Unix timestamps or datetime
    'o': [...],          # Open prices
    'h': [...],          # High prices
    'l': [...],          # Low prices
    'c': [...],          # Close prices
    'v': [...]           # Volumes
})

asset_view.load_from_dataframe(
    df,
    date_column='timestamp',
    open_column='o',
    high_column='h',
    low_column='l',
    close_column='c',
    volume_column='v'
)
```

---

## Unified Exports

Export all timeframes in various formats for analysis or ML applications.

### to_dict() - Python Dictionary

```python
data = asset_view.to_dict()
# {
#   '1T': [{'date': ..., 'open_price': ..., ...}, ...],
#   '5T': [{'date': ..., 'open_price': ..., ...}, ...],
#   '1H': [...]
# }

# Access specific timeframe periods
periods_1m = data['1T']
```

### to_pandas_multi() - Multi-Index DataFrame

```python
df = asset_view.to_pandas_multi()
# MultiIndex (timeframe, open_date)

# Access by timeframe
df_1m = df.loc['1T']

# Access specific period
df.loc[('5T', specific_timestamp)]

# Filter and analyze
df.groupby(level='timeframe')['volume'].sum()
```

### to_numpy_all() - NumPy Arrays

```python
arrays = asset_view.to_numpy_all()
# {
#   '1T': np.array([...]),  # Shape: (n_periods, n_features)
#   '5T': np.array([...]),
#   '1H': np.array([...])
# }

# Use for numerical analysis
import numpy as np
mean_price = np.mean(arrays['1T'][:, 3])  # Column 3 = close_price
```

### to_normalize_all() - ML-Ready Normalized Data 🤖

```python
normalized = asset_view.to_normalize_all()
# All values in [0, 1] range - ready for ML models

# Use with machine learning
import numpy as np

# Stack all timeframes for multi-timeframe model
X_1m = normalized['1T']  # Shape: (n_periods, n_features)
X_5m = normalized['5T']
X_1h = normalized['1H']

# Feed to neural network
model.fit([X_1m, X_5m, X_1h], y)
```

**Normalization Strategy:**
- OHLC + price indicators (SMA, Bollinger Bands): Unified Min-Max
- Volume: Independent Min-Max
- RSI: Fixed 0-100 → [0,1]
- MACD: Min-Max on own values

---

## State Management

Save and restore complete AssetView state for persistence.

### Save/Load State (Dict)

```python
# Save current state
state = asset_view.save_state()

# Later - restore to new instance
asset_view2 = AssetView("BTC/USDT", timeframes=["1T", "5T"])
asset_view2.load_state(state)
```

### Pickle Serialization

```python
# Save to file
asset_view.to_pickle('btc_state.pkl')

# Load from file
asset_view = AssetView.from_pickle('btc_state.pkl')
```

---

## Statistics & Metrics

Get comprehensive statistics across all timeframes.

```python
stats = asset_view.get_statistics()

# Output structure:
{
    'symbol': 'BTC/USDT',
    'total_periods': {
        '1T': 120,
        '5T': 24,
        '1H': 2
    },
    'price_range': {
        'min': 49900.0,
        'max': 51000.0
    },
    'volume_stats': {
        'total': 12000.0,
        'avg': 100.0,
        'min': 95.0,
        'max': 105.0
    },
    'indicators': {
        '1T': ['RSI_14', 'SMA_20'],
        '5T': ['SMA_20'],
        '1H': ['SMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
    },
    'latest_prices': {
        '1T': 50500.0,
        '5T': 50500.0,
        '1H': 50500.0
    },
    'timeframe_coverage': {
        '1T': {
            'start': '2024-01-01T12:00:00',
            'end': '2024-01-01T13:59:59.999999'
        },
        ...
    }
}
```

---

## Temporal Alignment

Get synchronized periods across timeframes for a specific timestamp.

```python
from datetime import datetime

# Get all periods containing this timestamp
aligned = asset_view.get_aligned_periods(datetime(2024, 1, 1, 12, 30, 0))

# {
#   '1T': Period(12:30:00 - 12:30:59),
#   '5T': Period(12:30:00 - 12:34:59),
#   '1H': Period(12:00:00 - 12:59:59)
# }

# Access aligned data
period_1m = aligned['1T']
period_1h = aligned['1H']

if period_1m and period_1h:
    print(f"1m close: {period_1m.close_price}")
    print(f"1h close: {period_1h.close_price}")
```

---

## Indicator Management

Comprehensive indicator support with flexible configuration.

### Add to Specific Timeframe

```python
from trading_frame.indicators import RSI, SMA, BollingerBands

# Single timeframe
rsi = RSI(length=14)
asset_view.add_indicator("1T", rsi, "RSI_14")

# Multi-column indicator
bb = BollingerBands(period=20, std_dev=2.0)
asset_view.add_indicator("1H", bb, ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"])
```

### Add to All Timeframes

```python
# Same indicator across all timeframes
sma = SMA(period=20)
asset_view.add_indicator_to_all(sma, "SMA_20")

# Now all timeframes have SMA_20
```

### Remove Indicators

```python
# From specific timeframe
asset_view.remove_indicator("1T", "RSI_14")

# From all timeframes
asset_view.remove_indicator_from_all("SMA_20")
```

### List Indicators

```python
# Get indicator columns for a timeframe
indicators = asset_view.get_indicator_columns("1T")
# ['RSI_14', 'SMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
```

---

## Advanced Example: ML Pipeline

Complete example with events, indicators, and normalized export.

```python
from datetime import datetime, timedelta
from trading_frame import Candle
from trading_frame.indicators import RSI, SMA, BollingerBands
from trading_asset_view import AssetView

# Setup
asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "15T", "1H"])

# Add indicators
asset_view.add_indicator_to_all(SMA(period=20), "SMA_20")
asset_view.add_indicator("1T", RSI(length=14), "RSI_14")
asset_view.add_indicator("1H", BollingerBands(period=20),
                         ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"])

# Event tracking
closes_detected = []

def on_all_closes(av, timeframes):
    closes_detected.append(timeframes)
    print(f"Multiple closes: {timeframes}")

asset_view.on('all_periods_closed', on_all_closes)

# Load historical data
base_time = datetime(2024, 1, 1, 12, 0, 0)
candles = [
    Candle(
        date=base_time + timedelta(minutes=i),
        open=50000.0 + (i % 20 - 10) * 50,
        high=50100.0 + (i % 20 - 10) * 50,
        low=49900.0 + (i % 20 - 10) * 50,
        close=50050.0 + (i % 20 - 10) * 50,
        volume=100.0 + i % 10
    )
    for i in range(120)  # 2 hours of 1-minute data
]

asset_view.load_historical_data(candles)

# Get statistics
stats = asset_view.get_statistics()
print(f"Total periods: {stats['total_periods']}")
print(f"Price range: {stats['price_range']}")

# Export normalized data for ML
normalized = asset_view.to_normalize_all()

# Build features for model
X_1m = normalized['1T']   # (120, n_features)
X_5m = normalized['5T']   # (24, n_features)
X_15m = normalized['15T'] # (8, n_features)
X_1h = normalized['1H']   # (2, n_features)

# Ready for neural network
# model.fit([X_1m, X_5m, X_15m, X_1h], y)

# Save state for later
asset_view.to_pickle('btc_trained_state.pkl')
```

---

## API Reference

### AssetView

#### Constructor
```python
AssetView(symbol: str, timeframes: List[str], max_periods: int = 250)
```

#### Data Methods
- `feed(candle: Candle)` - Feed single candle to all timeframes
- `load_historical_data(candles: List[Candle])` - Batch load candles
- `load_from_dataframe(df, **columns)` - Load from pandas DataFrame

#### Frame Access
- `get_frame(timeframe: str) -> TimeFrame` - Get specific timeframe
- `__getitem__(timeframe: str) -> TimeFrame` - Dict-like access

#### Indicator Methods
- `add_indicator(timeframe, indicator, column_name)` - Add to one timeframe
- `add_indicator_to_all(indicator, column_name)` - Add to all timeframes
- `remove_indicator(timeframe, column_name)` - Remove from one
- `remove_indicator_from_all(column_name)` - Remove from all
- `get_indicator_columns(timeframe) -> List[str]` - List indicators

#### Event Methods
- `on(event: str, handler: Callable)` - Register event handler

#### Export Methods
- `to_dict() -> Dict[str, List[Dict]]` - Export to dictionary
- `to_pandas_multi() -> DataFrame` - Export to MultiIndex DataFrame
- `to_numpy_all() -> Dict[str, ndarray]` - Export to numpy arrays
- `to_normalize_all() -> Dict[str, ndarray]` - Export normalized arrays

#### State Methods
- `save_state() -> Dict` - Save state to dictionary
- `load_state(state: Dict)` - Load state from dictionary
- `to_pickle(filepath: str)` - Save to pickle file
- `from_pickle(filepath: str) -> AssetView` - Load from pickle (classmethod)

#### Analysis Methods
- `get_statistics() -> Dict` - Get comprehensive statistics
- `get_aligned_periods(timestamp: datetime) -> Dict` - Get aligned periods
- `validate_consistency() -> bool` - Validate data consistency

#### Properties
- `symbol: str` - Asset symbol
- `timeframes: List[str]` - List of timeframe strings
- `frames: Dict[str, TimeFrame]` - All TimeFrame instances

---

## License

MIT

## Contributing

Issues and pull requests welcome at [GitHub repository](https://github.com/Morgiver/trading-asset-view)

## Related Projects

- [trading-frame](https://github.com/Morgiver/trading-frame) - Core timeframe aggregation engine
