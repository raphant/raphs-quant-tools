# Trade Management Module

This module provides tools for managing and reporting on trading activities in a reinforcement learning context.

## Components

### TradeManager (`trade_manager.py`)

The `TradeManager` class handles all trade-related operations and calculations. It maintains the state of trades and provides various performance metrics.

#### Key Features

- Trade lifecycle management (open/close trades)
- Capital management and profit tracking
- Performance metrics calculation:
  - Sortino Ratio
  - Sharpe Ratio
  - Maximum Drawdown
  - Profit Factor
  - Volatility
  - Market Correlation

#### Main Properties

- `closed_trades`: List of completed trades
- `open_trades`: List of active trades
- `closed_capital`: Available capital from closed trades
- `average_profit_percent`: Average profit percentage across all trades
- `total_profit`: Sum of all closed trade profits

#### Performance Metrics

- Sortino Ratio: Measures risk-adjusted return focusing on downside volatility
- Sharpe Ratio: Measures risk-adjusted return using standard deviation
- Max Drawdown: Maximum observed loss from a peak to a trough
- Profit Factor: Ratio of gross profit to gross loss
- Volatility: Standard deviation of returns
- Market Correlation: Correlation between strategy returns and market returns

### TradeReporter (`trade_reporter.py`)

The `TradeReporter` class provides rich, formatted reporting capabilities for trade analysis and performance visualization.

#### Key Features

- Detailed trade list generation with cumulative profits
- Performance summary tables
- Rich formatting using the `rich` library
- Customizable console output

#### Main Methods

- `get_trade_list()`: Generates a detailed table of all trades with profit/loss information
- `get_summary_table()`: Creates a comprehensive summary of trading performance metrics

#### Report Components

- Trade-by-trade breakdown
- Capital and returns summary
- Trade statistics (win rate, average profit)
- Risk metrics (drawdown, volatility)
- Performance ratios (Sortino, Sharpe, Market Correlation)

## Usage Example

```python
from rqt.rl.tm.trade_manager import TradeManager
from rqt.rl.tm.trade_reporter import TradeReporter
from rqt.rl.data_provider import DataProvider

# Initialize components
dp = DataProvider(...)
tm = TradeManager(initial_capital=10000, data_provider=dp)
reporter = TradeReporter(tm)

# Open a trade
tm.open_trade(stake_amount=1000, open_price=100, open_date=pd.Timestamp('2023-01-01'))

# Close a trade
tm.close_trade(trade_id=0, close_price=110, close_date=pd.Timestamp('2023-01-02'))

# Generate reports
reporter.get_trade_list()  # Get detailed trade list
reporter.get_summary_table(current_price=110)  # Get performance summary
```
