import logging
from dataclasses import dataclass

import pandas as pd

from rqt.rl.data_provider import DataProvider

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    id: int
    stake_amount: float
    open_price: float
    open_date: pd.Timestamp
    close_date: pd.Timestamp = None
    close_price: float = None

    @property
    def profit(self):
        if self.is_open:
            raise ValueError("Trade not closed yet.")
        return (self.close_price - self.open_price) * (
            self.stake_amount / self.open_price
        )

    @property
    def is_open(self):
        return self.close_price is None


class TradeManager:
    def __init__(self, initial_capital: float, data_provider: DataProvider):
        self.initial_capital = initial_capital
        self.trades: list[Trade] = []
        self.dp = data_provider
        self.all_prices: dict[pd.Timestamp, float] = self.dp.get_prices(column="close")

    @property
    def closed_trades(self):
        return [trade for trade in self.trades if not trade.is_open]

    @property
    def open_trades(self):
        return [trade for trade in self.trades if trade.is_open]

    @property
    def closed_capital(self):
        """
        Calculate the capital from closed trades.

        Returns:
        capital: The capital after accounting for profits from closed trades.
        """
        capital = self.initial_capital
        for trade in self.closed_trades:
            capital += trade.profit
        return capital

    @property
    def average_profit_percent(self):
        closed_trades = self.closed_trades
        if not closed_trades:
            return 0
        total_profit = sum(trade.profit for trade in closed_trades)
        average_profit = total_profit / len(closed_trades)
        return (average_profit / self.initial_capital) * 100

    @property
    def total_profit(self):
        return sum(trade.profit for trade in self.closed_trades)
    
    @property
    def current_trade(self):
        return self.open_trades[0] if self.open_trades else None

    @property
    def last_completed_trade(self):
        return self.closed_trades[-1] if self.closed_trades else None
    
    @property
    def max_drawdown(self) -> float:
        """Calculate the maximum drawdown percentage."""
        if not self.closed_trades:
            return 0.0
        
        peak = self.initial_capital
        max_drawdown = 0
        current_capital = self.initial_capital
        
        for trade in self.closed_trades:
            current_capital += trade.profit
            peak = max(peak, current_capital)
            drawdown = (peak - current_capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100  # Convert to percentage
    
    @property
    def profit_factor(self) -> float:
        """Calculate the profit factor (gross profit / gross loss)."""
        if not self.closed_trades:
            return 0.0
        
        gross_profit = sum(trade.profit for trade in self.closed_trades if trade.profit > 0)
        gross_loss = abs(sum(trade.profit for trade in self.closed_trades if trade.profit < 0))
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')

    @property
    def volatility(self) -> float:
        """Calculate the trading strategy's volatility (standard deviation of returns)."""
        if not self.closed_trades:
            return 0.0
        
        returns = [trade.profit for trade in self.closed_trades]
        if not returns:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    @property
    def market_correlation(self) -> float:
        """Calculate correlation between strategy returns and market returns."""
        if not self.closed_trades:
            return 0.0
        
        # Get market returns (using close prices)
        market_prices = pd.Series(self.dp.get_prices('close'))
        market_returns = market_prices.pct_change().dropna()
        
        # Get strategy returns for corresponding dates
        strategy_returns = []
        dates = []
        current_capital = self.initial_capital
        
        for trade in self.closed_trades:
            if trade.close_date in market_returns.index:
                strategy_returns.append(trade.profit / current_capital)
                dates.append(trade.close_date)
                current_capital += trade.profit
        
        if not strategy_returns:
            return 0.0
        
        # Calculate correlation
        strategy_series = pd.Series(strategy_returns, index=dates)
        market_series = market_returns[strategy_series.index]
        
        if len(strategy_series) < 2 or len(market_series) < 2:
            return 0.0
            
        return strategy_series.corr(market_series)
    
    def calculate_open_capital(self, current_date: pd.Timestamp):
        """
        Calculate the capital from open trades based on the current date.

        Parameters:
        current_date: The current date to calculate open capital for.

        Returns:
        capital: The capital after accounting for unrealized profits from open trades.
        """
        capital = 0
        for trade in self.open_trades:
            if trade.close_price is None:
                current_price = self.all_prices[current_date]
                unrealized_profit = (current_price - trade.open_price) * (
                    trade.stake_amount / trade.open_price
                )
                capital += unrealized_profit
        return capital

    def open_trade(
        self, stake_amount: float, open_date: pd.Timestamp
    ):
        if stake_amount > self.closed_capital:
            raise ValueError("Insufficient capital to open trade.")

        current_price = self.all_prices[open_date]
        new_trade = Trade(id=len(self.trades), stake_amount=stake_amount, open_price=current_price, open_date=open_date)
        self.trades.append(new_trade)

    def close_trade(self, trade_id: int, close_date: pd.Timestamp):
        """
        Close a trade using the close price from the data provider for the given date.

        Parameters:
        trade_id: The ID of the trade to close
        close_date: The date to close the trade on

        Raises:
        ValueError: If trade is not found or already closed
        """
        try:
            trade_idx = [t.id for t in self.trades].index(trade_id)
        except ValueError:
            raise ValueError(f"Trade with ID {trade_id} not found.")
        
        if not self.trades[trade_idx].is_open:
            raise ValueError(f"Trade {trade_id} is already closed.")

        self.trades[trade_idx].close_price = self.all_prices[close_date]
        self.trades[trade_idx].close_date = close_date

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0):
        closed_trades = self.closed_trades
        if not closed_trades:
            return 0
        returns = [trade.profit for trade in closed_trades]
        downside_returns = [min(0, r - risk_free_rate) for r in returns]
        expected_return = sum(returns) / len(returns) - risk_free_rate
        downside_deviation = (
            sum(d**2 for d in downside_returns) / len(downside_returns)
        ) ** 0.5
        if downside_deviation == 0:
            return float("inf")
        return expected_return / downside_deviation

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate the Sharpe ratio of the trading strategy."""
        if not self.closed_trades:
            return 0.0
        
        returns = [trade.profit for trade in self.closed_trades]
        if not returns:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        if std_dev == 0:
            return float('inf') if avg_return > 0 else float('-inf')
        
        return (avg_return - risk_free_rate) / std_dev

    def reset(self):
        # Reset the TradeManagement to its initial state
        self.trades = []
