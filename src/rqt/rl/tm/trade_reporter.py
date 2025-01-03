import logging
from typing import Optional, List, Dict

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from rqt.rl.tm.trade_manager import TradeManager

logger = logging.getLogger(__name__)


class TradeReporter:
    def __init__(self, trade_manager: TradeManager):
        """
        Initialize the trade reporter with a trade manager instance.

        Args:
            trade_manager: Instance of TradeManagement to report on
        """
        self.tm = trade_manager

    def get_trade_list(self, console: Optional["Console"] = None) -> Optional["Table"]:
        """
        Generate a detailed trade list with cumulative profits.

        Args:
            console: Optional Rich Console instance for direct printing.
                    If None, returns the table object instead.

        Returns:
            Optional[Table]: Rich Table object if console is None, otherwise None
        """
        if not self.tm.closed_trades:
            return None

        # Create trades summary
        trades_summary = []
        for trade in self.tm.closed_trades:
            trades_summary.append(
                {
                    "id": trade.id,
                    "open_date": trade.open_date,
                    "close_date": trade.close_date,
                    "stake": trade.stake_amount,
                    "open_price": trade.open_price,
                    "close_price": trade.close_price,
                    "profit": trade.profit,
                }
            )

        trade_table = Table(title="\n📜 Detailed Trade List", box=box.ROUNDED)
        trade_table.add_column("ID", justify="right", style="dim")
        trade_table.add_column("Open Date", justify="right")
        trade_table.add_column("Close Date", justify="right")
        trade_table.add_column("Stake ($)", justify="right")
        trade_table.add_column("Open Price", justify="right")
        trade_table.add_column("Close Price", justify="right")
        trade_table.add_column("Profit/Loss ($)", justify="right", style="green")
        trade_table.add_column("Return (%)", justify="right", style="green")
        trade_table.add_column("Cum. Profit ($)", justify="right", style="green")

        cumulative_profit = 0
        for trade in trades_summary:
            profit_color = "green" if trade["profit"] > 0 else "red"
            return_pct = (trade["profit"] / trade["stake"]) * 100
            cumulative_profit += trade["profit"]
            cum_profit_color = "green" if cumulative_profit > 0 else "red"

            trade_table.add_row(
                str(trade["id"]),
                trade["open_date"].strftime("%Y-%m-%d %H:%M"),
                trade["close_date"].strftime("%Y-%m-%d %H:%M"),
                f"{trade['stake']:.2f}",
                f"{trade['open_price']:.2f}",
                f"{trade['close_price']:.2f}",
                f"[{profit_color}]{trade['profit']:.2f}[/]",
                f"[{profit_color}]{return_pct:+.1f}%[/]",
                f"[{cum_profit_color}]{cumulative_profit:,.2f}[/]",
            )

        if console:
            console.print(trade_table)
            return None

        return trade_table

    def get_summary_table(
        self, current_price: float = None, console: Optional["Console"] = None
    ) -> Optional["Table"]:
        """
        Generate a summary table of trading performance metrics.

        Args:
            current_price: Optional current price for calculating open trade values
            console: Optional Rich Console instance for direct printing.
                    If None, returns the table object instead.

        Returns:
            Optional[Table]: Rich Table object if console is None, otherwise None
        """
        ohlcv: pd.DataFrame = self.tm.dp._pre_normalized_data
        table = Table(title="📊 Trading Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        final_capital = self.tm.closed_capital
        if self.tm.open_trades and current_price is not None:
            final_capital += self.tm.calculate_open_capital(current_price)

        # Calculate metrics
        closed_trades = self.tm.closed_trades
        n_trades = len(closed_trades)
        if n_trades > 0:
            profits = [trade.profit for trade in closed_trades]
            win_rate = sum(1 for p in profits if p > 0) / n_trades
            avg_profit = sum(profits) / n_trades
            total_return = ((final_capital - self.tm.initial_capital) / self.tm.initial_capital) * 100
        else:
            win_rate = None
            avg_profit = None
            total_return = 0.0

        # Capital and Returns
        table.add_row("💰 Initial Capital", f"${self.tm.initial_capital:,.2f}")
        table.add_row("💎 Final Capital", f"${final_capital:,.2f}")
        table.add_row("📈 Total Profit/Loss", f"${self.tm.total_profit:,.2f}")
        table.add_row("📊 Total Return", f"{total_return:+.2f}%")

        # Trade Statistics
        table.add_row("🔢 Number of Trades", str(n_trades))
        table.add_row(
            "🎯 Win Rate", f"{win_rate:.1%}" if win_rate is not None else "N/A"
        )
        table.add_row(
            "💫 Average Profit/Trade",
            f"${avg_profit:,.2f}" if avg_profit is not None else "N/A",
        )

        # Risk Metrics
        table.add_row("📉 Max Drawdown", f"{self.tm.max_drawdown:.1f}%")
        table.add_row("📊 Volatility", f"${self.tm.volatility:.2f}")
        table.add_row("📈 Profit Factor", f"{self.tm.profit_factor:.2f}")

        # Performance Ratios
        table.add_row("📊 Sortino Ratio", f"{self.tm.calculate_sortino_ratio():.3f}")
        table.add_row("📊 Sharpe Ratio", f"{self.tm.calculate_sharpe_ratio():.3f}")
        table.add_row("📊 Market Correlation", f"{self.tm.market_correlation:.2f}")

        if console:
            console.print(table)
            return None

        return table 