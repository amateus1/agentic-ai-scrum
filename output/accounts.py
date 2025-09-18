import datetime
from typing import Dict, List, Union


class Account:
    def __init__(self, user_id: str, initial_deposit: float, language: str = 'EN'):
        self.user_id = user_id
        self.balance = initial_deposit
        self.initial_deposit = initial_deposit
        self.transactions = []
        self.holdings = {}
        self.language = language
    
    def deposit(self, amount: float) -> str:
        if amount <= 0:
            return self._get_message(
                "Deposit amount must be positive.",
                "存款金额必须为正数。"
            )
        self.balance += amount
        self._add_transaction('deposit', amount)
        return self._get_message(
            "Deposit successful.",
            "存款成功。"
        )
    
    def withdraw(self, amount: float) -> str:
        if amount <= 0:
            return self._get_message(
                "Withdrawal amount must be positive.",
                "提款金额必须为正数。"
            )
        if self.balance >= amount:
            self.balance -= amount
            self._add_transaction('withdraw', -amount)
            return self._get_message(
                "Withdrawal successful.",
                "提款成功。"
            )
        else:
            return self._get_message(
                "Insufficient funds for withdrawal.",
                "提款金额不足。"
            )
    
    def buy_shares(self, symbol: str, quantity: int, price: float = None) -> str:
        if price is None:
            price = get_share_price(symbol)
            
        if quantity <= 0:
            return self._get_message(
                "Quantity must be positive.",
                "数量必须为正数。"
            )
            
        total_cost = quantity * price
        if self.balance >= total_cost:
            self.balance -= total_cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
            self._add_transaction('buy', -total_cost, symbol, quantity, price)
            return self._get_message(
                f"Purchased {quantity} shares of {symbol} at {price} per share.",
                f"以每股{price}的价格购买了{quantity}股{symbol}。"
            )
        else:
            return self._get_message(
                "Insufficient funds to buy shares.",
                "资金不足以购买股票。"
            )
    
    def sell_shares(self, symbol: str, quantity: int, price: float = None) -> str:
        if price is None:
            price = get_share_price(symbol)
            
        if quantity <= 0:
            return self._get_message(
                "Quantity must be positive.",
                "数量必须为正数。"
            )
            
        if self.holdings.get(symbol, 0) >= quantity:
            total_revenue = quantity * price
            self.balance += total_revenue
            self.holdings[symbol] -= quantity
            if self.holdings[symbol] == 0:
                del self.holdings[symbol]  # Remove the symbol if no shares left
            self._add_transaction('sell', total_revenue, symbol, quantity, price)
            return self._get_message(
                f"Sold {quantity} shares of {symbol} at {price} per share.",
                f"以每股{price}的价格售出了{quantity}股{symbol}。"
            )
        else:
            return self._get_message(
                "Not enough shares to sell.",
                "没有足够的股票出售。"
            )
    
    def get_portfolio_value(self) -> float:
        value = sum(quantity * get_share_price(symbol) for symbol, quantity in self.holdings.items())
        return value
    
    def get_profit_or_loss(self) -> float:
        portfolio_value = self.get_portfolio_value()
        current_balance = self.balance
        total_value = current_balance + portfolio_value
        return total_value - self.initial_deposit
    
    def report_holdings(self) -> str:
        if not self.holdings:
            return self._get_message(
                "You don't have any holdings.",
                "您没有任何持股。"
            )
            
        holdings_str_en = "Current Holdings:\n"
        holdings_str_cn = "当前持股：\n"
        
        for symbol, quantity in self.holdings.items():
            current_price = get_share_price(symbol)
            value = quantity * current_price
            holdings_str_en += f"- {symbol}: {quantity} shares at {current_price} per share = {value}\n"
            holdings_str_cn += f"- {symbol}: {quantity}股，每股{current_price} = {value}\n"
            
        return self._get_message(holdings_str_en, holdings_str_cn)
    
    def report_profit_and_loss(self) -> str:
        profit_loss = self.get_profit_or_loss()
        portfolio_value = self.get_portfolio_value()
        
        if profit_loss > 0:
            message_en = f"Profit: ${profit_loss:.2f}"
            message_cn = f"盈利：${profit_loss:.2f}"
        elif profit_loss < 0:
            message_en = f"Loss: ${-profit_loss:.2f}"
            message_cn = f"亏损：${-profit_loss:.2f}"
        else:
            message_en = "Break even: $0.00"
            message_cn = "收支平衡：$0.00"
            
        details_en = f"Initial deposit: ${self.initial_deposit:.2f}\nCurrent balance: ${self.balance:.2f}\nPortfolio value: ${portfolio_value:.2f}\nTotal value: ${self.balance + portfolio_value:.2f}"
        details_cn = f"初始存款：${self.initial_deposit:.2f}\n当前余额：${self.balance:.2f}\n投资组合价值：${portfolio_value:.2f}\n总价值：${self.balance + portfolio_value:.2f}"
        
        return self._get_message(
            f"{message_en}\n{details_en}",
            f"{message_cn}\n{details_cn}"
        )
    
    def list_transactions(self) -> str:
        if not self.transactions:
            return self._get_message(
                "No transactions found.",
                "未找到交易记录。"
            )
            
        transactions_en = "Transaction History:\n"
        transactions_cn = "交易历史：\n"
        
        for i, tx in enumerate(self.transactions, 1):
            date_str = tx['date'].strftime("%Y-%m-%d %H:%M:%S")
            
            if tx['type'] == 'deposit':
                tx_en = f"{i}. {date_str} - Deposit: +${tx['amount']:.2f}"
                tx_cn = f"{i}. {date_str} - 存款: +${tx['amount']:.2f}"
            elif tx['type'] == 'withdraw':
                tx_en = f"{i}. {date_str} - Withdrawal: ${tx['amount']:.2f}"
                tx_cn = f"{i}. {date_str} - 提款: ${tx['amount']:.2f}"
            elif tx['type'] == 'buy':
                tx_en = f"{i}. {date_str} - Bought {tx['quantity']} {tx['symbol']} at ${tx['price']:.2f}: ${tx['amount']:.2f}"
                tx_cn = f"{i}. {date_str} - 购买 {tx['quantity']}股 {tx['symbol']}，价格 ${tx['price']:.2f}: ${tx['amount']:.2f}"
            elif tx['type'] == 'sell':
                tx_en = f"{i}. {date_str} - Sold {tx['quantity']} {tx['symbol']} at ${tx['price']:.2f}: +${tx['amount']:.2f}"
                tx_cn = f"{i}. {date_str} - 卖出 {tx['quantity']}股 {tx['symbol']}，价格 ${tx['price']:.2f}: +${tx['amount']:.2f}"
            
            transactions_en += f"{tx_en}\n"
            transactions_cn += f"{tx_cn}\n"
        
        return self._get_message(transactions_en, transactions_cn)
    
    def get_current_stock_price(self, symbol: str) -> str:
        price = get_share_price(symbol)
        return self._get_message(
            f"Current price of {symbol}: ${price:.2f}",
            f"{symbol}当前价格：${price:.2f}"
        )
    
    def get_account_summary(self) -> str:
        portfolio_value = self.get_portfolio_value()
        profit_loss = self.get_profit_or_loss()
        total_value = self.balance + portfolio_value
        
        summary_en = f"Account Summary for User {self.user_id}:\n"
        summary_en += f"Cash Balance: ${self.balance:.2f}\n"
        summary_en += f"Portfolio Value: ${portfolio_value:.2f}\n"
        summary_en += f"Total Value: ${total_value:.2f}\n"
        summary_en += f"Profit/Loss: ${profit_loss:.2f}"
        
        summary_cn = f"用户{self.user_id}的账户摘要：\n"
        summary_cn += f"现金余额：${self.balance:.2f}\n"
        summary_cn += f"投资组合价值：${portfolio_value:.2f}\n"
        summary_cn += f"总价值：${total_value:.2f}\n"
        summary_cn += f"盈亏：${profit_loss:.2f}"
        
        return self._get_message(summary_en, summary_cn)
    
    def _add_transaction(self, transaction_type: str, amount: float, symbol: str = '', quantity: int = 0, price: float = 0.0):
        transaction = {
            'type': transaction_type,
            'amount': amount,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'date': datetime.datetime.now()
        }
        self.transactions.append(transaction)
    
    def _get_message(self, en_message: str, cn_message: str) -> str:
        if self.language == 'EN' or self.language == 'CN':
            return f"{en_message}\n{cn_message}"
        else:
            return en_message


def get_share_price(symbol: str) -> float:
    """Get the current price of a share. This is a test implementation."""
    prices = {
        'AAPL': 150.0,
        'TSLA': 750.0,
        'GOOGL': 2800.0
    }
    return prices.get(symbol, 0.0)