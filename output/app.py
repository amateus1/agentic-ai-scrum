import gradio as gr
from accounts import Account, get_share_price
import datetime

# Initialize the account
account = Account("user001", 10000.0, "EN")

def update_stock_price_display():
    """Get current stock prices for display"""
    symbols = ['AAPL', 'TSLA', 'GOOGL']
    price_text = ""
    for symbol in symbols:
        price = get_share_price(symbol)
        price_text += f"{symbol}: ${price:.2f} / {symbol}: ${price:.2f}\n"
    return price_text

def create_account_ui(initial_deposit):
    global account
    account = Account("user001", float(initial_deposit), "EN")
    return account.get_account_summary()

def deposit_funds_ui(amount):
    result = account.deposit(float(amount))
    return result, account.get_account_summary()

def withdraw_funds_ui(amount):
    result = account.withdraw(float(amount))
    return result, account.get_account_summary()

def buy_shares_ui(symbol, quantity):
    result = account.buy_shares(symbol, int(quantity))
    return result, account.get_account_summary()

def sell_shares_ui(symbol, quantity):
    result = account.sell_shares(symbol, int(quantity))
    return result, account.get_account_summary()

def get_holdings_ui():
    return account.report_holdings()

def get_profit_loss_ui():
    return account.report_profit_and_loss()

def get_transactions_ui():
    return account.list_transactions()

def get_account_summary_ui():
    return account.get_account_summary()

def get_stock_price_ui(symbol):
    return account.get_current_stock_price(symbol)

# CSS styling
css = """
body { background-color: #f8f9fa; }
.gr-block { margin-bottom: 15px; }
h1, h2, h3 { color: #1e88e5; }
.gr-button { background-color: #4caf50 !important; color: white !important; }
.gr-button:hover { background-color: #45a049 !important; }
"""

with gr.Blocks(css=css, title="Trading Simulation Platform / 交易模拟平台") as demo:
    gr.Markdown("# Trading Simulation Platform / 交易模拟平台")
    
    # Stock price display at the top
    stock_prices = update_stock_price_display()
    gr.Markdown(f"## Current Stock Prices / 当前股票价格\n{stock_prices}")
    
    with gr.Tabs():
        with gr.TabItem("Create Account / 创建账户"):
            gr.Markdown("## Create New Account / 创建新账户")
            initial_deposit = gr.Number(label="Initial Deposit Amount / 初始存款金额", value=10000.0)
            create_btn = gr.Button("Create Account / 创建账户")
            account_output = gr.Textbox(label="Account Status / 账户状态", interactive=False)
            create_btn.click(create_account_ui, inputs=initial_deposit, outputs=account_output)
        
        with gr.TabItem("Deposit/Withdraw / 存款/取款"):
            gr.Markdown("## Deposit or Withdraw Funds / 存款或取款")
            deposit_amount = gr.Number(label="Deposit Amount / 存款金额")
            deposit_btn = gr.Button("Deposit / 存款")
            withdraw_amount = gr.Number(label="Withdraw Amount / 取款金额")
            withdraw_btn = gr.Button("Withdraw / 取款")
            result_output = gr.Textbox(label="Transaction Result / 交易结果", interactive=False)
            summary_output = gr.Textbox(label="Account Summary / 账户摘要", interactive=False)
            
            deposit_btn.click(deposit_funds_ui, inputs=deposit_amount, outputs=[result_output, summary_output])
            withdraw_btn.click(withdraw_funds_ui, inputs=withdraw_amount, outputs=[result_output, summary_output])
        
        with gr.TabItem("Buy/Sell Shares / 买入/卖出股票"):
            gr.Markdown("## Buy or Sell Shares / 买入或卖出股票")
            symbol = gr.Dropdown(choices=["AAPL", "TSLA", "GOOGL"], label="Stock Symbol / 股票代码")
            quantity = gr.Number(label="Quantity / 数量", value=1)
            buy_btn = gr.Button("Buy Shares / 买入股票")
            sell_btn = gr.Button("Sell Shares / 卖出股票")
            trade_result = gr.Textbox(label="Trade Result / 交易结果", interactive=False)
            trade_summary = gr.Textbox(label="Account Summary / 账户摘要", interactive=False)
            
            buy_btn.click(buy_shares_ui, inputs=[symbol, quantity], outputs=[trade_result, trade_summary])
            sell_btn.click(sell_shares_ui, inputs=[symbol, quantity], outputs=[trade_result, trade_summary])
        
        with gr.TabItem("Portfolio / 投资组合"):
            gr.Markdown("## Portfolio Management / 投资组合管理")
            holdings_btn = gr.Button("View Holdings / 查看持股")
            profit_loss_btn = gr.Button("View Profit/Loss / 查看盈亏")
            transactions_btn = gr.Button("View Transactions / 查看交易记录")
            summary_btn = gr.Button("View Account Summary / 查看账户摘要")
            portfolio_output = gr.Textbox(label="Portfolio Information / 投资组合信息", interactive=False)
            
            holdings_btn.click(get_holdings_ui, inputs=None, outputs=portfolio_output)
            profit_loss_btn.click(get_profit_loss_ui, inputs=None, outputs=portfolio_output)
            transactions_btn.click(get_transactions_ui, inputs=None, outputs=portfolio_output)
            summary_btn.click(get_account_summary_ui, inputs=None, outputs=portfolio_output)
        
        with gr.TabItem("Stock Price / 股票价格"):
            gr.Markdown("## Check Stock Price / 查询股票价格")
            price_symbol = gr.Dropdown(choices=["AAPL", "TSLA", "GOOGL"], label="Stock Symbol / 股票代码")
            price_btn = gr.Button("Get Price / 获取价格")
            price_output = gr.Textbox(label="Current Price / 当前价格", interactive=False)
            price_btn.click(get_stock_price_ui, inputs=price_symbol, outputs=price_output)

if __name__ == "__main__":
    demo.launch()