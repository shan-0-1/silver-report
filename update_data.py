# -*- coding: utf-8 -*-
import investpy
import pandas as pd
import os
from datetime import datetime, timedelta
import time

# --- 配置 ---
DOWNLOAD_DIR = r'C:\Users\assistant3\Downloads' # 数据保存目录 - 使用原始字符串或双反斜杠
STOCK_SYMBOL = 'GSBD'          # GSBD 在 Investing.com 的股票代码 (通常是 Ticker)
STOCK_COUNTRY = 'united states' # 股票所在的国家/市场
STOCK_FILENAME = 'GSBD历史数据.csv'

COMMODITY_SYMBOL = 'XAG/CNY' # 白银/人民币在 Investing.com 的代码 (需要确认是否准确)
# 注意: XAG/CNY 可能需要作为货币对(currency_cross)获取，或者作为商品(commodity)获取(如 XAG/USD 再转换)
# investpy 可能不直接支持 XAG/CNY，需要测试。如果不行，可能需要获取 XAG/USD 和 USD/CNY 再计算。
COMMODITY_FILENAME = 'XAG_CNY历史数据.csv'

# 获取多长时间的数据（例如，最近5年）
YEARS_OF_DATA = 5
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=YEARS_OF_DATA * 365)
# investpy 需要 'dd/mm/yyyy' 格式
DATE_FORMAT_INVESTPY = '%d/%m/%Y'
START_DATE_STR = START_DATE.strftime(DATE_FORMAT_INVESTPY)
END_DATE_STR = END_DATE.strftime(DATE_FORMAT_INVESTPY)

# --- 确保下载目录存在 ---
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- 函数：获取并保存数据 ---
def fetch_and_save_data(symbol, filename, data_type, country=None):
    """
    从 Investing.com 获取数据并保存为 CSV，失败时自动重试。

    Args:
        symbol (str): Investing.com 上的代码。
        filename (str): 要保存的 CSV 文件名。
        data_type (str): 'stock', 'commodity', 或 'currency_cross'。
        country (str, optional): 股票或某些商品需要的国家。
    """
    full_path = os.path.join(DOWNLOAD_DIR, filename)
    print(f"--- 开始更新: {symbol} ({data_type}) ---")
    print(f"目标文件: {full_path}")

    retry_delay = 0 # 失败后重试的延迟时间（秒）

    while True: # 无限循环直到成功
        try:
            print(f"尝试获取 {symbol} 数据... 时间范围: {START_DATE_STR} 到 {END_DATE_STR}")
            df = None
            if data_type == 'stock':
                if not country:
                    print("错误: 获取股票数据需要指定国家 (country)。")
                    return # 配置错误，不重试
                df = investpy.get_stock_historical_data(
                    stock=symbol,
                    country=country,
                    from_date=START_DATE_STR,
                    to_date=END_DATE_STR
                )
            elif data_type == 'commodity':
                 print(f"注意: 尝试获取商品 '{symbol}' 并计算。")
                 # 获取 XAG/USD
                 print("获取 XAG/USD 数据...")
                 df_xagusd = investpy.get_commodity_historical_data(
                     commodity='Silver',
                     from_date=START_DATE_STR,
                     to_date=END_DATE_STR
                 )
                 # 获取 USD/CNY
                 print("获取 USD/CNY 数据...")
                 df_usdcny = investpy.get_currency_cross_historical_data(
                     currency_cross='USD/CNY',
                     from_date=START_DATE_STR,
                     to_date=END_DATE_STR
                 )
                 # 合并计算 XAG/CNY (价格 = XAG/USD * USD/CNY)
                 df = pd.merge(df_xagusd[['Close']], df_usdcny[['Close']], left_index=True, right_index=True, how='inner', suffixes=('_xagusd', '_usdcny'))
                 df['Close'] = df['Close_xagusd'] * df['Close_usdcny']
                 df = df[['Close']] # 只保留最终计算的收盘价

            elif data_type == 'currency_cross':
                print(f"尝试直接获取货币对: {symbol}")
                df = investpy.get_currency_cross_historical_data(
                    currency_cross=symbol,
                    from_date=START_DATE_STR,
                    to_date=END_DATE_STR
                )
            else:
                print(f"错误: 不支持的数据类型 '{data_type}'。")
                return # 配置错误，不重试

            if df is not None and not df.empty:
                print(f"成功获取 {len(df)} 条 '{symbol}' 相关数据。")
                df = df.reset_index()
                df = df[['Date', 'Close']]
                df.rename(columns={'Date': '日期', 'Close': '收盘'}, inplace=True)
                df['日期'] = pd.to_datetime(df['日期']).dt.normalize()
                df.to_csv(full_path, index=False, encoding='utf-8-sig')
                print(f"数据已成功保存到: {full_path}")
                break # <--- 成功！跳出 while 循环
            elif df is not None and df.empty:
                print(f"警告: 获取到 '{symbol}' 的数据为空。视为成功，不再重试。")
                break # <--- 获取到空数据，也跳出循环
            else:
                 print(f"未能获取 '{symbol}' 的数据框架 (investpy 返回 None?)。视为成功，不再重试。")
                 break # <--- 未知原因未能获取 df，跳出循环

        except Exception as e:
            print(f"错误: 更新 '{symbol}' 时发生错误: {e}")
            print(f"将在 {retry_delay} 秒后重试...")
            time.sleep(retry_delay) # 等待后继续下一次循环
            # 不使用 break，循环将继续

# --- 主程序 ---
if __name__ == "__main__":
    print("开始执行数据自动更新脚本...")

    # 1. 更新股票数据 (GSBD) - 函数内部会无限重试直到成功
    fetch_and_save_data(STOCK_SYMBOL, STOCK_FILENAME, 'stock', country=STOCK_COUNTRY)

    print("-" * 30)

    # 2. 更新商品/货币对数据 (XAG/CNY) - 函数内部会无限重试直到成功
    fetch_and_save_data('XAG/CNY', COMMODITY_FILENAME, 'commodity')

    print("\n数据更新脚本执行完毕。") # 只有在两个 fetch 都成功后才会执行到这里 