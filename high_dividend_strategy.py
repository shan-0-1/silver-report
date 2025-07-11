# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import datetime
# --- 新增：导入 Plotly --- 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# --- 结束新增 ---

# --- 策略参数 ---
# 移动平均线窗口
SMA_SHORT = 12
SMA_MEDIUM = 30
SMA_LONG = 60
# RSI 参数
RSI_WINDOW = 14
RSI_OVERSOLD_THRESHOLD = 30 # 股票常用超卖阈值
# 布林带参数
BB_WINDOW = 20
BB_STD_DEV = 3
# --- 新增：威廉%R 参数 ---
WILLIAMS_R_WINDOW = 21
# --- 结束新增 ---
# 股息率参数
# MIN_ABSOLUTE_DIVIDEND_YIELD = 0.04 # 示例：最低要求 4% 的股息率 - REMOVED
# DIVIDEND_YIELD_HISTORY_WINDOW = 252 # 约等于一年 - REMOVED
# DIVIDEND_YIELD_QUANTILE = 0.6 # 要求股息率高于过去一年 60% 的时间 - REMOVED
# 信号条件
MIN_CONDITIONS_REQUIRED = 2 # 触发买入所需满足的最少核心条件数 (调整为 2，因为移除了股息条件)

# --- 数据加载 --- 
def load_stock_data(stock_files): # <-- 修改为接受文件列表
    """
    加载股票数据，只需要包含 日期, 收盘 列。
    确保文件与脚本在同一目录，或提供完整/相对路径。
    """
    all_stock_data = {}
    for file_name in stock_files:
        print(f"正在加载文件: {file_name}")
        # 尝试在脚本同目录查找
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            # --- 优先检查指定路径 ---
            os.path.join(r'C:\Users\assistant3\Downloads', file_name), # <-- 确认检查 Downloads 目录
            # --- 结束新增 ---
            os.path.join(script_dir, file_name),
            file_name # 也尝试直接使用文件名（如果在工作目录）
        ]

        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                print(f"数据文件找到: {csv_path}")
                break

        if csv_path is None:
            print(f"错误：未找到 '{file_name}'。请确保文件存在于:") # <-- 错误信息会显示尝试的文件名
            for p in possible_paths:
                print(f"- {p}")
            sys.exit()

        try:
            # --- 修改：只读取 '日期' 和 '收盘' ---
            required_cols = ['日期', '收盘']
            # 检查文件实际包含哪些列，只读取存在的
            try:
                # 快速读取第一行获取列名
                header = pd.read_csv(csv_path, nrows=0).columns.tolist()
                read_these_cols = [col for col in required_cols if col in header]
                if '收盘' not in read_these_cols:
                    print(f"错误: 文件 '{csv_path}' 中未找到必需的 '收盘' 列。")
                    sys.exit()
                if '日期' not in read_these_cols:
                    print(f"错误: 文件 '{csv_path}' 中未找到必需的 '日期' 列。")
                    sys.exit()

            except Exception as e:
                print(f"错误：读取文件列名失败: {e}")
                sys.exit()

            df = pd.read_csv(
                csv_path,
                usecols=read_these_cols, # 只读取 '日期', '收盘'
                parse_dates=['日期'],
                converters={ # 处理可能的逗号
                    '收盘': lambda x: float(str(x).replace(',', ''))
                }
            )
            # --- 修改：重命名 '收盘' 为 'Price' ---
            df = df.rename(columns={'收盘': 'Price'})

            # --- 移除对 DividendYield 的检查和处理 ---

            # 数据清洗和排序 (保持不变)
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None).dt.normalize()
            df = df.sort_values('日期', ascending=True).reset_index(drop=True)

            # 填充缺失值 (Price 保持不变)
            df['Price'] = df['Price'].ffill().bfill()
            # --- 移除 DivYield 的填充 ---

            if df.empty:
                print(f"错误：加载文件 '{file_name}' 后 DataFrame 为空。") # <-- 调整错误信息
                sys.exit()
            # --- 修改：只检查 Price 是否有 NaN ---
            if df['Price'].isnull().any():
                print(f"错误：数据加载后文件 '{file_name}' 的 Price 列仍存在 NaN 值，请检查数据源。") # <-- 调整错误信息
                print(df.isnull().sum())
                sys.exit()

            stock_name = os.path.splitext(file_name)[0].replace('历史数据', '') # 提取股票名称
            all_stock_data[stock_name] = df
            print(f"文件 '{file_name}' 数据加载完成，共 {len(df)} 条记录，从 {df['日期'].iloc[0].date()} 到 {df['日期'].iloc[-1].date()}。")

        except FileNotFoundError:
            print(f"错误: 文件 '{csv_path}' 未找到.")
            sys.exit()
        except ValueError as ve:
            print(f"错误: 转换文件 '{file_name}' 数据时出错，请检查 CSV 文件格式: {ve}")
            sys.exit()
        except KeyError as ke:
            # 这个 KeyError 现在主要捕获 usecols 或 rename 中可能的硬编码错误，理论上不应发生
            print(f"错误: 代码内部错误，指定的列名不存在: {ke}。")
            sys.exit()
        except Exception as e:
            print(f"加载或处理文件 '{file_name}' 数据时发生未知错误: {e}")
            sys.exit()

    return all_stock_data

# --- 新增：处理多股票数据函数 ---
def process_multi_stock_data(stock_dfs_dict):
    normalized_dfs = {}
    # 对每只股票进行价格归一化
    for stock_name, df in stock_dfs_dict.items():
        # 复制一份DataFrame以避免SettingWithCopyWarning，并确保排序
        df_copy = df.copy().sort_values('日期').reset_index(drop=True)
        
        # 归一化价格：相对于整个时期内的平均价格
        average_period_price = df_copy['Price'].mean()
        df_copy[f'{stock_name}_Normalized_Price'] = df_copy['Price'] / average_period_price
        normalized_dfs[stock_name] = df_copy[['日期', f'{stock_name}_Normalized_Price']]

    # 合并所有归一化的数据到同一个DataFrame
    composite_df = None
    for stock_name, df_norm in normalized_dfs.items():
        if composite_df is None:
            composite_df = df_norm
        else:
            composite_df = pd.merge(composite_df, df_norm, on='日期', how='outer')

    # 填充合并后可能产生的NaN值 (因为不同股票数据日期范围可能不同)
    composite_df = composite_df.sort_values('日期').reset_index(drop=True)
    for col in composite_df.columns:
        if 'Normalized_Price' in str(col):
            composite_df[col] = composite_df[col].ffill().bfill() # 前向填充再后向填充

    # 计算综合价格：所有归一化价格的平均值
    normalized_price_cols = [f'{s}_Normalized_Price' for s in stock_dfs_dict.keys()]
    # 确保只有在所有股票都有数据的情况下才计算平均值，否则可能引入偏差
    composite_df['Price'] = composite_df[normalized_price_cols].mean(axis=1)

    # 现在基于综合价格计算指标和信号
    print("正在计算综合策略指标...")
    composite_df_with_metrics = calculate_stock_strategy(composite_df.copy()) # 传递副本

    print("正在生成综合买入信号...")
    composite_df_final = generate_stock_signals(composite_df_with_metrics.copy()) # 传递副本

    return composite_df_final, normalized_dfs # 返回综合DF和包含个体归一化价格的字典

# --- 策略计算 ---
def calculate_stock_strategy(df):
    """计算策略所需的技术指标。"""
    df_calc = df.copy()

    # 1. 移动平均线
    df_calc[f'SMA{SMA_SHORT}'] = df_calc['Price'].rolling(window=SMA_SHORT, min_periods=1).mean()
    df_calc[f'SMA{SMA_MEDIUM}'] = df_calc['Price'].rolling(window=SMA_MEDIUM, min_periods=1).mean()
    df_calc[f'SMA{SMA_LONG}'] = df_calc['Price'].rolling(window=SMA_LONG, min_periods=1).mean()

    # 2. RSI
    delta = df_calc['Price'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=RSI_WINDOW, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=RSI_WINDOW, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan) # 避免除以零
    df_calc['RSI'] = 100 - (100 / (1 + rs))
    df_calc['RSI'] = df_calc['RSI'].fillna(50) # 中性填充

    # 3. 布林带
    df_calc['BB_Mid'] = df_calc['Price'].rolling(window=BB_WINDOW, min_periods=1).mean()
    rolling_std = df_calc['Price'].rolling(window=BB_WINDOW, min_periods=1).std().fillna(0)
    df_calc['BB_Upper'] = df_calc['BB_Mid'] + (BB_STD_DEV * rolling_std)
    df_calc['BB_Lower'] = df_calc['BB_Mid'] - (BB_STD_DEV * rolling_std)

    # --- 新增：计算威廉%R ---
    # 4. 威廉%R (Williams %R)
    hh = df_calc['Price'].rolling(window=WILLIAMS_R_WINDOW, min_periods=1).max() # Highest High in period
    ll = df_calc['Price'].rolling(window=WILLIAMS_R_WINDOW, min_periods=1).min() # Lowest Low in period
    close = df_calc['Price']
    # 计算 %R，处理分母为零的情况
    range_hh_ll = hh - ll
    # --- 修改：使用 np.where 避免 SettingWithCopyWarning --- 
    df_calc['WilliamsR'] = np.where(
        range_hh_ll == 0, 
        -50, # 或者 0, 或 np.nan - 当最高价等于最低价时给一个中间值
        (hh - close) / range_hh_ll * -100
    )
    # df_calc.loc[range_hh_ll == 0, 'WilliamsR'] = -50 # Avoid division by zero, set to mid-point
    # df_calc.loc[range_hh_ll != 0, 'WilliamsR'] = (hh - close) / range_hh_ll * -100
    df_calc['WilliamsR'] = df_calc['WilliamsR'].fillna(-50) # 填充可能因窗口期不足产生的 NaN
    # --- 结束新增 ---

    # --- 移除股息率历史分位数计算 ---
    # 4. 股息率历史分位数 - REMOVED
    # df_calc['DivYield_Hist_Quantile'] = df_calc['DivYield'].rolling(
    #     window=DIVIDEND_YIELD_HISTORY_WINDOW, min_periods=int(DIVIDEND_YIELD_HISTORY_WINDOW * 0.5)
    # ).quantile(DIVIDEND_YIELD_QUANTILE).ffill().bfill()

    # 最终填充，确保没有 NaN
    df_calc = df_calc.fillna(method='bfill').fillna(method='ffill') # 填充开头和结尾的 NaN

    return df_calc

# --- 信号生成 ---
def generate_stock_signals(df_metrics):
    """根据计算出的指标生成买入信号。"""
    df_signal = df_metrics.copy()
    df_signal['BuySignal'] = False # 初始化信号列

    # 定义核心买入条件 (只基于价格)
    # 条件1：价格低于长期均线 (估值参考)
    cond1 = df_signal['Price'] < df_signal[f'SMA{SMA_LONG}']
    # 条件2：RSI 低于超卖阈值 (动量参考)
    cond2 = df_signal['RSI'] < RSI_OVERSOLD_THRESHOLD
    # 条件3：价格接近或低于布林下轨 (波动性参考)
    cond3 = df_signal['Price'] <= df_signal['BB_Lower'] * 1.01 # 稍微放宽一点
    # --- 移除基于股息率的条件 4 和 5 ---
    # cond4 = df_signal['DivYield'] >= MIN_ABSOLUTE_DIVIDEND_YIELD
    # cond5 = df_signal['DivYield'] >= df_signal['DivYield_Hist_Quantile']

    # 记录每个条件是否满足（用于终端输出）
    df_signal['cond1_met'] = cond1
    df_signal['cond2_met'] = cond2
    df_signal['cond3_met'] = cond3
    # --- 移除条件 4 和 5 的记录 ---
    # df_signal['cond4_met'] = cond4
    # df_signal['cond5_met'] = cond5

    # --- 修改：组合条件生成信号 (满足 3 个价格条件中的至少 2 个) ---
    conditions_sum = cond1.astype(int) + cond2.astype(int) + cond3.astype(int)
    df_signal['BuySignal'] = conditions_sum >= MIN_CONDITIONS_REQUIRED # MIN_CONDITIONS_REQUIRED 已改为 2

    return df_signal

# --- 修改：创建图表函数 ---
def create_stock_visualization(composite_df, normalized_dfs):
    """使用 Plotly 生成基于价格的策略可视化图表对象，并在图3使用次要Y轴对齐RSI和W%R。
    现在显示归一化的个体股票价格，以及基于综合价格计算的SMA、布林带、RSI和威廉%R。
    """
    # 定义不同股票的颜色
    colors = ['navy', 'darkorange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_idx = 0
    
    # --- 修改：为第三行启用次要Y轴 ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        specs=[[{"secondary_y": False}], 
                               [{"secondary_y": False}], 
                               [{"secondary_y": True}]], # 指定第三行有次要Y轴
                        subplot_titles=(
                            '归一化价格与综合指标', 
                            '归一化价格与综合布林带通道', 
                            f'综合震荡指标 (左: RSI / 右: 威廉%R)' # 更新标题以区分左右轴
                        ))
    
    # --- 添加所有股票的归一化价格线条 --- 
    for stock_name, df_norm in normalized_dfs.items():
        current_color = colors[color_idx % len(colors)]
        color_idx += 1
        fig.add_trace(go.Scatter(x=df_norm['日期'], y=df_norm[f'{stock_name}_Normalized_Price'], 
                                 mode='lines', name=f'{stock_name} 归一化价格',
                                 line=dict(color=current_color, width=1.5)),
                      row=1, col=1)
        # 第二张图也显示归一化价格，但不显示图例，以避免重复
        fig.add_trace(go.Scatter(x=df_norm['日期'], y=df_norm[f'{stock_name}_Normalized_Price'], 
                                 mode='lines', name=f'{stock_name} 归一化价格', showlegend=False,
                                 line=dict(color=current_color, width=1.5)),
                      row=2, col=1)


    # --- 行 1: 综合SMA和买入信号 (基于 composite_df) ---
    # 添加综合SMA
    fig.add_trace(go.Scatter(x=composite_df['日期'], y=composite_df[f'SMA{SMA_LONG}'], mode='lines', name=f'综合SMA{SMA_LONG}',
                             line=dict(color='black', dash='dash', width=2)), # 综合SMA使用粗黑虚线
                  row=1, col=1)
    # 添加买入信号标记
    signal_df = composite_df[composite_df['BuySignal']]
    if not signal_df.empty:
        fig.add_trace(go.Scatter(x=signal_df['日期'], y=signal_df['Price'], mode='markers', name='综合买入信号',
                                 marker=dict(color='red', size=10, symbol='triangle-up', line=dict(width=1, color='black'))),
                      row=1, col=1)
    
    # --- 行 2: 综合布林带 (基于 composite_df) ---
    fig.add_trace(go.Scatter(x=composite_df['日期'], y=composite_df['BB_Upper'], mode='lines', name='综合布林上轨',
                             line=dict(color='grey', dash='dash', width=1.5)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=composite_df['日期'], y=composite_df['BB_Lower'], mode='lines', name='综合布林下轨',
                             line=dict(color='grey', dash='dash', width=1.5)),
                  row=2, col=1)
    # 填充综合布林带区域
    fig.add_trace(go.Scatter(x=composite_df['日期'], y=composite_df['BB_Upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=composite_df['日期'], y=composite_df['BB_Lower'], fill='tonexty', 
                             mode='lines', line=dict(width=0), fillcolor='rgba(150, 150, 150, 0.1)',
                             name='综合布林带范围'), row=2, col=1)

    # --- 行 3: 综合震荡指标 (RSI 在左轴, 威廉%R 在右轴，基于 composite_df) ---
    # RSI (画在主Y轴 - 左侧)
    fig.add_trace(go.Scatter(x=composite_df['日期'], y=composite_df['RSI'], mode='lines', name='综合RSI (左轴)',
                             line=dict(color='darkviolet', width=2)), # 综合RSI使用粗线
                  row=3, col=1, secondary_y=False) # 明确指定主轴
    # 添加 RSI 超卖阈值线 (关联到主轴，只添加一次)
    fig.add_hline(y=RSI_OVERSOLD_THRESHOLD, line_dash="dot", line_color="red", opacity=0.5, 
                  annotation_text=f"RSI超卖={RSI_OVERSOLD_THRESHOLD}", 
                  row=3, col=1, secondary_y=False) # 明确指定主轴

    # 威廉%R (画在次要Y轴 - 右侧)
    fig.add_trace(go.Scatter(x=composite_df['日期'], y=composite_df['WilliamsR'], mode='lines', name='综合威廉%R (右轴)',
                             line=dict(color='dodgerblue', width=2)), # 综合威廉R使用粗线
                  row=3, col=1, secondary_y=True) # 指定次要轴
    # 添加 威廉%R 超买/超卖线 (关联到次要轴，只添加一次)
    fig.add_hline(y=-20, line_dash="dash", line_color="grey", opacity=0.5, 
                  annotation_text="W%R超买=-20", 
                  row=3, col=1, secondary_y=True) # 指定次要轴
    fig.add_hline(y=-80, line_dash="dash", line_color="grey", opacity=0.5, 
                  annotation_text="W%R超卖=-80", 
                  row=3, col=1, secondary_y=True) # 指定次要轴

    # --- 更新整体布局 ---
    fig.update_layout(
        height=800,
        title_text='多股票归一化价格与综合策略指标可视化',
        hovermode='x unified',
        legend_title_text='图例说明',
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    # --- 修改：分别为图3的主次Y轴设置范围和标题 ---
    fig.update_yaxes(title_text="归一化价格", row=1, col=1)
    fig.update_yaxes(title_text="归一化价格", row=2, col=1)
    # 图3 主Y轴 (RSI)
    fig.update_yaxes(title_text="RSI (0 ~ 100)", range=[0, 100], row=3, col=1, secondary_y=False)
    # 图3 次要Y轴 (威廉%R)
    fig.update_yaxes(title_text="威廉%R (-100 ~ 0)", range=[-100, 0], row=3, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="日期", row=3, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    return fig
# --- 结束新增图表函数 ---

# --- 主程序 ---
if __name__ == "__main__":
    print("--- 股票价格择时策略分析 ---")

    # 1. 加载数据
    stock_files = ["GSBD历史数据.csv", "AGNC历史数据.csv", "GBDC历史数据.csv"] # 替换为实际的文件名
    all_stock_data = load_stock_data(stock_files)

    # 2. 处理多股票数据，计算综合指标和信号
    print("\n正在处理多股票数据，计算综合指标和信号...")
    composite_df_final, normalized_dfs = process_multi_stock_data(all_stock_data)

    # 3. 生成图表对象 (不生成 HTML)
    print("\n正在生成图表对象...")
    # 传递综合数据和归一化后的个体数据
    fig = create_stock_visualization(composite_df_final, normalized_dfs) 
    print("图表对象已生成 (变量名: fig)。")

    # 4. 终端输出结果 (现在只基于综合数据进行判断)
    print("\n--- 综合分析最新交易日数据 ---")
    if not composite_df_final.empty:
        latest_data = composite_df_final.iloc[-1]
        print(f"日期: {latest_data['日期'].strftime('%Y-%m-%d')}")
        print(f"综合归一化价格: {latest_data['Price']:.4f}")
        print("-" * 20)
        print("核心条件检查:")
        print(f"  1. 价格 < SMA{SMA_LONG} ({latest_data[f'SMA{SMA_LONG}']:.4f}): {'满足' if latest_data['cond1_met'] else '不满足'} (当前综合价: {latest_data['Price']:.4f})")
        print(f"  2. RSI < {RSI_OVERSOLD_THRESHOLD} ({latest_data['RSI']:.1f}): {'满足' if latest_data['cond2_met'] else '不满足'}")
        print(f"  3. 价格 <= BB下轨*1.01 ({latest_data['BB_Lower']*1.01:.4f}): {'满足' if latest_data['cond3_met'] else '不满足'} (当前综合价: {latest_data['Price']:.4f})")
        print("-" * 20)
        conditions_met_count = latest_data[['cond1_met', 'cond2_met', 'cond3_met']].sum()
        print(f"满足条件数: {conditions_met_count} / 3 (要求 >= {MIN_CONDITIONS_REQUIRED})")

        if latest_data['BuySignal']:
            print("\n>> 最终综合建议: 买入 <<")
        else:
            print("\n>> 最终综合建议: 持有/观望 <<")

        # 打印历史综合买入信号汇总
        buy_signals_history = composite_df_final[composite_df_final['BuySignal']]
        print(f"\n--- 历史综合买入信号 ({len(buy_signals_history)} 个) ---")
        if not buy_signals_history.empty:
            print(buy_signals_history[['日期', 'Price', 'RSI']].tail(10).to_string(index=False))
        else:
            print("历史记录中没有产生综合买入信号。")

    else:
        print("未能获取最终综合数据进行分析。")

    print("\n分析结束。")
    
    # --- 新增：如何显示或保存图表对象的注释 ---
    # 要在浏览器或 Plotly 支持的环境中显示图表，取消下面这行的注释:
    fig.show()
    
    