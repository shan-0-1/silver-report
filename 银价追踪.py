# -*- coding: utf-8 -*-  # 必须放在文件第一行
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio # 引入 plotly.io 用于 HTML 导出
import subprocess # 用于执行 Git 命令
import datetime # 用于生成提交信息时间戳

# --- 保留用于查找数据文件的打包相关代码 ---
# (虽然我们不再打包成 EXE, 但保留此逻辑无害，且万一以后需要此脚本在打包环境运行其他任务时有用)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 获取脚本所在目录

# 策略参数优化
BASE_WINDOW_SHORT = 30
BASE_WINDOW_LONG = 90
MIN_WINDOW_SHORT = 5
WINDOW_DECAY_RATE = 0.9
MIN_PURCHASE_INTERVAL = 1

# 在参数定义区域新增窗口设定
HISTORY_WINDOW_SHORT = 24  # 新增短期窗口
HISTORY_WINDOW = HISTORY_WINDOW_SHORT * 2        # 原中型窗口保留
HISTORY_WINDOW_LONG = HISTORY_WINDOW * 2   # 新增长期窗口


def load_silver_data():
    """改进版数据加载（查找 CSV 文件）"""
    # --- 修改数据文件路径逻辑 ---
    # 优先尝试从脚本同目录或 _MEIPASS (打包环境) 加载
    possible_paths = [
        os.path.join(BASE_DIR, 'XAG_USD历史数据.csv'), # 打包环境或脚本同目录
        r'C:\Users\assistant3\Downloads\XAG_USD历史数据.csv' # 绝对路径作为后备
    ]

    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"成功找到数据文件: {csv_path}")
            break

    if csv_path is None:
         print(f"错误：在以下位置均未找到 'XAG_USD历史数据.csv':")
         for p in possible_paths:
             print(f"- {p}")
         exit()
    # --- 结束修改 ---

    try:
        df = pd.read_csv(
            csv_path, # 使用找到的路径
            usecols=['日期', '收盘'],
            parse_dates=['日期'],
            converters={
                '收盘': lambda x: float(str(x).replace(',', ''))
            }
        ).rename(columns={'收盘': 'Price'}).dropna(subset=['Price'])

        # 确保日期列为tz-naive并排序
        df['日期'] = df['日期'].dt.tz_localize(None).dt.normalize()
        df = df.sort_values('日期', ascending=True).reset_index(drop=True)
        df['Price'] = df['Price'].ffill().bfill()
        df['交易日'] = df.index + 1
        return df
    except Exception as e:
        print(f"数据加载或处理失败：{str(e)}")
        exit()


WINDOW_WEIGHT_FACTOR = 0.8  # 窗口参数在决策中的权重占比
WINDOW_CHANGE_THRESHOLD = 0.2  # 窗口变化显著阈值


def calculate_strategy(df):
    """优化后的策略计算核心"""
    # 计算自上次采购以来的天数
    df['signal_flag'] = df['采购信号'].astype(int)
    df['group'] = df['signal_flag'].cumsum()
    df['days_since_last'] = df.groupby('group').cumcount()

    # 计算调整周期数：超过7天后每天调整一次
    df['adjustment_cycles'] = np.where(
        df['days_since_last'] >= 7,
        df['days_since_last'] - 6,
        0
    )

    # 动态窗口计算 (使用 lower)
    df['动态短窗口'] = (BASE_WINDOW_SHORT *
                        (WINDOW_DECAY_RATE ** df['adjustment_cycles'])
                        ).astype(int).clip(lower=MIN_WINDOW_SHORT) # 使用 lower

    # 确保 df['动态短窗口'] 是 Series 或类似结构以用于比较
    dynamic_short_window = df['动态短窗口']
    if not hasattr(dynamic_short_window, '__iter__'):
        dynamic_short_window = pd.Series([dynamic_short_window] * len(df))

    df['动态长窗口'] = (BASE_WINDOW_LONG *
                        (WINDOW_DECAY_RATE ** df['adjustment_cycles'])
                        ).astype(int).clip(lower=dynamic_short_window * 2) # 使用 lower

    # 滚动均线计算（缓存优化）
    # --- 确保 df['动态短窗口'] 内是有效整数 ---
    # 转换前处理可能存在的非数值类型
    df['动态短窗口'] = pd.to_numeric(df['动态短窗口'], errors='coerce').fillna(MIN_WINDOW_SHORT).astype(int)
    df['动态短窗口'] = df['动态短窗口'].clip(lower=MIN_WINDOW_SHORT) # 再次确保最低值

    df['SMA动态短'] = [
        df['Price'].iloc[max(0, i - int(w) + 1):i + 1].mean() # 确保 w 是整数
        for i, w in enumerate(df['动态短窗口'])
    ]
    # --- 确保 df['动态长窗口'] 内是有效整数 ---
    df['动态长窗口'] = pd.to_numeric(df['动态长窗口'], errors='coerce').fillna(MIN_WINDOW_SHORT * 2).astype(int)
    # 动态长窗口也要确保最低值
    min_long_window = df['动态短窗口'] * 2 # 这是一个 Series
    # --- 修正：使用 np.maximum 进行元素级比较 --- 
    df['动态长窗口'] = np.maximum(df['动态长窗口'], min_long_window)
    # --- 结束修正 --- 

    df['SMA动态长'] = df['Price'].rolling(
        # window 参数需要单个整数，取该列最大值作为近似
        # 注意：这可能不是预期行为，如果希望逐行使用动态长窗口，需要类似上面的列表推导
        window=int(df['动态长窗口'].max()), # 确保 window 是整数
        min_periods=1
    ).mean().values

    # 工业指标计算（矢量运算）
    df['动量因子'] = 0.0  # 初始化动量因子列
    for i in range(len(df)):
        window = int(df.loc[i, '动态短窗口']) # 确保 window 是整数
        start_idx = max(0, i - window + 1)
        if start_idx < i + 1: # 确保窗口至少有2个点
            # --- 添加缩进 ---
            pct_changes = df['Price'].iloc[start_idx:i + 1].pct_change().abs()
            df.loc[i, '动量因子'] = pct_changes.mean()
        else:
            # --- 添加缩进 ---
            df.loc[i, '动量因子'] = 0 # 或者 np.nan

    df['动量因子'] = df['动量因子'].fillna(0) # 填充可能的 NaN

    # --- 确保分母不为零 ---
    # 确保 Series 是 float 类型再替换和计算
    sma_short_safe = pd.to_numeric(df['SMA动态短'], errors='coerce').replace(0, np.nan)
    sma_long_safe = pd.to_numeric(df['SMA动态长'], errors='coerce').replace(0, np.nan)

    df['工业指标'] = (df['Price'] / sma_short_safe) * \
                     (df['Price'] / sma_long_safe) * \
                     (1 - df['动量因子'])
    df['工业指标'] = df['工业指标'].fillna(1.0) # 用一个中性值填充可能的 NaN
    # --- 结束分母检查 ---


    # 动态阈值计算（滚动分位数）
    df['基线阈值_短'] = df['工业指标'].rolling(
        HISTORY_WINDOW_SHORT,
        min_periods=2
    ).quantile(0.25).ffill().clip(0.3, 2.0)

    df['基线阈值'] = df['工业指标'].rolling(
        HISTORY_WINDOW,
        min_periods=2
    ).quantile(0.25).ffill().clip(0.3, 2.0)

    df['基线阈值_长'] = df['工业指标'].rolling(
        HISTORY_WINDOW_LONG,
        min_periods=2
    ).quantile(0.25).ffill().clip(0.3, 2.0)

    # 新增波动率通道指标
    df['ATR'] = df['Price'].rolling(14).apply(lambda x: np.max(x) - np.min(x) if len(x)>1 else 0).shift(1).fillna(0)
    df['波动上轨'] = df['SMA动态短'] + 1.5 * df['ATR']
    df['波动下轨'] = df['SMA动态短'] - 0.7 * df['ATR']

    # 引入复合动量指标（结合RSI与MACD）
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    # --- 避免除以零 ---
    rs = gain / loss.replace(0, np.nan) # 替换 0 为 NaN
    df['修正RSI'] = 100 - (100 / (1 + rs))
    df['修正RSI'] = df['修正RSI'].fillna(50) # 用中性值 50 填充 NaN
    # --- 结束除零 ---

    # 计算三重EMA动量系统
    df['EMA9'] = df['Price'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Price'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['Price'].ewm(span=50, adjust=False).mean()

    df['ema_ratio'] = df['EMA9'] / df['EMA21'].replace(0, np.nan) # 避免除以零
    df['ema_ratio'] = df['ema_ratio'].fillna(1.0) # 中性填充

    # 修改EMA金叉条件计算公式
    df['dynamic_ema_threshold'] = 1 + (0.5 * df['动量因子'])  # 使阈值与波动率正相关
    df['EMA金叉'] = df['ema_ratio'] > df['dynamic_ema_threshold']
    # 增加EMA合理性检查
    df['EMA金叉'] = df['EMA金叉'] & (df['dynamic_ema_threshold'] < 1.5)

    # ==== 新增波动因子动态调整 ====
    df['低波动'] = df['动量因子'] < df['动量因子'].rolling(60).quantile(0.25)
    # 构建自适应布林通道
    rolling_std = df['Price'].rolling(20).std().fillna(0)
    df['布林中轨'] = df['Price'].rolling(20).mean().fillna(df['Price']) # 填充早期 NaN
    df['布林带宽'] = rolling_std / df['布林中轨'].replace(0, np.nan) # 避免除以零
    df['布林带宽'] = df['布林带宽'].fillna(0) # 填充早期 NaN

    df['布林上轨'] = df['布林中轨'] + (2 * rolling_std * (1 + df['动量因子']))
    df['布林下轨'] = df['布林中轨'] - (2 * rolling_std * (1 - df['动量因子']))

    # 动态窗口更新 (使用 lower)
    # 先计算 np.where 的结果
    short_window_values = np.where(
        df['布林带宽'] > 0.2,
        (df['动态短窗口'] * 0.8).astype(int),
        df['动态短窗口']
    )
    # 然后对结果应用 clip
    # 需要确保 short_window_values 是 Pandas Series 才能调用 .clip
    if not isinstance(short_window_values, pd.Series):
         short_window_values = pd.Series(short_window_values, index=df.index)

    df['动态短窗口'] = short_window_values.clip(lower=MIN_WINDOW_SHORT) # 使用 lower

    # RSI阈值动态计算
    df['RSI阈值'] = df['修正RSI'].rolling(63).quantile(0.3).shift(1).ffill().fillna(30) # 前向填充并设置默认值

    df['趋势相位'] = np.arctan2(df['SMA动态短'].diff(3), df['Price'].diff(3))

    # 改进EMA系统
    df['EMA梯度'] = df['EMA21'] - df['EMA50']
    df['EMA趋势'] = np.where(
        (df['EMA9'] > df['EMA21']) & (df['EMA梯度'] > 0),
        1,
        np.where(
            (df['EMA9'] < df['EMA21']) & (df['EMA梯度'] < 0),
            -1,
            0
        )
    )

    df['窗口变化率'] = (BASE_WINDOW_SHORT - df['动态短窗口']) / BASE_WINDOW_SHORT
    df['窗口状态'] = np.select(
        [
            df['窗口变化率'] > WINDOW_CHANGE_THRESHOLD,
            df['窗口变化率'] < -WINDOW_CHANGE_THRESHOLD
        ],
        [
            2,  # 窗口显著收缩
            0.5,  # 窗口显著扩张
        ],
        default=1.0  # 正常波动
    )

    df['低波动阈值'] = df['动量因子'].rolling(45).quantile(0.35).ffill().fillna(0.01) # 前向填充并设置默认值
    df['低波动'] = df['动量因子'] < df['低波动阈值']
    # 新增窗口动量指标
    df['窗口动量'] = df['动态短窗口'].rolling(5).apply(
        lambda x: (x.iloc[0] - x.iloc[-1]) / x.iloc[-1] if len(x) == 5 and x.iloc[-1] != 0 else 0, raw=False # 改进 lambda
    ).fillna(0)
    condition_cols = [
        'core_cond1_met', 'core_cond2_met', 'core_cond3_met',
        'core_cond4_met', 'core_cond5_met', 'core_cond6_met'
    ]
    for col in condition_cols:
        if col not in df.columns:
            df[col] = False

    # --- 确保关键列是数值类型 ---
    numeric_cols = ['工业指标', '基线阈值', '修正RSI', 'Price', 'EMA21', '布林下轨',
                    'ema_ratio', 'dynamic_ema_threshold', '动量因子', '低波动阈值',
                    '动态短窗口', '动态长窗口', 'SMA动态短', 'SMA动态长'] # 添加窗口和SMA
    for col in numeric_cols:
        # 检查列是否存在
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # 转换失败则为 NaN
        else:
            print(f"警告：列 '{col}' 不存在于 DataFrame 中，跳过数值转换。")

    # 填充计算中可能产生的 NaN
    df.fillna(method='ffill', inplace=True) # 可以选择更合适的填充策略
    df.fillna(method='bfill', inplace=True)
    # 提供更具体的填充值
    df.fillna({'修正RSI': 50, '动量因子': 0, 'ATR': 0, '波动上轨': df['Price'], '波动下轨': df['Price']}, inplace=True)

    return df


def generate_signals(df):
    """整合所有条件到核心条件"""
    df = df.assign(采购信号=False) if '采购信号' not in df.columns else df

    # --- 在进行比较前处理可能的 NaN ---
    # (calculate_strategy 中已添加填充逻辑，这里作为双重保障)
    df.fillna({
        '工业指标': 1.0, '基线阈值': 1.0, '修正RSI': 50, 'Price': df['Price'].median(),
        'EMA21': df['Price'].median(), '布林下轨': df['Price'].median() * 0.9,
        'ema_ratio': 1.0, 'dynamic_ema_threshold': 1.0, '动量因子': 0.01, '低波动阈值': 0.01
    }, inplace=True)
    # --- 结束 NaN 处理 ---

    # 合并后的核心条件(原基础+增强)
    try:
        # --- 添加缩进 ---
        core_conditions = [
            df['工业指标'] < df['基线阈值'],            # 原核心条件1
            df['修正RSI'] < 45,                       # 原核心条件2
            df['Price'] < df['EMA21'],                # 原核心条件3
            df['Price'] < df['布林下轨'] * 1.05,       # 原核心条件4
            df['ema_ratio'] > df['dynamic_ema_threshold'],  # 原增强条件1
            df['动量因子'] < df['低波动阈值']            # 原增强条件2
        ]

        # 确保所有条件都是布尔系列
        # --- 添加缩进 ---
        for i, cond in enumerate(core_conditions):
            if not pd.api.types.is_bool_dtype(cond):
                 # 尝试转换，如果失败则设为 False
                 core_conditions[i] = pd.to_numeric(cond, errors='coerce').fillna(0).astype(bool)

        # --- 添加缩进 ---
        df['base_pass'] = np.sum(core_conditions, axis=0) >= 4
        # 确保 peak_filter 返回布尔系列
        peak_filter_result = peak_filter(df)
        if not pd.api.types.is_bool_dtype(peak_filter_result):
             peak_filter_result = pd.to_numeric(peak_filter_result, errors='coerce').fillna(1).astype(bool) # 假设过滤失败=True

        # --- 添加缩进 ---
        new_signals = df['base_pass'] & peak_filter_result

    except Exception as e:
        # --- 添加缩进 ---
        print(f"生成信号时出错: {e}")
        # 在出错时，默认不产生任何新信号
        new_signals = pd.Series([False] * len(df))
        df['base_pass'] = pd.Series([False] * len(df))
        core_conditions = [pd.Series([False] * len(df))] * 6 # 初始化为全 False


    # 记录所有条件状态 (确保条件是 Series)
    for i in range(6):
        col_name = f'core_cond{i+1}_met'
        if i < len(core_conditions) and isinstance(core_conditions[i], pd.Series):
            df[col_name] = core_conditions[i]
        else:
            df[col_name] = False # 如果条件生成失败，默认为 False

    return process_signals(df.assign(采购信号=new_signals))


def peak_filter(df):
    """过滤价格形态 (添加空值处理)"""
    price_diff = df['Price'].diff(3)
    # 使用 fillna 避免 NaN 参与比较导致错误
    price_diff_shifted_filled = price_diff.shift(1).fillna(0)
    # 确保均值计算在非空 Series 上进行
    price_diff_mean_filled = price_diff.dropna().mean() if not price_diff.dropna().empty else 0
    price_diff_filled = price_diff.fillna(0)

    peak_condition = (price_diff_shifted_filled > price_diff_mean_filled) & (price_diff_filled < 0)

    # 计算 ATR 比率前检查分母是否为零或 NaN
    atr_denominator = (df['波动上轨'] - df['波动下轨']).replace(0, np.nan)
    atr_ratio = (df['Price'] - df['波动下轨']) / atr_denominator
    atr_ratio_filled = atr_ratio.fillna(0.5) # 用中性值填充无法计算的比率

    overbought_atr = atr_ratio_filled > 0.8

    # 确保返回布尔类型 Series
    return ~(peak_condition | overbought_atr).astype(bool)

def process_signals(df):

    processed_df = df.copy()

    # 确保信号列是布尔类型
    if '采购信号' not in processed_df.columns:
        processed_df['采购信号'] = False
    processed_df['采购信号'] = processed_df['采购信号'].astype(bool)

    # 显式类型转换（解决FutureWarning）
    # 确保 rolling 操作前 Series 是布尔型
    signal_shifted = processed_df['采购信号'].shift(1).fillna(False).astype(bool)
    shifted = signal_shifted.rolling(
        MIN_PURCHASE_INTERVAL, min_periods=1
    ).max().astype(bool)
    processed_df['采购信号'] = processed_df['采购信号'] & ~shifted

    # 限制最大连续信号
    signal_int = processed_df['采购信号'].astype(int)
    # groupby 的 key 需要能 hash，使用 cumsum 的结果是 OK 的
    group_keys = (~processed_df['采购信号']).cumsum()
    signal_streak = signal_int.groupby(group_keys).transform('cumsum') # Use cumsum for streak count

    processed_df['采购信号'] = processed_df['采购信号'] & (signal_streak <= MIN_PURCHASE_INTERVAL)

    processed_df.loc[processed_df['采购信号'], 'adjustment_cycles'] = 0
    # 在process_signals中添加
    processed_df['动态短窗口'] = np.where(
        processed_df['采购信号'],
        BASE_WINDOW_SHORT,
        processed_df['动态短窗口']
    )

    # 放宽连续信号限制 (使用 transform('sum') 可能更符合原意，如果 streak 是指组内总数)
    # 如果 streak 是指连续计数，用 transform('cumsum')
    # 假设原意是组内总数限制
    signal_streak_total = signal_int.groupby(group_keys).transform('sum')
    processed_df['采购信号'] = processed_df['采购信号'] & (signal_streak_total <= MIN_PURCHASE_INTERVAL * 1.5)


    return processed_df


def generate_report(df):
    """
    生成包含详细解释和悬停提示的 HTML 格式分析报告。
    此报告旨在帮助用户（即使不熟悉金融交易）理解当前的白银市场状况以及策略的买入建议。
    优化：移除了文本中可见的(?)标记，悬停提示功能保留。
    新增：为带有悬停提示的元素添加 CSS 样式（始终显示虚线下划线，悬停时变色）。
    """
    if df.empty:
        return "<h2>⚠️ 数据为空，无法生成报告</h2>"

    # 确保所有需要的列都存在，否则返回错误信息
    required_cols = [
        '日期', 'Price', '工业指标', '基线阈值', '采购信号', '动态短窗口', '动态长窗口',
        'SMA动态短', '动量因子', '修正RSI', 'EMA21', '布林下轨', 'ema_ratio',
        'dynamic_ema_threshold', '低波动阈值', 'EMA趋势', '波动下轨', '波动上轨',
        'core_cond1_met', 'core_cond2_met', 'core_cond3_met', 'core_cond4_met',
        'core_cond5_met', 'core_cond6_met'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return f"<h2>⚠️ 报告生成失败：缺失列 {', '.join(missing_cols)}</h2>"

    # 填充可能存在的NaN值，避免格式化错误
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    # 对于特定列，提供更合理的默认值
    df.fillna({'修正RSI': 50, '动量因子': 0, 'ATR': 0, '波动上轨': df['Price'], '波动下轨': df['Price']}, inplace=True)


    current = df.iloc[-1]
    def safe_float(value, default=0.0):
        # ... (safe_float 函数不变) ...
        try:
            f_val = float(value)
            if pd.isna(f_val) or not np.isfinite(f_val): return default
            return f_val
        except (ValueError, TypeError): return default

    # --- 计算所有需要的数值 ---
    price = safe_float(current['Price'])
    indicator = safe_float(current['工业指标'])
    threshold = safe_float(current['基线阈值'])
    short_sma = safe_float(current['SMA动态短'], default=price)
    long_sma = safe_float(current.get('SMA动态长', price), default=price)
    volatility = safe_float(current['动量因子'])
    rsi = safe_float(current['修正RSI'], default=50)
    ema9 = safe_float(current.get('EMA9', price), default=price)
    ema21 = safe_float(current['EMA21'], default=price)
    ema50 = safe_float(current.get('EMA50', price), default=price)
    lower_band = safe_float(current['布林下轨'], default=price * 0.95)
    ema_ratio = safe_float(current['ema_ratio'], default=1.0)
    dynamic_threshold = safe_float(current['dynamic_ema_threshold'], default=1.0)
    vol_threshold = safe_float(current['低波动阈值'], default=0.01)
    atr_lower = safe_float(current['波动下轨'], default=price * 0.95)
    atr_upper = safe_float(current['波动上轨'], default=price * 1.05)
    price_trend_vs_sma = ((price / short_sma) - 1) * 100 if short_sma != 0 else 0
    dynamic_short_window = int(current.get('动态短窗口', BASE_WINDOW_SHORT))
    dynamic_long_window = int(current.get('动态长窗口', BASE_WINDOW_LONG))

    # --- 定义悬停提示文本 (HOVER_TEXTS) ---
    HOVER_TEXTS = {
        'price': "从数据源获取的每日收盘价。",
        'indicator': "计算思路: (价格/短期均线) * (价格/长期均线) * (1 - 动量因子)。综合衡量价格位置和波动性。",
        'threshold': f"计算思路: 最近 {HISTORY_WINDOW} 天工业指标的25%分位数。是工业指标的动态买入参考线。",
        'signal': "综合所有核心条件和阻断规则得出的最终建议。",
        'dynamic_window': f"计算思路: 基准窗口({BASE_WINDOW_SHORT}/{BASE_WINDOW_LONG}天)根据距离上次购买天数进行衰减({WINDOW_DECAY_RATE}率)，最短{MIN_WINDOW_SHORT}天。距离越久，窗口越短，越灵敏。",
        'price_trend': "计算思路: (当前价格 / 短期动态均线 - 1) * 100%。表示价格偏离近期平均成本的程度。",
        'volatility': f"计算思路: 最近{dynamic_short_window}天内每日价格变化百分比绝对值的平均值。此指标衡量价格波动的剧烈程度（即近期波动率），值越低表示市场越平静。",
        'core_cond1': f"工业指标 ({indicator:.2f}) 是否低于基线阈值 ({threshold:.2f})？",
        'core_cond2': f"修正RSI ({rsi:.1f}) 是否低于 45？RSI通过计算一定时期内上涨日和下跌日的平均涨跌幅得到，衡量买卖力量，低于45通常表示超卖。",
        'core_cond3': f"当前价格 ({price:.2f}) 是否低于 EMA21 ({ema21:.2f})？EMA是指数移动平均线，给予近期价格更高权重。",
        'core_cond4': f"当前价格 ({price:.2f}) 是否低于布林下轨 ({lower_band:.2f}) 的 1.05 倍 ({lower_band * 1.05:.2f})？布林通道基于移动平均线加减标准差得到，衡量价格相对波动范围。",
        'core_cond5': f"EMA9/EMA21比率 ({ema_ratio:.3f}) 是否大于动态阈值 ({dynamic_threshold:.3f})？该阈值会根据波动性调整。",
        'core_cond6': f"动量因子 ({volatility:.3f}) 是否低于其动态阈值 ({vol_threshold:.3f})？该阈值是动量因子自身的45日35%分位数。",
        # ... (其他 HOVER_TEXTS 保持不变) ...
         'peak_filter': f"一个内部过滤器，检查近3日价格形态是否不利（如冲高回落），以及价格是否处于ATR计算的通道上轨({atr_upper:.2f})80%以上位置，用于排除一些潜在的顶部信号。",
         'interval': f"距离上次系统发出买入信号的天数，要求至少间隔 {MIN_PURCHASE_INTERVAL} 天才能再次买入。",
    }
    # --- 定义六个核心买入条件的简短描述 (用于动态分析部分) ---
    CONDITION_EXPLANATIONS_SHORT = {
        'cond1': ("工业指标 < 阈值", f"{indicator:.2f} < {threshold:.2f}"),
        'cond2': ("RSI < 45 (超卖区域)", f"RSI {rsi:.1f} < 45"),
        'cond3': ("价格 < EMA21", f"价格 {price:.2f} < EMA21 {ema21:.2f}"),
        'cond4': ("价格 < 布林下轨附近", f"价格 {price:.2f} < 下轨参考 {lower_band * 1.05:.2f}"),
        'cond5': ("短期EMA动能 > 阈值", f"EMA比率 {ema_ratio:.3f} > 阈值 {dynamic_threshold:.3f}"),
        'cond6': ("波动性 < 阈值 (市场平静)", f"波动 {volatility:.3f} < 阈值 {vol_threshold:.3f}")
    }


    # --- 计算用于动态分析的数据 ---
    condition_scores = sum([current.get(f'core_cond{i}_met', False) for i in range(1, 7)])
    base_req_met = condition_scores >= 4
    peak_filter_series = peak_filter(df)
    peak_filter_passed = peak_filter_series.iloc[-1] if isinstance(peak_filter_series, pd.Series) else True
    atr_denominator = atr_upper - atr_lower
    atr_value = ((price - atr_lower) / atr_denominator) * 100 if atr_denominator != 0 else 50.0
    atr_overbought = atr_value > 80
    if not peak_filter_passed: peak_status_display = '<span style="color:red;">形态不利</span>'
    elif atr_overbought: peak_status_display = f'<span style="color:red;">ATR超买({atr_value:.1f}%)</span>'
    else: peak_status_display = '<span style="color:green;">通过</span>'

    last_signal_index = df[df['采购信号']].index[-1] if df['采购信号'].any() else -1
    interval_days = len(df) - 1 - last_signal_index if last_signal_index != -1 else 999
    interval_ok = interval_days >= MIN_PURCHASE_INTERVAL
    interval_check_text = '<span style="color:green;">满足</span>' if interval_ok else f'<span style="color:orange;">不满足 (还需等待 {max(0, MIN_PURCHASE_INTERVAL - interval_days)}天)</span>'

    block_reasons = []
    if not base_req_met: block_reasons.append(f"核心条件不足({condition_scores}/6)")
    if not interval_ok: block_reasons.append(f"采购间隔限制(还需{max(0, MIN_PURCHASE_INTERVAL - interval_days)}天)")
    if not peak_filter_passed: block_reasons.append("价格形态不利")
    if atr_overbought: block_reasons.append(f"ATR通道超买({atr_value:.1f}%)")

    # --- 返回包含所有计算结果和状态的字典 ---
    return {
        'current_date': current['日期'],
        'price': price,
        'indicator': indicator,
        'threshold': threshold,
        'signal': current['采购信号'],
        'dynamic_short_window': dynamic_short_window,
        'dynamic_long_window': dynamic_long_window,
        'price_trend_vs_sma': price_trend_vs_sma,
        'volatility': volatility,
        'hover_texts': HOVER_TEXTS, # 传递悬停文本
        'condition_scores': condition_scores,
        'conditions_explanation_short': CONDITION_EXPLANATIONS_SHORT, # 传递简短解释
        'current_conditions_met': {f'cond{i}': current.get(f'core_cond{i}_met', False) for i in range(1, 7)},
        'peak_status_display': peak_status_display,
        'interval_days': interval_days,
        'interval_check_text': interval_check_text,
        'min_purchase_interval': MIN_PURCHASE_INTERVAL,
        'base_req_met': base_req_met,
        'block_reasons': block_reasons
    }

def create_visualization(df):
    """
    使用 Plotly 生成交互式 HTML 图表，包含三个子图，帮助可视化分析。
    新增功能：鼠标悬停在图表线上时，会显示该线的名称、数值以及简要计算说明。
    图表解读指南... (保持不变)
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('价格与信号 (看红色三角是否出现)',
                                        '策略指标分析 (看蓝色线是否低于红色虚线/进入绿色区域)',
                                        '动量指标分析 (看紫色线是否低于红色点线)'))

    # --- 定义悬停模板 ---
    # <extra></extra> 用于移除 Plotly 默认添加的额外信息框
    hovertemplate_price = "<b>价格</b>: %{y:.2f} USD<br>日期: %{x|%Y-%m-%d}<br><i>来源: 每日收盘价</i><extra></extra>"
    hovertemplate_sma = "<b>%{data.name}</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 最近%{customdata}天收盘价的算术平均</i><extra></extra>"
    hovertemplate_ema = "<b>%{data.name}</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 指数移动平均，近期价格权重更高</i><extra></extra>"
    hovertemplate_signal = "<b>⭐采购信号⭐</b><br>价格: %{y:.2f} USD<br>日期: %{x|%Y-%m-%d}<br><i>策略建议买入点</i><extra></extra>"
    hovertemplate_indicator = "<b>核心工业指标</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: (价/短均)*(价/长均)*(1-动量)</i><extra></extra>"
    hovertemplate_threshold = "<b>%{data.name}</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 近期工业指标的25%分位数</i><extra></extra>"
    hovertemplate_rsi = "<b>修正RSI</b>: %{y:.1f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 基于14日平均涨跌幅，衡量超买超卖</i><extra></extra>"
    hovertemplate_rsi_threshold = "<b>动态RSI阈值</b>: %{y:.1f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 近63日RSI的30%分位数</i><extra></extra>"
    hovertemplate_fill = "<b>指标低于阈值区域</b><br>日期: %{x|%Y-%m-%d}<br>工业指标: %{y:.2f}<br><i>满足买入条件1</i><extra></extra>"


    # --- 行 1: 价格与信号 ---
    fig.add_trace(go.Scatter(x=df['日期'], y=df['Price'], mode='lines', name='白银价格 (USD)',
                             line=dict(color='navy', width=1.5), legendgroup='price', legendrank=1,
                             hovertemplate=hovertemplate_price),
                  row=1, col=1)
    # 注意：为SMA动态短添加 customdata 以便悬停时显示窗口天数
    fig.add_trace(go.Scatter(x=df['日期'], y=df['SMA动态短'], mode='lines', name='短期均线 (近期趋势)',
                             line=dict(color='darkorange', dash='dash'), legendgroup='price', legendrank=2,
                             customdata=df['动态短窗口'], # 将窗口天数传给 customdata
                             hovertemplate=hovertemplate_sma),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['EMA9'], mode='lines', name='EMA9 (更短趋势)',
                             line=dict(color='firebrick', width=1), legendgroup='price', legendrank=3, opacity=0.7,
                             hovertemplate=hovertemplate_ema),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['EMA21'], mode='lines', name='EMA21 (中期趋势)',
                             line=dict(color='seagreen', width=1), legendgroup='price', legendrank=4, opacity=0.7,
                             hovertemplate=hovertemplate_ema),
                  row=1, col=1)

    signal_df = df[df['采购信号']]
    if not signal_df.empty:
        fig.add_trace(go.Scatter(x=signal_df['日期'], y=signal_df['Price'], mode='markers', name='⭐采购信号⭐',
                                 marker=dict(color='red', size=8, symbol='triangle-up', line=dict(width=1, color='black')),
                                 legendgroup='signal', legendrank=5,
                                 hovertemplate=hovertemplate_signal),
                      row=1, col=1)

    # --- 行 2: 策略指标分析 ---
    fig.add_trace(go.Scatter(x=df['日期'], y=df['工业指标'], mode='lines', name='核心工业指标',
                             line=dict(color='royalblue'), legendgroup='indicator', legendrank=6,
                             hovertemplate=hovertemplate_indicator),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['基线阈值_短'], mode='lines', name=f'短期阈值 ({HISTORY_WINDOW_SHORT}日)',
                             line=dict(color='darkorange', dash='dot', width=1), legendgroup='indicator', legendrank=7, opacity=0.7,
                             hovertemplate=hovertemplate_threshold),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['基线阈值'], mode='lines', name=f'中期阈值 ({HISTORY_WINDOW}日) - 警戒线',
                             line=dict(color='crimson', dash='dash', width=1.5), legendgroup='indicator', legendrank=8,
                             hovertemplate=hovertemplate_threshold),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['基线阈值_长'], mode='lines', name=f'长期阈值 ({HISTORY_WINDOW_LONG}日)',
                             line=dict(color='purple', dash='dashdot', width=1), legendgroup='indicator', legendrank=9, opacity=0.8,
                             hovertemplate=hovertemplate_threshold),
                  row=2, col=1)

    # --- 修改填充逻辑：使用 NaN 来创建条件性填充 ---
    y_upper = df['基线阈值']
    y_lower = df['工业指标']
    # 创建一个新的 Series 用于填充，默认等于阈值（填充的上限）
    y_fill_lower = y_upper.copy()
    # 只有当工业指标严格小于阈值时，才将填充下边界设置为工业指标的值
    fill_mask = y_lower < y_upper
    y_fill_lower[fill_mask] = y_lower[fill_mask]
    # 在工业指标 >= 阈值的地方，y_fill_lower 保持为 y_upper 的值，这样填充会闭合到线上，视觉上无填充
    # 或者，另一种方法是设置为 NaN，明确断开填充：
    # y_fill_lower_nan = y_lower.copy()
    # y_fill_lower_nan[~fill_mask] = np.nan
    # 我们先尝试让不满足条件的区域填充到阈值线上，如果效果不好再换 NaN

    # 添加透明的上/目标边界线 (实际是阈值线)
    fig.add_trace(go.Scatter(x=df['日期'], y=y_upper, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False, hoverinfo='skip'), row=2, col=1)

    # 添加用于填充的轨迹。x是完整的日期，y是经过条件处理的y_fill_lower
    # 当 y_fill_lower 等于 y_upper 时，填充区域高度为0，不可见。
    # 当 y_fill_lower 等于 y_lower (且小于y_upper) 时，填充可见。
    fig.add_trace(go.Scatter(x=df['日期'], y=y_fill_lower, # 使用条件处理后的下边界
                             fill='tonexty', # 填充到上一条轨迹 (透明的y_upper)
                             mode='lines',
                             line=dict(width=0), # 设置线条宽度为0，只显示填充
                             fillcolor='rgba(144, 238, 144, 0.3)',
                             name='指标低于阈值区域 (买入条件1)',
                             legendgroup='indicator',
                             legendrank=10,
                             # hoverinfo='skip' # 不对填充区域本身设置悬停，让下方的指标线响应悬停
                             # 或者保留悬停，但可能显示阈值或指标值
                             hovertemplate=hovertemplate_fill # 保持悬停模板
                             ), row=2, col=1)
    # --- 结束修改填充逻辑 ---

    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="指标参考基准=1", row=2, col=1)

    # --- 行 3: 动量指标分析 ---
    fig.add_trace(go.Scatter(x=df['日期'], y=df['修正RSI'], mode='lines', name='修正RSI (市场强弱)',
                             line=dict(color='darkviolet'), legendgroup='momentum', legendrank=11,
                             hovertemplate=hovertemplate_rsi),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['RSI阈值'], mode='lines', name='动态RSI阈值',
                             line=dict(color='darkorange', dash='dash'), legendgroup='momentum', legendrank=12,
                             hovertemplate=hovertemplate_rsi_threshold),
                  row=3, col=1)
    fig.add_hline(y=45, line_dash="dot", line_color="red", opacity=0.5, annotation_text="RSI超卖参考线=45 (买入条件2)", row=3, col=1, name="RSI 45")

    # --- 更新整体布局 ---
    fig.update_layout(
        height=900,
        title_text='银价分析与策略可视化 (交互式图表)',
        hovermode='x unified',
        legend_title_text='图例说明',
        margin=dict(l=60, r=60, t=100, b=60)
    )
    fig.update_yaxes(title_text="价格 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="指标值", row=2, col=1)
    fig.update_yaxes(title_text="RSI 值 (0-100)", row=3, col=1)
    fig.update_xaxes(title_text="日期", row=3, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    return fig


# --- 主程序：生成 HTML 报告 ---
if __name__ == "__main__":
    print("开始执行银价分析...")

    # 1. 加载数据
    print("正在加载数据...")
    df = load_silver_data()
    df['采购信号'] = False # 初始化信号列

    # 2. 计算策略与信号 (执行两轮)
    print("正在计算策略与信号 (第一轮)...")
    df = calculate_strategy(df)
    df = generate_signals(df)
    print("正在计算策略与信号 (第二轮)...")
    df = calculate_strategy(df)
    df = generate_signals(df)

    # 3. 生成报告所需数据
    print("正在生成报告数据...")
    analysis_data = generate_report(df) # 直接接收包含所有数据的字典

    # 4. 生成图表 Figure 对象
    print("正在生成图表...")
    fig = create_visualization(df)

    # 5. 将图表转换为 HTML div
    print("正在转换图表为HTML...")
    try:
        chart_html_div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        if not chart_html_div or len(chart_html_div.strip()) == 0:
             print("警告：生成的图表 HTML 为空。")
             chart_html_div = "<p style='color:orange;'>图表生成似乎为空。</p>"
    except Exception as e:
        print(f"错误：将 Plotly 图表转换为 HTML 时失败: {e}")
        chart_html_div = "<p style='color:red;'>图表生成失败。</p>"

    # 6. 构建包含两列布局的完整 HTML 页面
    print("正在构建最终HTML页面...")
    # 提取常用变量简化后续代码
    hover_texts = analysis_data['hover_texts']
    signal = analysis_data['signal']
    condition_scores = analysis_data['condition_scores']
    conditions_explanation_short = analysis_data['conditions_explanation_short']
    current_conditions_met = analysis_data['current_conditions_met']
    base_req_met = analysis_data['base_req_met']
    block_reasons = analysis_data['block_reasons']


    # 构建"今日解读"部分的结论文本
    if signal:
        conclusion_html = f'<li><strong>结论：</strong><span style="color:green;">由于满足了足够的买入条件 ({condition_scores}/6)，且没有触发阻断信号，因此策略建议采购。</span></li>'
    else:
        reason_prefix = f"由于满足的核心条件数量不足 ({condition_scores}/6)，" if not base_req_met else ""
        reason_block = f"具体阻断原因：{' + '.join(block_reasons)}" if block_reasons else ("" if base_req_met else f"核心条件未满足 ({condition_scores}/6)。") # 如果有阻断原因就列出，否则如果是条件不足导致就说明条件不足
        conclusion_html = f'<li><strong>结论：</strong><span style="color:red;">{reason_prefix}{reason_block}因此策略建议持币观望。</span></li>'


    final_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>银价分析报告</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; font-size: 14px; }} /* 基础字体大小 */
        .main-container {{ display: flex; flex-wrap: wrap; gap: 20px; }} /* Flex 布局容器 */
        .left-column {{ flex: 3; min-width: 600px; }} /* 左栏（图表），允许收缩但有最小宽度 */
        .right-column {{ flex: 2; min-width: 350px; }} /* 右栏（文本），允许收缩但有最小宽度 */
        .report-section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
        .chart-container {{ border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); padding: 10px; }}
        h1 {{ text-align: center; width: 100%; margin-bottom: 20px; font-size: 1.8em; }}
        h2 {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 0; font-size: 1.4em; }}
        h3 {{ color: #444; margin-bottom: 10px; font-size: 1.2em; }}
        ul {{ padding-left: 20px; margin-top: 5px; }}
        li {{ margin-bottom: 8px; }} /* 增加列表项间距 */
        .interpretation-list li {{ margin-bottom: 12px; }} /* 解读列表项间距更大 */
        .summary-item {{ margin-bottom: 10px; }}
        .summary-label {{ font-weight: bold; }}
        .tooltip-trigger {{ border-bottom: 1px dotted #bbb; cursor: help; position: relative; }}
        .tooltip-trigger:hover::after {{ content: attr(data-tooltip); position: absolute; left: 50%; transform: translateX(-50%); bottom: 125%; white-space: pre-wrap; background-color: rgba(0, 0, 0, 0.85); color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 12px; z-index: 10; width: max-content; max-width: 280px; box-shadow: 1px 1px 3px rgba(0,0,0,0.2); }}
        .legend text {{ font-size: 10px !important; }} /* 进一步缩小图例字体 */
        .annotation-text {{ font-size: 9px !important; }} /* 进一步缩小注释字体 */
        .interpretation-section ul {{ list-style-type: none; padding-left: 0; }} /* 移除列表标记 */
        .interpretation-section li strong {{ color: #0056b3; }} /* 强调关键名词 */
    </style>
</head>
<body>
    <h1>银价走势分析与定投参考报告</h1>
    <p style="text-align: center; margin-top: -15px; margin-bottom: 25px; font-size: 0.9em; color: #666;">报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="main-container">
        <div class="left-column">
            <div class="chart-container">
                 <h2>📊 交互式图表分析</h2>
                 <p style="font-size: 0.9em; color: #555;">将鼠标悬停在图表线上可查看详细数据。您可以缩放和平移图表进行探索。</p>
                {chart_html_div}
            </div>
        </div>

        <div class="right-column">
            <div class="report-section">
                <h2>📈 关键指标与最新信号</h2>
                <div class="summary-item">
                    <span class="summary-label">报告日期：</span>{analysis_data['current_date'].strftime('%Y-%m-%d')}
                </div>
                <div class="summary-item">
                    <span class="summary-label tooltip-trigger" data-tooltip="{hover_texts['price']}">当前价格：</span>{analysis_data['price']:.2f} USD
                </div>
                <div class="summary-item">
                    <span class="summary-label tooltip-trigger" data-tooltip="{hover_texts['indicator']}">核心指标：</span>{analysis_data['indicator']:.2f}
                    <span class="tooltip-trigger" data-tooltip="{hover_texts['threshold']}" style="font-size: 0.9em; color: #666;"> (参考阈值: < {analysis_data['threshold']:.2f})</span>
                </div>
                 <div class="summary-item">
                    <span class="summary-label tooltip-trigger" data-tooltip="{hover_texts['dynamic_window']}">动态窗口 (短/长)：</span>{analysis_data['dynamic_short_window']}天 / {analysis_data['dynamic_long_window']}天
                </div>
                 <div class="summary-item">
                    <span class="summary-label tooltip-trigger" data-tooltip="{hover_texts['volatility']}">市场波动性：</span>{analysis_data['volatility']*100:.1f}%
                </div>
                <div class="summary-item" style="margin-top: 15px;">
                    <span class="summary-label tooltip-trigger" data-tooltip="{hover_texts['signal']}">今日策略建议：</span>
                    {'<span style="color:green; font-weight:bold;">建议采购</span>' if signal else '<span style="color:orange; font-weight:bold;">建议持币观望</span>'}
                </div>
            </div>

            <div class="report-section interpretation-section">
                <h2>📖 图表解读与策略说明</h2>
                <p style="font-size: 0.9em; color: #555; margin-bottom: 15px;"><strong>提示：</strong>以下解读旨在帮助理解图表信息和策略逻辑，并非投资操作指南。</p>
                <h3>图表元素含义：</h3>
                <ul>
                    <li><strong>价格 (深蓝线):</strong> 每日收盘价，是所有分析的基础。</li>
                    <li><strong>短期均线 (橙虚线):</strong> 近期价格的平均趋势线。</li>
                    <li><strong>EMA (红/绿细线):</strong> 指数移动平均线，更关注近期价格变化，辅助判断短期动能。</li>
                    <li><strong>采购信号 (<span style='color:red; font-size: 1.2em;'>▲</span>):</strong> 策略输出的潜在买入时机提示。</li>
                    <li><strong>核心工业指标 (中图，蓝线):</strong> 策略核心，衡量价格相对历史水平和波动性的综合指标。</li>
                    <li><strong>中期阈值 (中图，红虚线):</strong> 核心指标的关键参考线，低于此线是主要买入条件之一。</li>
                    <li><strong>修正RSI (下图，紫线):</strong> 市场相对强弱指标，低于45 (红点线) 是另一主要买入条件。</li>
                </ul>
                <h3>策略决策逻辑：</h3>
                <ol style="padding-left: 20px;">
                    <li>满足至少4项核心条件（见下方"今日解读"中的具体条件）。</li>
                    <li>且无信号阻断情况（近期形态、ATR超买、采购间隔过短）。</li>
                </ol>
                <p style="font-size: 0.9em; color: #555;">策略旨在寻找核心指标和RSI均显示价格可能偏低，且市场未出现明显风险信号的时机。</p>
            </div>

            <div class="report-section">
                 <h3 style="background-color: #f0f0f0; padding: 8px 12px; margin: -15px -15px 15px -15px; border-left: 5px solid #007bff; font-size: 1.1em;">💡 对今天 ({analysis_data['current_date'].strftime('%Y-%m-%d')}) 的解读</h3>
                 <p style="margin-top: -5px; margin-bottom: 15px;"><strong>今日策略建议：{'<span style="color:green; font-weight:bold;">建议采购</span>' if signal else '<span style="color:orange; font-weight:bold;">建议持币观望</span>'}</strong></p>
                <p><strong>原因分析：</strong></p>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>主要条件满足情况 (要求≥4项): <strong>{condition_scores}/6</strong> 项满足。
                        <ul style="font-size: 0.9em; color: #333; margin-top: 8px; padding-left: 15px;">
                            {''.join([f'<li style="margin-bottom: 4px;" class="tooltip-trigger" data-tooltip="{hover_texts[f"core_cond{i}"]}"><span style="color:{"green" if current_conditions_met[f"cond{i}"] else "red"}; margin-right: 5px;">{ "✔️" if current_conditions_met[f"cond{i}"] else "❌"}</span> 条件{i} ({conditions_explanation_short[f"cond{i}"][0]}): {conditions_explanation_short[f"cond{i}"][1]}</li>' for i in range(1, 7)])}
                        </ul>
                    </li>
                    <li style="margin-top: 10px;">信号阻断检查:
                        <ul style="font-size: 0.9em; color: #333; margin-top: 8px; padding-left: 15px;">
                            <li style="margin-bottom: 4px;" class="tooltip-trigger" data-tooltip="{hover_texts['peak_filter']}">价格形态/ATR过滤: {analysis_data['peak_status_display']}</li>
                            <li style="margin-bottom: 4px;" class="tooltip-trigger" data-tooltip="{hover_texts['interval']}">采购间隔检查 (≥{analysis_data['min_purchase_interval']}天): {analysis_data['interval_days']} 天 → {analysis_data['interval_check_text']}</li>
                        </ul>
                    </li>
                    <li style="margin-top: 15px; border-top: 1px dashed #ccc; padding-top: 10px;">
                       {conclusion_html}
                    </li>
                </ul>
                <p style="font-size: 0.85em; color: #888; margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px;"><strong>再次强调：</strong> 本分析仅为程序化策略的演示输出，不构成任何投资建议。市场有风险，请独立判断，谨慎决策。</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

    # 7. 将完整的 HTML 写入文件
    output_filename = "index.html"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"成功将报告写入文件: {output_filename}")

        # 8. 自动执行 Git 命令推送到 GitHub
        print("尝试将更新推送到 GitHub...")
        try:
            # 定义 Git 命令
            commit_message = f"Update report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            # 添加 index.html, .gitignore 和脚本本身
            files_to_add = [output_filename, ".gitignore", __file__]
            git_commands = [
                ["git", "add"] + files_to_add,
                ["git", "commit", "-m", commit_message],
                ["git", "push", "origin", "master"] # 推送到 origin 的 master 分支
            ]

            for cmd in git_commands:
                print(f"执行命令: {' '.join(cmd)}")
                # 使用 capture_output=True 捕获输出, text=True 转为字符串
                # check=False 不会在出错时抛出异常，我们手动检查返回值
                # 指定 utf-8 编码以防 Windows 默认编码问题
                result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')

                # 打印标准输出和标准错误
                if result.stdout:
                    print(f"Git 输出:\n{result.stdout.strip()}")
                if result.stderr:
                    # 忽略常见的 "nothing to commit" 和 "up-to-date"，因为我们总是尝试提交和推送
                    if "nothing to commit" not in result.stderr and "up-to-date" not in result.stderr:
                         print(f"Git 错误:\n{result.stderr.strip()}")
                    else:
                        print(f"Git 信息 (可忽略): {result.stderr.strip()}")

                # 如果命令失败（非 0 返回码），且不是可忽略的信息，则停止后续命令
                if result.returncode != 0 and ("nothing to commit" not in result.stderr and "up-to-date" not in result.stderr):
                    print(f"Git 命令执行失败，返回码: {result.returncode}。停止推送。")
                    break # 遇到真实错误则停止
            else:
                 # 如果循环正常结束（没有 break），则打印完成信息
                 print("GitHub 推送操作序列完成（或无事可做）。")

        except FileNotFoundError:
            print("错误：找不到 'git' 命令。请确保 Git 已安装并添加到系统 PATH。")
        except Exception as git_e:
            print(f"错误：执行 Git 命令时出错: {git_e}")

    except Exception as e:
        print(f"错误：写入 HTML 文件或执行 Git 命令时出错: {e}")


    print("分析完成。")


