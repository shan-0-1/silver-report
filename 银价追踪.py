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
        os.path.join(BASE_DIR, 'XAG_CNY历史数据.csv'), # 修改文件名：USD -> CNY
        r'C:\Users\assistant3\Downloads\XAG_CNY历史数据.csv' # 修改文件名：USD -> CNY
    ]

    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"成功找到数据文件: {csv_path}")
            break

    if csv_path is None:
         print(f"错误：在以下位置均未找到 'XAG_CNY历史数据.csv':")
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

    # --- 新增：直接比较 EMA9 和 EMA21 用于视觉交叉判断 --- 
    # 填充 EMA 计算初期的 NaN 值，避免比较错误
    df['EMA9'].fillna(method='bfill', inplace=True)
    df['EMA21'].fillna(method='bfill', inplace=True)
    df['ema9_above_ema21'] = df['EMA9'] > df['EMA21']
    # --- 结束新增 --- 

    df['ema_ratio'] = df['EMA9'] / df['EMA21'].replace(0, np.nan) # 避免除以零
    df['ema_ratio'] = df['ema_ratio'].fillna(1.0) # 中性填充

    # 修改EMA金叉条件计算公式 (此列仍用于 core_cond5)
    df['dynamic_ema_threshold'] = 1 + (0.5 * df['动量因子'])  # 使阈值与波动率正相关
    df['EMA金叉'] = df['ema_ratio'] > df['dynamic_ema_threshold'] 
    # 增加EMA合理性检查 (此列仍用于 core_cond5)
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

    # --- 确保 current 中的值是有效的数字 ---
    def safe_float(value, default=0.0):
        try:
            # 先尝试直接转换
            f_val = float(value)
            # 检查是否为 NaN 或无穷大
            if pd.isna(f_val) or not np.isfinite(f_val):
                return default
            return f_val
        except (ValueError, TypeError):
            return default

    price = safe_float(current['Price'])
    indicator = safe_float(current['工业指标']) # "工业指标"是本策略的核心，衡量价格相对历史均值和波动性的位置
    threshold = safe_float(current['基线阈值']) # "基线阈值"是工业指标的动态门槛，低于此值表明价格可能偏低
    short_sma = safe_float(current['SMA动态短'], default=price) # 短期移动平均线，反映近期价格趋势
    long_sma = safe_float(current.get('SMA动态长', price), default=price) # 长期移动平均线 (报告中未直接显示，但用于计算工业指标)
    volatility = safe_float(current['动量因子']) # "动量因子"衡量价格波动的剧烈程度，低波动有时是买入时机
    rsi = safe_float(current['修正RSI'], default=50) # 修正后的相对强弱指数(RSI)，低于特定值（如45）通常表示超卖，可能是买点
    ema9 = safe_float(current.get('EMA9', price), default=price) # 9日指数移动平均线
    ema21 = safe_float(current['EMA21'], default=price) # 21日指数移动平均线，价格低于它表示短期偏弱
    ema50 = safe_float(current.get('EMA50', price), default=price) # 50日指数移动平均线
    bollinger_mid = safe_float(current.get('布林中轨', price), default=price) # 布林通道中轨 (通常是20日简单移动平均)
    # 尝试从 current 获取 std，如果不存在则重新计算最后值
    rolling_std_series = df['Price'].rolling(20).std()
    bollinger_std = safe_float(current.get('布林标准差', rolling_std_series.iloc[-1]), default=price*0.05) # 布林通道标准差
    lower_band = safe_float(current['布林下轨'], default=price * 0.95) # 布林通道下轨，价格接近或跌破下轨可能表示超卖
    ema_ratio = safe_float(current['ema_ratio'], default=1.0) # 短期EMA与中期EMA的比率，用于判断趋势动能
    dynamic_threshold = safe_float(current['dynamic_ema_threshold'], default=1.0) # EMA比率的动态阈值
    vol_threshold = safe_float(current['低波动阈值'], default=0.01) # 动量因子的动态阈值
    atr_lower = safe_float(current['波动下轨'], default=price * 0.95) # 基于ATR计算的波动下轨
    atr_upper = safe_float(current['波动上轨'], default=price * 1.05) # 基于ATR计算的波动上轨

    # 计算当前价格相对于短期均线的百分比偏差
    price_trend_vs_sma = ((price / short_sma) - 1) * 100 if short_sma != 0 else 0

    # --- 定义悬停提示信息 ---
    HOVER_TEXTS = {
        'price': "从数据源获取的每日收盘价。",
        'indicator': "计算思路: (价格/短期均线) * (价格/长期均线) * (1 - 动量因子)。综合衡量价格位置和波动性。",
        'threshold': f"计算思路: 最近 {HISTORY_WINDOW} 天工业指标的25%分位数。是工业指标的动态买入参考线。",
        'signal': "综合所有核心条件和阻断规则得出的最终建议。",
        'dynamic_window': f"计算思路: 基准窗口({BASE_WINDOW_SHORT}/{BASE_WINDOW_LONG}天)根据距离上次购买天数进行衰减({WINDOW_DECAY_RATE}率)，最短{MIN_WINDOW_SHORT}天。距离越久，窗口越短，越灵敏。",
        'price_trend': "计算思路: (当前价格 / 短期动态均线 - 1) * 100%。表示价格偏离近期平均成本的程度。",
        'volatility': f"计算思路: 最近{{dynamic_short_window_val}}天内每日价格变化百分比绝对值的平均值。此指标衡量价格波动的剧烈程度（即近期波动率），值越低表示市场越平静。注意：名称可能易误导，它主要反映波动性而非趋势动量。", # 使用占位符
        'core_cond1': f"工业指标 ({indicator:.2f}) 是否低于基线阈值 ({threshold:.2f})？",
        'core_cond2': f"修正RSI ({rsi:.1f}) 是否低于 45？RSI通过计算一定时期内上涨日和下跌日的平均涨跌幅得到，衡量买卖力量，低于45通常表示超卖。",
        'core_cond3': f"当前价格 ({price:.2f}) 是否低于 EMA21 ({ema21:.2f})？EMA是指数移动平均线，给予近期价格更高权重。",
        'core_cond4': f"当前价格 ({price:.2f}) 是否低于布林下轨 ({lower_band:.2f}) 的 1.05 倍 ({lower_band * 1.05:.2f})？布林通道基于移动平均线加减标准差得到，衡量价格相对波动范围。",
        'core_cond5': f"EMA9/EMA21比率 ({ema_ratio:.3f}) 是否大于动态阈值 ({dynamic_threshold:.3f})？该阈值会根据波动性调整。",
        'core_cond6': f"动量因子 ({volatility:.3f}) 是否低于其动态阈值 ({vol_threshold:.3f})？该阈值是动量因子自身的45日35%分位数。",
        'cond_score': "满足以上6个核心条件的数量，至少需要满足4个才能初步考虑买入。",
        'peak_filter': f"一个内部过滤器，检查近3日价格形态是否不利（如冲高回落），以及价格是否处于ATR计算的通道上轨({{atr_upper_val:.2f}})80%以上位置，用于排除一些潜在的顶部信号。", # 使用占位符
        'interval': f"距离上次系统发出买入信号的天数，要求至少间隔 {MIN_PURCHASE_INTERVAL} 天才能再次买入。",
        'window_decay': "显示当前动态短窗口相比基准窗口缩短了多少天，反映了衰减机制的效果。",
        'ema_trend': f"基于EMA9({{ema9_val:.2f}}), EMA21({{ema21_val:.2f}}), EMA50({{ema50_val:.2f}})的相对位置判断短期趋势。当EMA9>EMA21且EMA21>EMA50时为多头，反之为空头。", # 使用占位符
        'final_block': "总结导致最终未能产生买入信号的具体原因。",
        '3day_change': "最近三个交易日的价格变化绝对值和方向。",
        'ema_crossover': "基于 EMA9 和 EMA21 的直接相对位置。金叉状态 (EMA9 > EMA21) 通常视为看涨倾向，死叉状态 (EMA9 < EMA21) 通常视为看跌倾向。图表上的标记 (↑/↓) 显示精确的交叉点。" # 新增EMA交叉解释
    }

    # --- 构建 HTML 报告字符串 ---
    # 使用 format 方法动态填充 HOVER_TEXTS 中的变量
    dynamic_short_window_val = int(current.get('动态短窗口', BASE_WINDOW_SHORT))
    atr_upper_val = safe_float(current.get('波动上轨', price * 1.05))
    ema9_val = safe_float(current.get('EMA9', price))
    ema21_val = safe_float(current['EMA21'], default=price)
    ema50_val = safe_float(current.get('EMA50', price))

    # 填充 HOVER_TEXTS
    for key in HOVER_TEXTS:
        try:
            HOVER_TEXTS[key] = HOVER_TEXTS[key].format(
                HISTORY_WINDOW=HISTORY_WINDOW,
                BASE_WINDOW_SHORT=BASE_WINDOW_SHORT,
                BASE_WINDOW_LONG=BASE_WINDOW_LONG,
                WINDOW_DECAY_RATE=WINDOW_DECAY_RATE,
                MIN_WINDOW_SHORT=MIN_WINDOW_SHORT,
                indicator=indicator, threshold=threshold,
                rsi=rsi, price=price, ema21=ema21, lower_band=lower_band,
                ema_ratio=ema_ratio, dynamic_threshold=dynamic_threshold,
                volatility=volatility, vol_threshold=vol_threshold,
                atr_upper_val=atr_upper_val, # 使用填充后的值
                MIN_PURCHASE_INTERVAL=MIN_PURCHASE_INTERVAL,
                ema9_val=ema9_val, # 使用填充后的值
                ema21_val=ema21_val, # 使用填充后的值
                ema50_val=ema50_val, # 使用填充后的值
                dynamic_short_window_val=dynamic_short_window_val # 使用填充后的值
            )
        except KeyError as e:
            # 如果某个 key 的 format 字符串包含未定义的占位符，打印警告
            print(f"警告: 在格式化 HOVER_TEXTS['{key}'] 时缺少键: {e}")
        except Exception as e:
            print(f"警告: 格式化 HOVER_TEXTS['{key}'] 时发生错误: {e}")

    # 重新填充该特定 key
    try:
        HOVER_TEXTS['ema_crossover'] = HOVER_TEXTS['ema_crossover'] # 这里只是为了触发可能的 format，如果之前有占位符的话
    except Exception as e:
         print(f"警告: 格式化 HOVER_TEXTS['ema_crossover'] 时发生错误: {e}")

    report_html = f"""
    <div style="font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: auto; padding: 20px; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
        <h2 style="text-align: center; border-bottom: 1px solid #ccc; padding-bottom: 10px;">银价采购分析报告</h2>
        <p><strong>报告日期：</strong>{current['日期'].strftime('%Y-%m-%d')}</p>
        <p><strong title='{HOVER_TEXTS['price']}'>当前价格：</strong>{price:.2f} CNY</p>
        <p><strong title='{HOVER_TEXTS['indicator']}'>核心指标（工业指标）：</strong>{indicator:.2f} <span title='{HOVER_TEXTS['threshold']}'>（买入参考阈值：低于 {threshold:.2f}）</span></p>

        <h3 title='{HOVER_TEXTS['signal']}'>🛒 今日建议：{'<span style="color:green; font-weight:bold;">立即采购</span>' if current['采购信号'] else '<span style="color:orange; font-weight:bold;">持币观望</span>'}</h3>
        <p><em>（此建议基于以下综合分析，需至少满足4个核心条件且无阻断信号）</em></p>

        <h3>策略状态：</h3>
        <ul>
            <li title='{HOVER_TEXTS['dynamic_window']}'>动态窗口：短均线 {int(current.get('动态短窗口', BASE_WINDOW_SHORT))}天 / 长均线 {int(current.get('动态长窗口', BASE_WINDOW_LONG))}天</li>
            <li title='{HOVER_TEXTS['price_trend']}'>价格趋势：当前价格比短期均线 {'高' if price_trend_vs_sma > 0 else '低'} {abs(price_trend_vs_sma):.1f}%</li>
            <li title='{HOVER_TEXTS['volatility']}'>市场波动性（动量因子）：{volatility*100:.1f}%</li>
        </ul>

        <h3 title='{HOVER_TEXTS['ema_crossover']}'>📈 短期趋势信号 (EMA交叉)：</h3>
        <ul>
    """
    # --- 修改EMA交叉状态判断逻辑 --- 
    ema_crossover_status = "状态未知"
    ema_crossover_color = "gray"
    # 检查是否存在 'ema9_above_ema21' 列
    if 'ema9_above_ema21' in df.columns and not df.empty:
        # 确保该列是布尔类型
        if pd.api.types.is_bool_dtype(df['ema9_above_ema21']):
            current_ema9_above = current.get('ema9_above_ema21', None)
            if current_ema9_above is True:
                ema_crossover_status = "金叉状态 (EMA9 > EMA21，看涨倾向)"
                ema_crossover_color = "green"
            elif current_ema9_above is False:
                ema_crossover_status = "死叉状态 (EMA9 < EMA21，看跌倾向)"
                ema_crossover_color = "red"
            # 如果 current_ema9_above 是 None (例如因为计算失败或数据不足)
            # status 保持 "状态未知"
        else:
            print("警告: 'ema9_above_ema21' 列不是布尔类型，无法判断交叉状态。")
    else:
        print("警告: 缺少 'ema9_above_ema21' 列，无法判断交叉状态。")
    # --- 结束修改 --- 

    report_html += f'<li>当前状态：<strong style="color:{ema_crossover_color};">{ema_crossover_status}</strong></li>'
    report_html += "</ul>"

    # --- 定义六个核心买入条件的中文解释和当前状态 ---
    CONDITION_EXPLANATIONS = {
        'core': {
            'cond1': ("工业指标 < 阈值", f"{indicator:.2f} < {threshold:.2f}", HOVER_TEXTS['core_cond1']),
            'cond2': ("RSI < 45 (超卖区域)", f"RSI {rsi:.1f} < 45", HOVER_TEXTS['core_cond2']),
            'cond3': ("价格 < EMA21", f"价格 {price:.2f} < EMA21 {ema21:.2f}", HOVER_TEXTS['core_cond3']),
            'cond4': ("价格 < 布林下轨附近", f"价格 {price:.2f} < 下轨参考 {lower_band * 1.05:.2f}", HOVER_TEXTS['core_cond4']),
            'cond5': ("短期EMA动能 > 阈值", f"EMA比率 {ema_ratio:.3f} > 阈值 {dynamic_threshold:.3f}", HOVER_TEXTS['core_cond5']),
            'cond6': ("波动性 < 阈值 (市场平静)", f"波动 {volatility:.3f} < 阈值 {vol_threshold:.3f}", HOVER_TEXTS['core_cond6'])
        }
    }

    report_html += """
        <h3>🎯 触发条件分析（满足其中至少4项是买入的前提）：</h3>
        <p><strong>【核心条件验证】</strong></p>
        <ul style="list-style-type: none; padding-left: 0;">
    """
    for i in range(1, 7):
        col = f'core_cond{i}_met'
        is_met = current.get(col, False)
        desc = CONDITION_EXPLANATIONS['core'][f'cond{i}']
        status_icon = "✔️" if is_met else "❌"
        status_color = "green" if is_met else "red"
        # 简化 title 属性的引号，确保HTML有效
        title_attr = desc[2].replace('"', '&quot;') # 转义双引号
        report_html += f'<li style="margin-bottom: 5px;" title="{title_attr}"><span style="color: {status_color}; margin-right: 5px;">{status_icon}</span> {i}. {desc[0]}：{desc[1]}</li>'
    report_html += "</ul>"

    report_html += "<h3>🔍 信号阻断分析（即使满足4个以上条件，以下情况也会阻止买入）：</h3><ul>"

    condition_scores = sum([current.get(f'core_cond{i}_met', False) for i in range(1, 7)])
    base_req_met = condition_scores >= 4
    # 简化 title 属性的引号
    report_html += f"<li title='{HOVER_TEXTS['cond_score'].replace('\"','&quot;')}'>核心条件满足数量：{condition_scores}/6 ({'<span style=\"color:green;\">达标 (≥4)</span>' if base_req_met else '<span style=\"color:red;\">未达标 (<4)</span>'})</li>"

    peak_filter_series = peak_filter(df)
    peak_filter_passed = peak_filter_series.iloc[-1] if isinstance(peak_filter_series, pd.Series) else True
    peak_status_text = '<span style="color:green;">未触发阻断</span>' if peak_filter_passed else '<span style="color:red;">触发阻断</span>'
    atr_denominator = atr_upper - atr_lower
    atr_value = ((price - atr_lower) / atr_denominator) * 100 if atr_denominator != 0 else 50.0
    atr_overbought = atr_value > 80
    atr_status_text = '<span style="color:red;">超买区域 (>80%)</span>' if atr_overbought else f'{atr_value:.1f}%'
    # 简化 title 属性的引号
    report_html += f"<li title='{HOVER_TEXTS['peak_filter'].replace('\"','&quot;')}'>价格形态/ATR过滤：形态 {peak_status_text} | ATR通道位置 {atr_status_text}</li>"

    last_signal_index = df[df['采购信号']].index[-1] if df['采购信号'].any() else -1
    interval_days = len(df) - 1 - last_signal_index if last_signal_index != -1 else 999
    interval_ok = interval_days >= MIN_PURCHASE_INTERVAL
    interval_check_text = '<span style="color:green;">满足</span>' if interval_ok else f'<span style="color:orange;">不满足 (还需等待 {MIN_PURCHASE_INTERVAL - interval_days}天)</span>'
    # 简化 title 属性的引号
    report_html += f"<li title='{HOVER_TEXTS['interval'].replace('\"','&quot;')}'>采购间隔：距离上次已 {interval_days}天 (要求≥{MIN_PURCHASE_INTERVAL}天) → {interval_check_text}</li>"

    window_effect = BASE_WINDOW_SHORT - int(current.get('动态短窗口', BASE_WINDOW_SHORT))
    # 简化 title 属性的引号
    report_html += f"<li title='{HOVER_TEXTS['window_decay'].replace('\"','&quot;')}'>窗口衰减效果：当前短窗口比基准小 {window_effect}天 (基准{BASE_WINDOW_SHORT} → 当前{int(current.get('动态短窗口', BASE_WINDOW_SHORT))})</li>" # 确保是整数

    ema_trend_val = current.get('EMA趋势', 0)
    ema_trend_text = '<span style="color:green;">多头</span>' if ema_trend_val == 1 else '<span style="color:red;">空头</span>' if ema_trend_val == -1 else "震荡"
    # 简化 title 属性的引号
    report_html += f"<li title='{HOVER_TEXTS['ema_trend'].replace('\"','&quot;')}'>EMA趋势状态：{ema_trend_text}</li>"

    report_html += "</ul>"

    if current['采购信号']:
        report_html += "<h3>✅ 综合评估：<span style='color:green;'>满足买入条件，无阻断信号。</span></h3>"
    else:
        block_reasons = []
        if not base_req_met: block_reasons.append("核心条件不足 (未满足≥4项)")
        if not interval_ok: block_reasons.append(f"采购间隔限制 (还需{max(0, MIN_PURCHASE_INTERVAL - interval_days)}天)") # 确保不显示负数
        if not peak_filter_passed: block_reasons.append("价格形态不利")
        if atr_overbought: block_reasons.append("ATR通道超买 (>80%)")
        reason_str = ' + '.join(block_reasons) if block_reasons else '核心条件未完全满足或其它因素'
        # 简化 title 属性的引号
        report_html += f"<h3 title='{HOVER_TEXTS['final_block'].replace('\"','&quot;')}'>⛔ 最终阻断原因：<span style='color:red;'>{reason_str}</span></h3>"

    current_idx = df.index[-1]
    three_day_ago_idx = current_idx - 3
    if three_day_ago_idx >= 0:
        three_day_ago_date_obj = df['日期'].iloc[three_day_ago_idx]
        three_day_ago_date = three_day_ago_date_obj.strftime('%Y-%m-%d') if pd.notna(three_day_ago_date_obj) else "N/A"
        three_day_ago_price = safe_float(df['Price'].iloc[three_day_ago_idx])
        three_day_diff = price - three_day_ago_price
        # 简化 title 属性的引号
        report_html += f"""
        <h3 title='{HOVER_TEXTS['3day_change'].replace('\"','&quot;')}'>📉 三日价格变化参考：</h3>
        <ul>
            <li>三日前 ({three_day_ago_date}) 价格：{three_day_ago_price:.2f}</li>
            <li>三日价格变动：{'<span style="color:green;">+' if three_day_diff >= 0 else '<span style="color:red;">'}{three_day_diff:.2f}</span></li>
        </ul>"""
    else:
         report_html += "<h3>📉 三日价格变化参考：数据不足</h3>"

    report_html += "</div>" # Close main div

    # --- 计算用于动态分析的数据 --- 
    condition_scores = sum([current.get(f'core_cond{i}_met', False) for i in range(1, 7)])
    base_req_met = condition_scores >= 4
    peak_filter_series = peak_filter(df)
    peak_filter_passed = peak_filter_series.iloc[-1] if isinstance(peak_filter_series, pd.Series) else True
    peak_status_text = '<span style="color:green;">未触发阻断</span>' if peak_filter_passed else '<span style="color:red;">触发阻断</span>'
    atr_denominator = atr_upper - atr_lower
    atr_value = ((price - atr_lower) / atr_denominator) * 100 if atr_denominator != 0 else 50.0
    atr_overbought = atr_value > 80
    # (peak_status_text 现在只反映形态过滤，需要组合ATR状态)
    if not peak_filter_passed:
        peak_status_display = '<span style=\"color:red;\">形态不利</span>'
    elif atr_overbought:
        peak_status_display = '<span style=\"color:red;\">ATR超买({atr_value:.1f}%)</span>'
    else:
        peak_status_display = '<span style=\"color:green;\">通过</span>'

    last_signal_index = df[df['采购信号']].index[-1] if df['采购信号'].any() else -1
    interval_days = len(df) - 1 - last_signal_index if last_signal_index != -1 else 999
    interval_ok = interval_days >= MIN_PURCHASE_INTERVAL
    interval_check_text = '<span style="color:green;">满足</span>' if interval_ok else f'<span style="color:orange;">不满足 (还需等待 {max(0, MIN_PURCHASE_INTERVAL - interval_days)}天)</span>'

    block_reasons = []
    if not base_req_met: block_reasons.append(f"核心条件不足({condition_scores}/6)")
    if not interval_ok: block_reasons.append(f"采购间隔限制(还需{max(0, MIN_PURCHASE_INTERVAL - interval_days)}天)")
    if not peak_filter_passed: block_reasons.append("价格形态不利")
    if atr_overbought: block_reasons.append(f"ATR通道超买({atr_value:.1f}%)")

    # --- 准备用于动态分析的数据 --- 
    indicator_threshold_diff = threshold - indicator # 正数表示低于阈值
    rsi_oversold_diff = 45 - rsi # 正数表示低于45

    # --- 为指标差距添加更详细的定性描述 ---
    indicator_diff_desc = ""
    if indicator_threshold_diff > 0.1:
        indicator_diff_desc = f"显著低于阈值 ({indicator:.2f} vs {threshold:.2f})"
    elif indicator_threshold_diff > 0:
         indicator_diff_desc = f"低于阈值 ({indicator:.2f} vs {threshold:.2f})"
    elif indicator_threshold_diff == 0:
         indicator_diff_desc = f"恰好达到阈值 ({indicator:.2f})"
    elif indicator_threshold_diff > -0.05: # 在阈值上方，但差距小于 0.05
         indicator_diff_desc = f"略高于阈值 ({indicator:.2f} vs {threshold:.2f}，差距{abs(indicator_threshold_diff):.2f})"
    else: # 在阈值上方，差距大于等于 0.05
        indicator_diff_desc = f"仍高于阈值 ({indicator:.2f} vs {threshold:.2f}，差距{abs(indicator_threshold_diff):.2f})"
        
    rsi_diff_desc = ""
    if rsi_oversold_diff > 10: # RSI < 35
        rsi_diff_desc = f"深入超卖区 ({rsi:.1f} vs 45)"
    elif rsi_oversold_diff > 5: # 35 <= RSI < 40
        rsi_diff_desc = f"位于超卖区 ({rsi:.1f} vs 45)"
    elif rsi_oversold_diff > 0: # 40 <= RSI < 45
        rsi_diff_desc = f"接近超卖区 ({rsi:.1f} vs 45)"
    elif rsi_oversold_diff == 0: # RSI = 45
        rsi_diff_desc = f"恰好在超卖线 ({rsi:.1f})"
    elif rsi_oversold_diff > -5: # 45 < RSI <= 50
         rsi_diff_desc = f"略高于超卖线 ({rsi:.1f} vs 45)"
    else: # RSI > 50
        rsi_diff_desc = f"远离超卖区 ({rsi:.1f} vs 45)"

    signal_strength = "" # 初始化信号强度描述
    if current['采购信号']:
        if condition_scores == 6:
            signal_strength = "强信号 (所有条件满足)"
        elif condition_scores == 5:
            signal_strength = "明确信号 (多数条件满足)"
        else: # condition_scores == 4
            signal_strength = "边缘信号 (勉强满足条件)"
    
    # ... (peak_status_display, interval_check_text, block_reasons, current_conditions_met 的计算保持不变) ...
    peak_filter_series = peak_filter(df)
    peak_filter_passed = peak_filter_series.iloc[-1] if isinstance(peak_filter_series, pd.Series) else True
    atr_denominator = atr_upper - atr_lower
    atr_value = ((price - atr_lower) / atr_denominator) * 100 if atr_denominator != 0 else 50.0
    atr_overbought = atr_value > 80
    if not peak_filter_passed:
        peak_status_display = '<span style="color:red;">形态不利</span>'
    elif atr_overbought:
        peak_status_display = f'<span style="color:red;">ATR超买({atr_value:.1f}%)</span>'
    else:
        peak_status_display = '<span style="color:green;">通过</span>'

    last_signal_index = df[df['采购信号']].index[-1] if df['采购信号'].any() else -1
    interval_days = len(df) - 1 - last_signal_index if last_signal_index != -1 else 999
    interval_ok = interval_days >= MIN_PURCHASE_INTERVAL
    interval_check_text = '<span style="color:green;">满足</span>' if interval_ok else f'<span style="color:orange;">不满足 (还需等待 {max(0, MIN_PURCHASE_INTERVAL - interval_days)}天)</span>'

    base_req_met = condition_scores >= 4 # 这个要在 block_reasons 之前计算
    block_reasons = []
    # 注意：不再将"核心条件不足"加入 block_reasons，因为它会在结论中单独处理
    # if not base_req_met: block_reasons.append(f"核心条件不足({condition_scores}/6)") 
    if not interval_ok: block_reasons.append(f"采购间隔限制(还需{max(0, MIN_PURCHASE_INTERVAL - interval_days)}天)")
    if not peak_filter_passed: block_reasons.append("价格形态不利")
    if atr_overbought: block_reasons.append(f"ATR通道超买({atr_value:.1f}%)")

    current_conditions_met = {f'cond{i}': current.get(f'core_cond{i}_met', False) for i in range(1, 7)}

    # 准备最终的 analysis_data 字典，加入新的描述字段
    analysis_data = {
        'current_date': current['日期'],
        'signal': current['采购信号'],
        'signal_strength': signal_strength, 
        'condition_scores': condition_scores,
        'current_conditions_met': current_conditions_met,
        'indicator': indicator,
        'threshold': threshold,
        'indicator_threshold_diff': indicator_threshold_diff, 
        'indicator_diff_desc': indicator_diff_desc, # 新增
        'rsi': rsi,
        'rsi_oversold_diff': rsi_oversold_diff, 
        'rsi_diff_desc': rsi_diff_desc, # 新增
        'price': price,
        'ema21': ema21,
        'lower_band_ref': lower_band * 1.05, 
        'ema_ratio': ema_ratio,
        'dynamic_ema_threshold': dynamic_threshold,
        'volatility': volatility,
        'vol_threshold': vol_threshold,
        'peak_status_display': peak_status_display,
        'interval_days': interval_days,
        'interval_check_text': interval_check_text,
        'min_purchase_interval': MIN_PURCHASE_INTERVAL,
        'base_req_met': base_req_met,
        'block_reasons': block_reasons, # 现在只包含明确的阻断原因
    }

    # 返回包含报告内容和增强后分析数据的字典
    return {
        'report_content': report_html, 
        'analysis_data': analysis_data 
    }

def create_visualization(df):
    """
    使用 Plotly 生成交互式 HTML 图表，包含三个子图，帮助可视化分析。
    新增功能：鼠标悬停在图表线上时，会显示该线的名称、数值以及简要计算说明。
    新增功能：在价格图上标记 EMA 金叉 (↑) 和死叉 (↓)。
    图表解读指南... (保持不变)
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(
                            # 修改标题以反映新标记
                            '价格与信号 (看红色三角/金叉绿色↑/死叉红色↓)', 
                            '策略指标分析 (看蓝色线是否低于红色虚线/进入绿色区域)',
                            '动量指标分析 (看紫色线是否低于红色点线)'
                        ))

    # --- 定义悬停模板 ---
    hovertemplate_price = "<b>价格</b>: %{y:.2f} CNY<br>日期: %{x|%Y-%m-%d}<br><i>来源: 每日收盘价</i><extra></extra>"
    hovertemplate_sma = "<b>%{data.name}</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 最近%{customdata}天收盘价的算术平均</i><extra></extra>"
    hovertemplate_ema = "<b>%{data.name}</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 指数移动平均，近期价格权重更高</i><extra></extra>"
    hovertemplate_signal = "<b>⭐采购信号⭐</b><br>价格: %{y:.2f} CNY<br>日期: %{x|%Y-%m-%d}<br><i>策略建议买入点</i><extra></extra>"
    hovertemplate_indicator = "<b>核心工业指标</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: (价/短均)*(价/长均)*(1-动量)</i><extra></extra>"
    hovertemplate_threshold = "<b>%{data.name}</b>: %{y:.2f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 近期工业指标的25%分位数</i><extra></extra>"
    hovertemplate_rsi = "<b>修正RSI</b>: %{y:.1f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 基于14日平均涨跌幅，衡量超买超卖</i><extra></extra>"
    hovertemplate_rsi_threshold = "<b>动态RSI阈值</b>: %{y:.1f}<br>日期: %{x|%Y-%m-%d}<br><i>计算: 近63日RSI的30%分位数</i><extra></extra>"
    hovertemplate_fill = "<b>指标低于阈值区域</b><br>日期: %{x|%Y-%m-%d}<br>工业指标: %{y:.2f}<br><i>满足买入条件1</i><extra></extra>"
    # EMA 交叉的悬停文本将在 annotations 中定义

    # --- 行 1: 价格与信号 --- 
    # 移除 legendgroup 使其可单独隐藏
    fig.add_trace(go.Scatter(x=df['日期'], y=df['Price'], mode='lines', name='白银价格 (CNY)',
                             line=dict(color='navy', width=1.5), # legendgroup='price', legendrank=1,
                             hovertemplate=hovertemplate_price),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['SMA动态短'], mode='lines', name='短期均线 (近期趋势)',
                             line=dict(color='darkorange', dash='dash'), # legendgroup='price', legendrank=2,
                             customdata=df['动态短窗口'],
                             hovertemplate=hovertemplate_sma),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['EMA9'], mode='lines', name='EMA9 (更短趋势)',
                             line=dict(color='firebrick', width=1), # legendgroup='price', legendrank=3, 
                             opacity=0.7,
                             hovertemplate=hovertemplate_ema),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['EMA21'], mode='lines', name='EMA21 (中期趋势)',
                             line=dict(color='seagreen', width=1), # legendgroup='price', legendrank=4, 
                             opacity=0.7,
                             hovertemplate=hovertemplate_ema),
                  row=1, col=1)

    # --- 使用 Annotations 添加 EMA 交叉标记 (修改检测逻辑) --- 
    # 使用新的 'ema9_above_ema21' 列来检测视觉交叉
    if 'ema9_above_ema21' in df.columns and pd.api.types.is_bool_dtype(df['ema9_above_ema21']) and len(df) > 1:
        # 检测 ema9_above_ema21 状态的变化
        # --- 修改：先转换为整数再求差分 ---
        cross_change = df['ema9_above_ema21'].astype(int).diff()
        # --- 结束修改 ---
        # diff == 1 表示从 False 变为 True (视觉金叉)
        golden_cross_points = df[(cross_change == 1)]
        # diff == -1 表示从 True 变为 False (视觉死叉)
        death_cross_points = df[(cross_change == -1)]

        # --- 移除调试打印 ---
        # print(f"检测到的视觉金叉点数量: {len(golden_cross_points)}")
        # print(f"检测到的视觉死叉点数量: {len(death_cross_points)}")
        # --- 结束移除 ---

        # 计算一个小的偏移量，让箭头稍微离开价格线
        # 使用 Y 轴范围的一个小比例作为偏移量，避免绝对值过大或过小
        y_range = df['Price'].max() - df['Price'].min()
        # --- 修改：增大偏移量 ---
        offset = y_range * 0.035 # Y轴范围的 3.5% 作为偏移
        # --- 结束修改 ---

        # --- 绘制金叉标记 --- 
        for i in range(len(golden_cross_points)):
            point = golden_cross_points.iloc[i]
            fig.add_annotation(
                x=point['日期'],
                y=point['Price'] - offset, # 放在价格下方
                # --- 修改：加粗箭头，增大字号 ---
                text="<b>↑</b>",
                showarrow=False,
                font=dict(size=16, color="green"),
                # --- 结束修改 ---
                # 更新悬停文本，明确是视觉交叉
                hovertext=f"<b>📈 EMA视觉金叉</b><br>日期: {point['日期']:%Y-%m-%d}<br>价格: {point['Price']:.2f}",
                hoverlabel=dict(bgcolor="white"),
                yanchor="top"
            )

        # --- 绘制死叉标记 --- 
        for i in range(len(death_cross_points)):
            point = death_cross_points.iloc[i]
            fig.add_annotation(
                x=point['日期'],
                y=point['Price'] + offset, # 放在价格上方
                 # --- 修改：加粗箭头，增大字号 ---
                text="<b>↓</b>",
                showarrow=False,
                font=dict(size=16, color="red"),
                # --- 结束修改 ---
                # 更新悬停文本，明确是视觉交叉
                hovertext=f"<b>📉 EMA视觉死叉</b><br>日期: {point['日期']:%Y-%m-%d}<br>价格: {point['Price']:.2f}",
                hoverlabel=dict(bgcolor="white"),
                yanchor="bottom"
            )

        # 添加一个不可见的散点轨迹用于图例显示 (Annotations 不会自动加入图例)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], # 没有实际数据点
            mode='markers', 
            marker=dict(color='green', symbol='triangle-up', size=8), # 用三角代替箭头显示
            name='📈 EMA视觉金叉事件'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], 
            mode='markers', 
            marker=dict(color='red', symbol='triangle-down', size=8),
            name='📉 EMA视觉死叉事件'
        ), row=1, col=1)

    # --- 保留原始采购信号标记 --- 
    # 同样移除 legendgroup
    signal_df = df[df['采购信号']]
    if not signal_df.empty:
        fig.add_trace(go.Scatter(x=signal_df['日期'], y=signal_df['Price'], mode='markers', name='⭐采购信号⭐',
                                 marker=dict(color='red', size=8, symbol='triangle-up', line=dict(width=1, color='black')),
                                 # legendgroup='signal', legendrank=5, 
                                 hovertemplate=hovertemplate_signal),
                      row=1, col=1)

    # --- 行 2: 策略指标分析 --- 
    # 移除 legendgroup
    fig.add_trace(go.Scatter(x=df['日期'], y=df['工业指标'], mode='lines', name='核心工业指标',
                             line=dict(color='royalblue'), # legendgroup='indicator', legendrank=8,
                             hovertemplate=hovertemplate_indicator),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['基线阈值_短'], mode='lines', name=f'短期阈值 ({HISTORY_WINDOW_SHORT}日)',
                             line=dict(color='darkorange', dash='dot', width=1), # legendgroup='indicator', legendrank=9, 
                             opacity=0.7,
                             hovertemplate=hovertemplate_threshold),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['基线阈值'], mode='lines', name=f'中期阈值 ({HISTORY_WINDOW}日) - 警戒线',
                             line=dict(color='crimson', dash='dash', width=1.5), # legendgroup='indicator', legendrank=10,
                             hovertemplate=hovertemplate_threshold),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['基线阈值_长'], mode='lines', name=f'长期阈值 ({HISTORY_WINDOW_LONG}日)',
                             line=dict(color='purple', dash='dashdot', width=1), # legendgroup='indicator', legendrank=11, 
                             opacity=0.8,
                             hovertemplate=hovertemplate_threshold),
                  row=2, col=1)
    
    # 填充区域逻辑保持不变，但填充区域本身不加图例或给个泛指名字
    y_upper = df['基线阈值']
    y_lower = df['工业指标']
    y_fill_lower = y_upper.copy()
    fill_mask = y_lower < y_upper
    y_fill_lower[fill_mask] = y_lower[fill_mask]
    # 上边界（透明）
    fig.add_trace(go.Scatter(x=df['日期'], y=y_upper, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False, hoverinfo='skip'), row=2, col=1)
    # 填充轨迹
    fig.add_trace(go.Scatter(x=df['日期'], y=y_fill_lower,
                             fill='tonexty',
                             mode='lines',
                             line=dict(width=0),
                             fillcolor='rgba(144, 238, 144, 0.3)',
                             name='指标<阈值区域', # 简洁图例名
                             # legendgroup='indicator', legendrank=12, 
                             hovertemplate=hovertemplate_fill
                             ), row=2, col=1)
    
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="指标参考基准=1", row=2, col=1)


    # --- 行 3: 动量指标分析 --- 
    # 移除 legendgroup
    fig.add_trace(go.Scatter(x=df['日期'], y=df['修正RSI'], mode='lines', name='修正RSI (市场强弱)',
                             line=dict(color='darkviolet'), # legendgroup='momentum', legendrank=13,
                             hovertemplate=hovertemplate_rsi),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['RSI阈值'], mode='lines', name='动态RSI阈值',
                             line=dict(color='darkorange', dash='dash'), # legendgroup='momentum', legendrank=14,
                             hovertemplate=hovertemplate_rsi_threshold),
                  row=3, col=1)
    fig.add_hline(y=45, line_dash="dot", line_color="red", opacity=0.5, annotation_text="RSI超卖参考线=45 (买入条件2)", row=3, col=1, name="RSI 45")

    # --- 更新整体布局 --- 
    fig.update_layout(
        height=900,
        title_text='银价分析与策略可视化 (交互式图表)',
        hovermode='x unified',
        legend_title_text='图例说明 (点击可隐藏/显示)', # 更新图例标题
        margin=dict(l=60, r=60, t=100, b=60),
        # 移除 legend traceorder 或设置为 'normal' 让其按添加顺序显示
        # legend=dict(traceorder='reversed+grouped') 
        legend=dict(traceorder='normal')
    )
    fig.update_yaxes(title_text="价格 (CNY)", row=1, col=1)
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

    # 3. 生成报告数据
    print("正在生成报告数据...")
    # 修改：接收包含内容和分析数据的字典
    report_data = generate_report(df)
    report_html_content = report_data['report_content']
    analysis_data = report_data['analysis_data'] # 提取分析数据

    # 4. 生成图表 Figure 对象
    print("正在生成图表...")
    fig = create_visualization(df)

    # 5. 将图表转换为 HTML div (改回使用 CDN)
    # 使用 include_plotlyjs='cdn' 使 HTML 文件更小，依赖网络加载 JS
    # 使用 full_html=False 只获取图表的 div 部分
    try:
        # --- 修改下面这行 ---
        chart_html_div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        # --- 结束修改 ---
        if not chart_html_div or len(chart_html_div.strip()) == 0:
             print("警告：生成的图表 HTML 为空。")
             chart_html_div = "<p style='color:orange;'>图表生成似乎为空。</p>"
    except Exception as e:
        print(f"错误：将 Plotly 图表转换为 HTML 时失败: {e}")
        chart_html_div = "<p style='color:red;'>图表生成失败。</p>"

    # 6. 构建完整的 HTML 页面
    
    # --- 6.1 预先构建动态"今日解读"部分的 HTML --- 
    today_interpretation_html = f'''
        <h3 style="background-color: #f0f0f0; padding: 10px; border-left: 5px solid #007bff;">💡 对今天 ({analysis_data['current_date'].strftime('%Y-%m-%d')}) 的策略信号解读：</h3>
        <p><strong>今日策略建议：{'<span style="color:green; font-weight:bold;">建议采购 ({})</span>'.format(analysis_data['signal_strength']) if analysis_data['signal'] else '<span style="color:orange; font-weight:bold;">建议持币观望</span>'}</strong></p>
        <p><strong>分析概要：</strong></p>
        <ul>
            <li>核心条件满足数量：<strong>{analysis_data['condition_scores']} / 6</strong> (策略要求至少满足 4 项)。</li>
            <li>信号阻断检查：{analysis_data['peak_status_display']} 且 {analysis_data['interval_check_text']}。</li>
    '''

    if analysis_data['signal']:
        today_interpretation_html += f'''<li>关键指标状态：
                <ul>
                    <li>核心工业指标: {analysis_data['indicator_diff_desc']}。</li>
                    <li>市场动量 (RSI): {analysis_data['rsi_diff_desc']}。</li>
                    {'<li>其余 {} 项辅助条件也满足要求。</li>'.format(analysis_data['condition_scores'] - 2) if analysis_data['condition_scores'] > 2 else ''}
                </ul>
            </li>
            <li><strong>结论：</strong><span style="color:green;">由于关键买入指标进入策略目标区域，满足了 {analysis_data['condition_scores']} 项核心条件，并且无明确的信号阻断因素，策略判定当前形成 <strong>{analysis_data['signal_strength']}</strong> 的采购信号。</span></li>
        '''
    else: # 如果是观望
        # 构建未满足条件的列表
        unmet_conditions_list = ''
        if not analysis_data['current_conditions_met']['cond1']:
            unmet_conditions_list += f'<li>核心工业指标: {analysis_data["indicator_diff_desc"]}.</li>'
        if not analysis_data['current_conditions_met']['cond2']:
             unmet_conditions_list += f'<li>市场动量 (RSI): {analysis_data["rsi_diff_desc"]}.</li>'
        if not analysis_data['current_conditions_met']['cond3']:
            unmet_conditions_list += f'<li>价格({analysis_data["price"]:.2f}) 未低于 EMA21({analysis_data["ema21"]:.2f}).</li>'
        if not analysis_data['current_conditions_met']['cond4']:
            unmet_conditions_list += f'<li>价格({analysis_data["price"]:.2f}) 未低于布林下轨参考({analysis_data["lower_band_ref"]:.2f}).</li>'
        if not analysis_data['current_conditions_met']['cond5']:
            unmet_conditions_list += f'<li>EMA比率({analysis_data["ema_ratio"]:.3f}) 未达动态阈值({analysis_data["dynamic_ema_threshold"]:.3f}).</li>'
        if not analysis_data['current_conditions_met']['cond6']:
            unmet_conditions_list += f'<li>波动性({analysis_data["volatility"]:.3f}) 高于动态阈值({analysis_data["vol_threshold"]:.3f}).</li>'
        
        if not unmet_conditions_list: # 如果所有条件都满足但仍然观望，说明是阻断
             unmet_conditions_list = "<li>所有核心条件均满足，观望是由于信号阻断规则。</li>"
             
        today_interpretation_html += f'<li>当前未能满足买入要求的主要条件：<ul>{unmet_conditions_list}</ul></li>'
        
        # 构建结论文本
        blocking_issues = analysis_data['block_reasons'] # 现在只包含明确阻断原因
        conclusion_text = ''
        if blocking_issues:
            conclusion_text = '信号因以下规则被阻断：' + '； '.join(blocking_issues) + '。'
        elif not analysis_data['base_req_met']:
             conclusion_text = f"由于仅满足 {analysis_data['condition_scores']}/6 项核心条件，未能达到策略要求的最低数量。"
        else: 
            conclusion_text = f"虽满足 {analysis_data['condition_scores']}/6 项核心条件，但可能存在其他未明确的阻断因素。"
            
        today_interpretation_html += f'<li><strong>结论：</strong><span style="color:red;">{conclusion_text} 因此，策略建议暂时持币观望。</span></li>'

    today_interpretation_html += '</ul>' # 闭合原因分析的 <ul>
    # --- 6.1 结束预构建 --- 
    
    # --- 6.2 构建最终 HTML，插入预构建的部分 --- 
    final_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>银价分析报告</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        .container {{ max-width: 900px; margin: auto; }}
        .report-content {{ margin-bottom: 30px; padding: 15px; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
        .chart-container {{ border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); padding: 10px; }}
        h2, h3 {{ color: #333; }}
        [title] {{ border-bottom: 1px dotted #bbb; cursor: help; position: relative; }}
        [title]:hover::after {{ content: attr(title); position: absolute; left: 50%; transform: translateX(-50%); bottom: 125%; white-space: pre-wrap; background-color: rgba(0, 0, 0, 0.8); color: #fff; padding: 5px 10px; border-radius: 4px; font-size: 12px; z-index: 10; width: max-content; max-width: 250px; }}
        .legend text {{ font-size: 11px !important; }}
        .annotation-text {{ font-size: 10px !important; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>银价走势分析与定投参考报告</h1>
        <p>生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="report-content">
            <h2>📈 关键指标与最新信号</h2>
            {report_html_content}
        </div>

        <div class="chart-container">
             <h2>📊 交互式图表分析</h2>
             <p>将鼠标悬停在图表线上可查看详细数据和计算说明。您可以缩放和平移图表进行探索。</p>
            {chart_html_div}
        </div>

        <div class="report-content" style="margin-top: 30px;">
            <h2>📖 图表与策略逻辑解读</h2>

            <h3>图表元素解析</h3>
            <p>以下是对图表中主要线条和标记的解释：</p>
            <ul>
                 <li><strong>上图 (价格与信号):</strong>
                    <ul>
                        <li><u>价格线 (深蓝)</u>: 代表每日的白银收盘价。这是所有分析的基础。</li>
                        <li><u>短期均线 (橙虚线)</u>: 计算指定周期内（例如{BASE_WINDOW_SHORT}天，根据策略动态调整）收盘价的算术平均值。它能平滑短期价格波动，帮助识别近期趋势方向。价格穿越均线常被视为趋势可能改变的信号。</li>
                        <li><u>EMA线 (红/绿细线)</u>: 指数移动平均线。与普通均线类似，但对更近期的价格赋予更高权重。这意味着EMA对价格变化的反应比普通均线更快，常用于捕捉更短期的趋势变化。</li>
                        <li><u>采购信号 (▲ 红三角)</u>: 当下方描述的所有策略买入条件均满足时，此标记出现。</li>
                    </ul>
                </li>
                <li><strong>中图 (策略核心指标):</strong>
                    <ul>
                        <li><u>核心工业指标 (蓝色实线)</u>: 这是本策略定制的一个综合指标。其计算综合考虑了当前价格与其短期、长期移动平均线的偏离程度，并结合了近期市场波动性（通过"动量因子"衡量）。其核心思想是：当价格相对其历史均值偏低，且市场波动性不高时，该指标值会较低，策略倾向于认为此时潜在的买入价值可能更高。</li>
                        <li><u>阈值线 (红色虚线等)</u>: 这些是根据近期"核心工业指标"的历史分布动态计算出来的参考线（通常是某个分位数，如25%分位数）。它们代表了策略认为的"相对便宜"的区域边界。当蓝色指标线低于关键的红色阈值线时，满足了策略的一个主要入场条件。</li>
                    </ul>
                </li>
                <li><strong>下图 (市场动量指标 - RSI):</strong>
                    <ul>
                        <li><u>修正RSI (紫色实线)</u>: 相对强弱指数（Relative Strength Index）。它通过比较一定时期内（通常是14天）价格上涨日和下跌日的平均涨跌幅度，来衡量市场买卖双方的力量对比，反映市场的景气程度。RSI的值域在0-100之间。通常认为，当RSI低于某个阈值（如此策略中的45）时，市场可能处于"超卖"状态，即下跌可能过度，短期内价格有反弹的可能性；反之，高于某个阈值（如70或80）则可能表示"超买"。策略利用RSI的超卖信号作为另一个关键的入场条件。</li>
                    </ul>
                </li>
            </ul>
            <h3>策略信号生成逻辑</h3>
             <p>策略生成采购信号 (▲) 需同时满足两大类条件：</p>
            <ol>
                <li><strong>核心条件达标：</strong>综合考量核心工业指标、RSI、价格与均线/通道关系、市场波动性等多个维度，需达到预设的触发数量（当前为至少4项）。</li>
                <li><strong>无信号阻断：</strong>排除近期不利价格形态、ATR超买以及过于频繁的信号（需满足最小间隔天数，当前为{analysis_data['min_purchase_interval']}天）。</li>
            </ol>

            <!-- 插入预先构建好的今日解读 HTML -->
            {today_interpretation_html}
        </div>
    </div>
</body>
</html>
"""

    # 7. 将完整的 HTML 写入文件
    output_filename = "index.html" # 确认输出文件名是 index.html
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"成功将报告写入文件: {output_filename}")

        # 8. 自动执行 Git 命令推送到 GitHub
        print("尝试将更新推送到 GitHub...")
        try:
            # 定义 Git 命令 (Add 和 Commit)
            commit_message = f"Update report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            files_to_add = [output_filename, ".gitignore", __file__]
            add_cmd = ["git", "add"] + files_to_add
            commit_cmd = ["git", "commit", "-m", commit_message]
            push_cmd = ["git", "push", "origin", "master"] # 单独定义 Push 命令

            # --- 执行 Add --- 
            print(f"执行命令: {' '.join(add_cmd)}")
            add_result = subprocess.run(add_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
            if add_result.stdout: print(f"Git 输出:\n{add_result.stdout.strip()}")
            if add_result.stderr: print(f"Git 错误:\n{add_result.stderr.strip()}")
            if add_result.returncode != 0:
                print(f"Git add 命令执行失败，返回码: {add_result.returncode}。停止。")
            else:
                # --- 执行 Commit --- 
                print(f"执行命令: {' '.join(commit_cmd)}")
                commit_result = subprocess.run(commit_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
                commit_success = False # 标记 Commit 是否成功
                if commit_result.stdout: print(f"Git 输出:\n{commit_result.stdout.strip()}")
                if commit_result.stderr:
                    if "nothing to commit" in commit_result.stderr:
                        print(f"Git 信息 (可忽略): {commit_result.stderr.strip()}")
                        commit_success = True # 没有东西提交也视为一种成功，可以尝试推送
                    else:
                         print(f"Git 错误:\n{commit_result.stderr.strip()}")
                # 检查返回码，如果为0，则认为成功
                if commit_result.returncode == 0:
                    commit_success = True

                if not commit_success:
                    print(f"Git commit 命令执行失败，返回码: {commit_result.returncode}。停止。")
                else:
                    # --- 执行 Push (无限重试直到成功) --- 
                    print(f"尝试推送: {' '.join(push_cmd)}")
                    while True:
                        push_result = subprocess.run(push_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
                        push_succeeded = False
                        push_stderr = push_result.stderr.strip() if push_result.stderr else ""

                        if push_result.stdout: print(f"Git 输出:\n{push_result.stdout.strip()}")
                        if push_stderr:
                            # Everything up-to-date 也视为成功
                            if "Everything up-to-date" in push_stderr or "up-to-date" in push_stderr:
                                print(f"Git 信息: {push_stderr}")
                                push_succeeded = True
                            else:
                                print(f"Git 错误:\n{push_stderr}")
                        
                        # 检查返回码是否为 0
                        if push_result.returncode == 0:
                            push_succeeded = True

                        if push_succeeded:
                            print("推送成功或无需推送。")
                            break # 跳出无限循环
                        else:
                            print(f"推送失败 (返回码: {push_result.returncode})，自动重试...")
                            # 无需 time.sleep，直接进入下一次循环尝试

        except FileNotFoundError:
            print("错误：找不到 'git' 命令。请确保 Git 已安装并添加到系统 PATH。")
        except Exception as git_e:
            print(f"错误：执行 Git 命令时出错: {git_e}")

    except Exception as e:
        print(f"错误：写入 HTML 文件失败: {e}")


    print("分析完成。")


