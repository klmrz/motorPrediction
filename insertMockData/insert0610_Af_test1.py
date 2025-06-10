import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import random
import uuid

# 数据库连接
db_url = 'postgresql://postgres:dz123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 读取正常范围表
range_df = pd.read_csv("E:/Work/NewProject/motorPrediction/normalRange/min_max_ranges.csv")

# 从数据库中获取每台电机的最新数据时间
latest_time_query = '''
                    SELECT f_device, MAX(f_time) as latest_time
                    FROM dj_mock_data
                    GROUP BY f_device \
                    '''
latest_times = pd.read_sql(latest_time_query, engine)


# 生成带趋势的波动曲线（支持渐进超出范围且不回归）
def generate_stable_trend_series(base_value, min_limit, max_limit, steps, exceed_direction):
    """
    base_value: 基准值（正常范围中间值）
    min_limit: 正常范围下限
    max_limit: 正常范围上限
    steps: 数据点数
    exceed_direction: 超出方向（1=向上超出，-1=向下超出）
    """
    # 每分钟波动一次（30个数据点，采样间隔2秒）
    fluctuation_interval = 30  # 1分钟=60秒，60/2=30个点

    # 初始化为基础值（正常范围中间值）
    series = np.full(steps, base_value)

    # 正常阶段波动（前90%数据）
    normal_steps = int(steps * 0.9)
    for j in range(0, normal_steps, fluctuation_interval):
        # 波动幅度：正常范围跨度的±5%（例如电流范围10-20A，波动±0.5A）
        fluctuation_amplitude = 0.05 * (max_limit - min_limit)
        fluctuation = random.uniform(-fluctuation_amplitude, fluctuation_amplitude)

        # 应用波动到接下来的1分钟数据（30个点）
        end_idx = min(j + fluctuation_interval, normal_steps)
        series[j:end_idx] += fluctuation

    # 异常阶段趋势（最后10%数据，逐渐超出范围）
    trend_start = normal_steps
    trend_length = steps - trend_start

    if exceed_direction == 1:  # 向上超出（电流、负载率、温度）
        exceed_target = max_limit * 1.1  # 最终超出上限10%
        # 线性增长序列（从当前值到超出目标）
        trend = np.linspace(0, exceed_target - max_limit, trend_length)
        # 确保增长过程中不超出正常范围（仅在最后一步超出）
        series[trend_start:] = np.clip(series[trend_start - 1], min_limit, max_limit) + trend
    else:  # 向下超出（电压）
        exceed_target = min_limit * 0.9  # 最终低于下限10%
        # 线性减少序列（从当前值到低于目标）
        trend = np.linspace(0, min_limit - exceed_target, trend_length)
        series[trend_start:] = np.clip(series[trend_start - 1], min_limit, max_limit) - trend

    return series


# 模拟参数设置
days = 3
sampling_interval = 2  # 秒
points = 24 * 3600 * days // sampling_interval  # 129600（3天总数据点）

# 初始化空列表，用于存储生成的数据
abnormal_data = []

# 模拟电机编号1~7
for i in range(1, 8):
    motor_id = f"电机{i}"
    range_row = range_df[range_df['f_device'] == motor_id].iloc[0]
    f_name = range_row['f_device']

    # 获取该电机最新数据时间
    latest_time = latest_times[latest_times['f_device'] == motor_id]['latest_time'].values[0]
    start_time = pd.to_datetime(latest_time) + timedelta(minutes=5)  # 从最新数据的后5分钟开始

    # 基线设置：正常范围中间值
    base_amp = (range_row['final_max_amp'] + range_row['final_min_amp']) / 2
    base_rate = (range_row['final_max_rate'] + range_row['final_min_rate']) / 2
    base_temp = (range_row['final_max_temp'] + range_row['final_min_temp']) / 2
    base_vol = (range_row['final_max_vol'] + range_row['final_min_vol']) / 2

    # 生成各参数的波动序列（带趋势）
    amps = generate_stable_trend_series(
        base_amp,
        range_row['final_min_amp'],
        range_row['final_max_amp'],
        points,
        exceed_direction=1  # 电流向上超出
    )
    rates = generate_stable_trend_series(
        base_rate,
        range_row['final_min_rate'],
        range_row['final_max_rate'],
        points,
        exceed_direction=1  # 负载率向上超出
    )
    temps = generate_stable_trend_series(
        base_temp,
        range_row['final_min_temp'],
        range_row['final_max_temp'],
        points,
        exceed_direction=1  # 温度向上超出
    )
    vols = generate_stable_trend_series(
        base_vol,
        range_row['final_min_vol'],
        range_row['final_max_vol'],
        points,
        exceed_direction=-1  # 电压向下超出
    )

    # 构造数据，每2秒钟生成一条数据
    for j in range(points):
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]

        # 生成21位唯一ID（时间戳+电机ID+随机数）
        current_time = start_time + timedelta(seconds=j * sampling_interval)
        micro_timestamp = int(current_time.timestamp() * 1000000)  # 微秒时间戳
        motor_part = f"{i:02d}"  # 2位电机ID
        random_part = f"{random.randint(0, 999):03d}"  # 3位随机数
        f_id = f"{micro_timestamp}{motor_part}{random_part}"

        # 异常标记：检查是否超出正常范围
        is_abnormal = (
                (amp > range_row['final_max_amp']) or (amp < range_row['final_min_amp']) or
                (rate > range_row['final_max_rate']) or (rate < range_row['final_min_rate']) or
                (temp > range_row['final_max_temp']) or (temp < range_row['final_min_temp']) or
                (vol > range_row['final_max_vol']) or (vol < range_row['final_min_vol'])
        )
        f_alarm = 1 if is_abnormal else 0

        # 确保正常数据不超出范围（仅在正常阶段）
        if f_alarm == 0:
            amp = np.clip(amp, range_row['final_min_amp'], range_row['final_max_amp'])
            rate = np.clip(rate, range_row['final_min_rate'], range_row['final_max_rate'])
            temp = np.clip(temp, range_row['final_min_temp'], range_row['final_max_temp'])
            vol = np.clip(vol, range_row['final_min_vol'], range_row['final_max_vol'])

        abnormal_data.append({
            'f_id': f_id,
            'f_device': motor_id,
            'f_err_code': str(0),
            'f_run_signal': 4,
            'f_time': current_time,
            'f_amp': round(amp, 3),
            'f_vol': round(vol, 3),
            'f_temp': round(temp, 3),
            'f_rate': int(round(rate)),
            'f_note': '模拟数据',
            'f_name': f_name,
            'f_alarm': f_alarm
        })

# 插入数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f"插入完成：{len(df)} 条模拟数据，其中异常条数：{df[df.f_alarm == 1].shape[0]}")