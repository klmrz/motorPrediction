import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import random
import time

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

# 生成带趋势的波动曲线（支持双向超出范围）
def generate_trend_series(base_value, min_limit, max_limit, steps, exceed_direction):
    """
    base_value: 基准值
    min_limit: 正常范围下限
    max_limit: 正常范围上限
    steps: 数据点数
    exceed_direction: 超出方向（1=向上超出，-1=向下超出）
    """
    # 基础波动（正弦波叠加随机噪声）
    x = np.linspace(0, 4 * np.pi, steps)
    base_series = base_value + 0.3 * (max_limit - min_limit) * np.sin(x) / 10

    # 添加随机波动
    random_noise = np.random.uniform(
        -0.1 * (max_limit - min_limit),
        0.1 * (max_limit - min_limit),
        steps
    )
    series = base_series + random_noise

    # 创建渐进趋势（最后10%数据点）
    trend_start = int(steps * 0.9)
    trend_length = steps - trend_start

    if exceed_direction == 1:  # 向上超出
        # 最终超过上限10%
        exceed_target = max_limit * 1.1
        trend_values = np.linspace(0, exceed_target - max_limit, trend_length)
        series[trend_start:] += trend_values
    else:  # 向下超出
        # 最终低于下限10%
        exceed_target = min_limit * 0.9
        trend_values = np.linspace(0, min_limit - exceed_target, trend_length)
        series[trend_start:] -= trend_values

    return series

# 模拟参数设置
days = 3
sampling_interval = 2  # 秒
points = 24 * 3600 * days // sampling_interval  # 129600

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

    # 基线设置：基线设置为正常运行范围的中间值
    base_amp = (range_row['final_max_amp'] + range_row['final_min_amp']) / 2
    base_rate = (range_row['final_max_rate'] + range_row['final_min_rate']) / 2
    base_temp = (range_row['final_max_temp'] + range_row['final_min_temp']) / 2
    base_vol = (range_row['final_max_vol'] + range_row['final_min_vol']) / 2

    # 生成趋势数据（各参数独立生成）
    # 电流：向上超出
    amps = generate_trend_series(
        base_amp,
        range_row['final_min_amp'],
        range_row['final_max_amp'],
        points,
        #exceed_direction=1 if random.random() > 0.5 else -1  # 随机选择超出方向
        exceed_direction=1
    )

    # 负载率：只向上超出
    rates = generate_trend_series(
        base_rate,
        range_row['final_min_rate'],
        range_row['final_max_rate'],
        points,
        exceed_direction=1
    )

    # 温度：只向上超出
    temps = generate_trend_series(
        base_temp,
        range_row['final_min_temp'],
        range_row['final_max_temp'],
        points,
        exceed_direction=1
    )

    # 电压：只向下超出
    vols = generate_trend_series(
        base_vol,
        range_row['final_min_vol'],
        range_row['final_max_vol'],
        points,
        exceed_direction=-1
    )

    # 构造数据，每2秒钟生成一条数据
    for j in range(points):
        # 获取当前参数值
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]

        # ===== 生成21位唯一ID =====
        # 16位微秒时间戳 + 2位电机ID + 3位随机数
        current_time = start_time + timedelta(seconds=j * sampling_interval)
        micro_timestamp = int(current_time.timestamp() * 1000000)  # 微秒时间戳
        motor_part = f"{i:02d}"  # 2位电机ID
        random_part = f"{random.randint(0, 999):03d}"  # 3位随机数
        f_id = f"{micro_timestamp}{motor_part}{random_part}"

        # 添加实时微波动（所有时间点）
        amp += random.uniform(-0.001, 0.001)
        rate += random.uniform(-0.1, 0.1)
        temp += random.uniform(-0.05, 0.05)
        vol += random.uniform(-0.005, 0.005)

        # ===== 异常标记 =====
        # 检查是否任一参数超出正常范围（高于最大值或低于最小值）
        is_abnormal = (
            (amp > range_row['final_max_amp']) or (amp < range_row['final_min_amp']) or  # 电流异常
            (rate > range_row['final_max_rate']) or (rate < range_row['final_min_rate']) or  # 负载率异常
            (temp > range_row['final_max_temp']) or (temp < range_row['final_min_temp']) or  # 温度异常
            (vol > range_row['final_max_vol']) or (vol < range_row['final_min_vol'])  # 电压异常
        )
        f_alarm = 1 if is_abnormal else 0

        # 确保正常数据不超出范围（仅在正常阶段）
        if f_alarm == 0:
            amp = np.clip(amp, range_row['final_min_amp'], range_row['final_max_amp'])
            rate = np.clip(rate, range_row['final_min_rate'], range_row['final_max_rate'])
            temp = np.clip(temp, range_row['final_min_temp'], range_row['final_max_temp'])
            vol = np.clip(vol, range_row['final_min_vol'], range_row['final_max_vol'])

        abnormal_data.append({
            'f_id': f_id,  # 21位唯一ID
            'f_device': motor_id,
            'f_err_code': str(0),  # 将 f_err_code 转为字符串类型
            'f_run_signal': 4,
            'f_time': current_time,
            'f_amp': round(amp, 3),
            'f_vol': round(vol, 3),
            'f_temp': round(temp, 3),
            'f_rate': int(round(rate)),  # 将 f_rate 转为整数类型
            'f_note': '模拟数据',
            'f_name': f_name,
            'f_alarm': f_alarm
        })

# 一次性插入所有数据到数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f" 插入完成：{len(df)} 条模拟数据，其中异常条数：{df[df.f_alarm == 1].shape[0]}")