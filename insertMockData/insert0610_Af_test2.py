import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import random

# ------------------ 1. 数据库连接 ------------------
db_url = 'postgresql://postgres:dz123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# ------------------ 2. 读取每台电机的正常范围 ------------------
range_df = pd.read_csv("E:/Work/NewProject/motorPrediction/normalRange/min_max_ranges.csv")

# ------------------ 3. 查询每台电机最新时间 ------------------
latest_times = pd.read_sql('''
    SELECT f_device, MAX(f_time) as latest_time
    FROM dj_mock_data
    GROUP BY f_device
''', engine)

# ------------------ 4. 波动+异常趋势序列生成函数 ------------------
def generate_stable_trend_series(base_value, min_limit, max_limit, steps, exceed_direction):
    """
    生成稳定波动 + 逐步异常的数据序列
    base_value: 中间基准值
    min_limit / max_limit: 正常上下限
    exceed_direction: 1表示上超（电流等），-1表示下超（电压）
    """
    series = np.full(steps, base_value)
    fluct_interval = 30  # 每分钟波动一次

    # 前90%的点用于正常波动
    normal_steps = int(steps * 0.9)
    for j in range(0, normal_steps, fluct_interval):
        fluctuation = random.uniform(-0.05 * (max_limit - min_limit), 0.05 * (max_limit - min_limit))
        end_idx = min(j + fluct_interval, normal_steps)
        series[j:end_idx] += fluctuation
        # 保证波动不超出正常范围
        series[j:end_idx] = np.clip(series[j:end_idx], min_limit, max_limit)

    # 后10%为异常区间，线性趋势上升/下降
    trend_start = normal_steps
    trend_length = steps - trend_start
    if exceed_direction == 1:
        target = max_limit * 1.1
        trend = np.linspace(0, target - max_limit, trend_length)
        series[trend_start:] = series[trend_start - 1] + trend
    else:
        target = min_limit * 0.9
        trend = np.linspace(0, min_limit - target, trend_length)
        series[trend_start:] = series[trend_start - 1] - trend

    return series

# ------------------ 5. 参数初始化 ------------------
days = 3
sampling_interval = 2  # 每2秒采样一次
points = 24 * 3600 * days // sampling_interval
abnormal_data = []

# ------------------ 6. 每台电机逐个生成数据 ------------------
for i in range(1, 8):
    motor_id = f"电机{i}"
    row = range_df[range_df['f_device'] == motor_id].iloc[0]
    latest_time = pd.to_datetime(latest_times[latest_times['f_device'] == motor_id]['latest_time'].values[0])
    start_time = latest_time + timedelta(minutes=5)

    # ------------------ 各参数基准值 ------------------
    base_amp = (row['final_max_amp'] + row['final_min_amp']) / 2
    base_rate = (row['final_max_rate'] + row['final_min_rate']) / 2
    base_temp = (row['final_max_temp'] + row['final_min_temp']) / 2
    base_vol = (row['final_max_vol'] + row['final_min_vol']) / 2

    # ------------------ 生成波动数据（含异常趋势） ------------------
    amps = generate_stable_trend_series(base_amp, row['final_min_amp'], row['final_max_amp'], points, 1)
    rates = generate_stable_trend_series(base_rate, row['final_min_rate'], row['final_max_rate'], points, 1)
    temps = generate_stable_trend_series(base_temp, row['final_min_temp'], row['final_max_temp'], points, 1)
    vols = generate_stable_trend_series(base_vol, row['final_min_vol'], row['final_max_vol'], points, -1)

    # ------------------ 生成每一条数据点 ------------------
    for j in range(points):
        now = start_time + timedelta(seconds=j * sampling_interval)
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]

        # ------------------ 判断是否异常 ------------------
        f_alarm = int(
            amp > row['final_max_amp'] or amp < row['final_min_amp'] or
            rate > row['final_max_rate'] or rate < row['final_min_rate'] or
            temp > row['final_max_temp'] or temp < row['final_min_temp'] or
            vol > row['final_max_vol'] or vol < row['final_min_vol']
        )

        # ------------------ 若为正常，强行限制不超出 ------------------
        if f_alarm == 0:
            amp = np.clip(amp, row['final_min_amp'], row['final_max_amp'])
            rate = np.clip(rate, row['final_min_rate'], row['final_max_rate'])
            temp = np.clip(temp, row['final_min_temp'], row['final_max_temp'])
            vol = np.clip(vol, row['final_min_vol'], row['final_max_vol'])

        # ------------------ 生成21位唯一 f_id ------------------
        timestamp = int(now.timestamp() * 1e6)  # 微秒时间戳
        f_id = f"{timestamp:017d}{i:02d}{random.randint(0,999):03d}"

        # ------------------ 添加进结果集 ------------------
        abnormal_data.append({
            'f_id': f_id,
            'f_device': motor_id,
            'f_err_code': '0',
            'f_run_signal': 4,
            'f_time': now,
            'f_amp': round(amp, 3),
            'f_vol': round(vol, 3),
            'f_temp': round(temp, 3),
            'f_rate': int(round(rate)),
            'f_note': '模拟数据',
            'f_name': motor_id,
            'f_alarm': f_alarm
        })

# ------------------ 7. 插入数据库 ------------------
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f"✅ 插入完成：{len(df)} 条，其中异常 {df[df.f_alarm == 1].shape[0]} 条")
