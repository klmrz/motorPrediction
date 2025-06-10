import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import random

# 数据库连接
db_url = 'postgresql://postgres:dz123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 读取正常范围表
range_df = pd.read_csv("E:/Work/NewProject/motorPrediction/normalRange/min_max_ranges.csv")

# 获取每台电机的最新数据时间
latest_times = pd.read_sql('''
    SELECT f_device, MAX(f_time) as latest_time
    FROM dj_mock_data
    GROUP BY f_device
''', engine)

# 波动+趋势函数
def generate_trend_series(base_value, min_limit, max_limit, steps, exceed_direction):
    x = np.linspace(0, 4 * np.pi, steps)
    base_series = base_value + 0.3 * (max_limit - min_limit) * np.sin(x) / 10
    noise = np.random.uniform(-0.1, 0.1, steps) * (max_limit - min_limit)
    series = base_series + noise

    # 最后10%添加趋势
    trend_start = int(steps * 0.9)
    trend_len = steps - trend_start
    if exceed_direction == 1:
        trend = np.linspace(0, max_limit * 0.15, trend_len)
        series[trend_start:] += trend
    else:
        trend = np.linspace(0, min_limit * 0.1, trend_len)
        series[trend_start:] -= trend

    return series

# 参数设置
days = 3
sampling_interval = 2
points = 24 * 3600 * days // sampling_interval
abnormal_data = []

# 生成每台电机数据
for i in range(1, 8):
    motor_id = f"电机{i}"
    row = range_df[range_df['f_device'] == motor_id].iloc[0]
    latest_time = pd.to_datetime(latest_times[latest_times['f_device'] == motor_id]['latest_time'].values[0])
    start_time = latest_time + timedelta(minutes=5)

    # 生成每个参数的序列（带波动和异常趋势）
    amps = generate_trend_series((row['final_max_amp'] + row['final_min_amp']) / 2,
                                 row['final_min_amp'], row['final_max_amp'], points, 1)
    rates = generate_trend_series((row['final_max_rate'] + row['final_min_rate']) / 2,
                                  row['final_min_rate'], row['final_max_rate'], points, 1)
    temps = generate_trend_series((row['final_max_temp'] + row['final_min_temp']) / 2,
                                  row['final_min_temp'], row['final_max_temp'], points, 1)
    vols = generate_trend_series((row['final_max_vol'] + row['final_min_vol']) / 2,
                                 row['final_min_vol'], row['final_max_vol'], points, -1)

    for j in range(points):
        now = start_time + timedelta(seconds=j * sampling_interval)
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]

        # 添加实时微噪声
        amp += random.uniform(-0.001, 0.001)
        rate += random.uniform(-0.1, 0.1)
        temp += random.uniform(-0.05, 0.05)
        vol += random.uniform(-0.005, 0.005)

        # 最后5%强制越界
        if j > int(points * 0.95):
            amp = max(amp, row['final_max_amp'] + 0.1)
            temp = max(temp, row['final_max_temp'] + 1)
            rate = max(rate, row['final_max_rate'] + 1)
            vol = min(vol, row['final_min_vol'] - 1)

        # 判断异常
        f_alarm = int(
            amp > row['final_max_amp'] or amp < row['final_min_amp'] or
            rate > row['final_max_rate'] or rate < row['final_min_rate'] or
            temp > row['final_max_temp'] or temp < row['final_min_temp'] or
            vol > row['final_max_vol'] or vol < row['final_min_vol']
        )

        if f_alarm == 0:
            amp = np.clip(amp, row['final_min_amp'], row['final_max_amp'])
            rate = np.clip(rate, row['final_min_rate'], row['final_max_rate'])
            temp = np.clip(temp, row['final_min_temp'], row['final_max_temp'])
            vol = np.clip(vol, row['final_min_vol'], row['final_max_vol'])

        timestamp = int(now.timestamp() * 1e6)
        f_id = f"{timestamp:017d}{i:02d}{random.randint(0,999):03d}"

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

# 插入数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f"✅ 插入完成：{len(df)} 条，其中异常：{df[df.f_alarm == 1].shape[0]} 条")
