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

# 波动+异常趋势生成函数
def generate_trend_series(base_value, min_limit, max_limit, steps, exceed_direction):
    normal_steps = int(steps * 0.9)
    pre_abnormal_steps = int(steps * 0.05)  # 第91%~95%过渡段
    hard_abnormal_steps = steps - normal_steps - pre_abnormal_steps  # 第96%~100%强异常段

    wave_range = 0.05 * (max_limit - min_limit)

    # 正常波动段（接近上限/下限）
    x = np.linspace(0, 2 * np.pi * (normal_steps // 30), normal_steps)
    wave = wave_range * np.sin(x)
    noise = np.random.uniform(-wave_range * 0.1, wave_range * 0.1, normal_steps)
    base_series = base_value + wave + noise

    if exceed_direction == 1:  # 趋势向上（如电流）
        base_series = np.clip(base_series, min_limit, max_limit * 0.99)
        pre_start = base_series[-1]
        pre_end = max_limit * 1.1
        hard_end = max_limit * 1.25
        pre_trend = np.linspace(pre_start, pre_end, pre_abnormal_steps)
        hard_trend = np.linspace(pre_end, hard_end, hard_abnormal_steps)
    else:  # 趋势向下（如电压）
        base_series = np.clip(base_series, min_limit * 1.01, max_limit)
        pre_start = base_series[-1]
        pre_end = min_limit * 0.9
        hard_end = min_limit * 0.8
        pre_trend = np.linspace(pre_start, pre_end, pre_abnormal_steps)
        hard_trend = np.linspace(pre_end, hard_end, hard_abnormal_steps)

    return np.concatenate([base_series, pre_trend, hard_trend])


# 参数设置
days = 3
sampling_interval = 2  # 每2秒
points = 24 * 3600 * days // sampling_interval
abnormal_data = []

# 生成数据
for i in range(1, 8):
    motor_id = f"电机{i}"
    row = range_df[range_df['f_device'] == motor_id].iloc[0]
    latest_time = pd.to_datetime(latest_times[latest_times['f_device'] == motor_id]['latest_time'].values[0])
    start_time = latest_time + timedelta(minutes=5)

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

        # 添加轻微扰动
        amp += random.uniform(-0.001, 0.001)
        rate += random.uniform(-0.1, 0.1)
        temp += random.uniform(-0.05, 0.05)
        vol += random.uniform(-0.005, 0.005)

        # 判断是否为异常
        f_alarm = int(
            amp > row['final_max_amp'] or amp < row['final_min_amp'] or
            rate > row['final_max_rate'] or rate < row['final_min_rate'] or
            temp > row['final_max_temp'] or temp < row['final_min_temp'] or
            vol > row['final_max_vol'] or vol < row['final_min_vol']
        )

        # 仅前90%可以 clip 回正常值
        if j <= int(points * 0.9) and f_alarm == 0:
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

# 写入数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f" 插入完成：{len(df)} 条，其中异常：{df[df.f_alarm == 1].shape[0]} 条")
