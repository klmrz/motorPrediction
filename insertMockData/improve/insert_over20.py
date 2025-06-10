# 修改一：
# 异常区间：改为电流、温度、负载率要超过最大值 20%，电压要低于最小值 20%。
# 提前让这些参数接近上限，逐渐升高，直到达到目标异常值
# 修改二：异常时电流、温度、负载率、低压同时异常（强相关）
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

# 波动+异常趋势生成函数（逐渐升高至目标异常值）
def generate_trend_series(base_value, min_limit, max_limit, steps, exceed_direction):
    normal_steps = int(steps * 0.9)
    abnormal_steps = steps - normal_steps
    wave_range = 0.05 * (max_limit - min_limit)

    # 正常波动（更高频）
    x = np.linspace(0, 6 * np.pi * (normal_steps // 30), normal_steps)
    wave = wave_range * np.sin(x)
    noise = np.random.uniform(-wave_range * 0.1, wave_range * 0.1, normal_steps)
    base_value = (max_limit + min_limit) / 2
    base_series = base_value + wave + noise
    base_series = np.clip(base_series, min_limit, max_limit * 0.98)

    # 缓冲阶段（接近上限）
    trend_start = np.linspace(max_limit * 0.98, max_limit, int(abnormal_steps * 0.2))
    # 异常阶段（超过上限20%）
    trend_abnormal = np.linspace(max_limit, max_limit * 1.2, abnormal_steps - len(trend_start))

    return np.concatenate([base_series, trend_start, trend_abnormal])

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

    # 使用电流趋势作为主趋势
    amp_base = generate_trend_series(row['final_min_amp'], row['final_max_amp'], points, 1)
    rate_base = amp_base / amp_base.max() * row['final_max_rate'] * 1.2
    temp_base = amp_base / amp_base.max() * row['final_max_temp'] * 1.2
    vol_base = 1 - (amp_base / amp_base.max())
    vol_base = vol_base / vol_base.max() * row['final_min_vol'] * 0.8

    for j in range(points):
        now = start_time + timedelta(seconds=j * sampling_interval)
        amp = amp_base[j] + random.uniform(-0.001, 0.001)
        rate = rate_base[j] + random.uniform(-0.1, 0.1)
        temp = temp_base[j] + random.uniform(-0.05, 0.05)
        vol = vol_base[j] + random.uniform(-0.005, 0.005)

        # 判断异常数据，确保超出20%
        f_alarm = int(
            amp > row['final_max_amp'] * 1.2 or amp < row['final_min_amp'] * 0.8 or
            rate > row['final_max_rate'] * 1.2 or rate < row['final_min_rate'] * 0.8 or
            temp > row['final_max_temp'] * 1.2 or temp < row['final_min_temp'] * 0.8 or
            vol > row['final_max_vol'] * 1.2 or vol < row['final_min_vol'] * 0.8
        )

        # 如果发生异常，电流、温度、负载率和电压一起异常
        if f_alarm == 1:
            # 电流异常时，其他参数也超出正常范围
            if amp > row['final_max_amp'] * 1.2:
                rate = rate_base[j] * 1.2  # 超过负载率最大值
                temp = temp_base[j] * 1.2  # 超过温度最大值
                vol = vol_base[j] * 0.8  # 电压低于下限

        # 仅前90%允许 clip 回正常值
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
            'f_name': motor_id,  # 保留为 motor_id
            'f_alarm': f_alarm
        })

# 写入数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f"✅ 插入完成：{len(df)} 条，其中异常：{df[df.f_alarm == 1].shape[0]} 条")

