import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine

# 数据库连接
db_url = 'postgresql://postgres:dz123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 读取正常范围表
range_df = pd.read_csv("min_max_ranges.csv")

# 从真实数据库中获取每台电机的最新数据时间
latest_time_query = '''
    SELECT f_device, MAX(f_time) as latest_time
    FROM dj_mock_data
    GROUP BY f_device
'''
latest_times = pd.read_sql(latest_time_query, engine)

# 生成更高频的波动曲线（五段波动）
def generate_fluctuating_series(start, peak1, trough, peak2, exceed_value, steps):
    # 使用sin/cos函数模拟曲线波动
    x = np.linspace(0, 2 * np.pi, steps)
    series = start + (peak1 - start) * (np.sin(x) + 1) / 2
    series[steps // 5: 2 * steps // 5] = trough + (peak2 - trough) * (np.sin(x[steps // 5: 2 * steps // 5]) + 1) / 2
    series[-1] = exceed_value  # 最后一条数据超过最大值或最小值（异常）
    return series

# 模拟参数设置
days = 3
sampling_interval = 2  # 秒
points = 24 * 3600 * days // sampling_interval  # 129600

# 模拟电机编号1~7
for i in range(1, 8):
    motor_id = f"电机{i}"
    range_row = range_df[range_df['f_device'] == motor_id].iloc[0]
    f_name = range_row['f_device']

    # 获取该电机最新数据时间
    latest_time = latest_times[latest_times['f_device'] == motor_id]['latest_time'].values[0]
    start_time = pd.to_datetime(latest_time) + timedelta(minutes=5)  # 从最新数据的后5分钟开始

    # 基线设置
    base_amp = range_row['final_max_amp'] * 0.9
    base_rate = range_row['final_max_rate'] * 0.9
    base_temp = range_row['final_max_temp'] * 0.9
    base_vol = range_row['final_min_vol'] * 1.05

    # 生成波动数据（每个电机 3 天的模拟数据）
    amps = generate_fluctuating_series(base_amp, base_amp * 1.05, base_amp * 0.98, base_amp * 1.1,
                                       range_row['final_max_amp'] + 0.05, steps=points)
    rates = generate_fluctuating_series(base_rate, base_rate * 1.05, base_rate * 0.98, base_rate * 1.1,
                                        range_row['final_max_rate'] + 0.5, steps=points)
    temps = generate_fluctuating_series(base_temp, base_temp * 1.05, base_temp * 0.98, base_temp * 1.1,
                                        range_row['final_max_temp'] + 1.0, steps=points)
    vols = generate_fluctuating_series(base_vol, base_vol * 0.97, base_vol * 1.01, base_vol * 0.95,
                                       range_row['final_min_vol'] - 1.0, steps=points)

    # 构造波动数据，每2秒钟生成一条数据
    for j in range(points):
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]

        # 每5分钟数据变化较小，直到第5分钟开始波动
        if j % 300 == 0:  # 5分钟变化后，开始发生波动
            amp += np.random.uniform(-0.01, 0.01)
            rate += np.random.uniform(-0.01, 0.01)
            temp += np.random.uniform(-0.1, 0.1)
            vol += np.random.uniform(-0.05, 0.05)

        # 判断是否超出范围（超过正常范围则为异常数据）
        f_alarm = int(
            (amp > range_row['final_max_amp']) or
            (rate > range_row['final_max_rate']) or
            (temp > range_row['final_max_temp']) or
            (vol < range_row['final_min_vol'])
        )

        abnormal_data.append({
            'f_device': motor_id,
            'f_err_code': 0,
            'f_run_signal': 4,
            'f_time': start_time + timedelta(seconds=j * interval_seconds),
            'f_amp': round(amp, 3),
            'f_vol': round(vol, 3),
            'f_temp': round(temp, 3),
            'f_rate': round(rate, 3),
            'f_note': '模拟数据',
            'f_name': f_name,
            'f_alarm': f_alarm
        })

# 一次性插入所有数据到数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f" 插入完成：{len(df)} 条模拟数据，其中异常条数：{df[df.f_alarm == 1].shape[0]}")
