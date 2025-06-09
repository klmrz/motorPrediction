import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine

# 数据库连接
db_url = 'postgresql://postgres:dz123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 读取正常范围表
range_df = pd.read_csv("min_max_ranges.csv")

# 从真实数据库中获取每台电机一个样本点（用于 f_name 字段）
query = '''
    SELECT DISTINCT ON (f_device) f_device, f_name
    FROM dj_mock_data
    ORDER BY f_device, f_time DESC
'''
device_info_df = pd.read_sql(query, engine)
device_info_map = dict(zip(device_info_df['f_device'], device_info_df['f_name']))

# 波动生成函数
def generate_fluctuating_series(start, peak1, trough, peak2, exceed_value, steps):
    third = steps // 3
    remain = steps - 3 * third
    series = np.concatenate([
        np.linspace(start, peak1, third),
        np.linspace(peak1, trough, third),
        np.linspace(trough, peak2, third + remain)
    ])
    series[-1] = exceed_value
    return series

# 模拟参数设置
interval_minutes = 5
points = 864  # 3天，每5分钟采样
start_time = datetime(2025, 6, 3, 0, 0)
abnormal_data = []

# 模拟电机编号1~7
for i in range(1, 8):
    motor_id = f"电机{i}"
    range_row = range_df[range_df['f_device'] == motor_id].iloc[0]
    f_name = device_info_map.get(motor_id, f"未知设备{i}")

    # 基线设置
    base_amp = range_row['final_max_amp'] * 0.9
    base_rate = range_row['final_max_rate'] * 0.9
    base_temp = range_row['final_max_temp'] * 0.9
    base_vol = range_row['final_min_vol'] * 1.05

    # 波动曲线生成
    amps = generate_fluctuating_series(base_amp, base_amp * 1.05, base_amp * 0.98, base_amp * 1.1,
                                       range_row['final_max_amp'] + 0.05, steps=points)
    rates = generate_fluctuating_series(base_rate, base_rate * 1.05, base_rate * 0.98, base_rate * 1.1,
                                        range_row['final_max_rate'] + 0.5, steps=points)
    temps = generate_fluctuating_series(base_temp, base_temp * 1.05, base_temp * 0.98, base_temp * 1.1,
                                        range_row['final_max_temp'] + 1.0, steps=points)
    vols = generate_fluctuating_series(base_vol, base_vol * 0.97, base_vol * 1.01, base_vol * 0.95,
                                       range_row['final_min_vol'] - 1.0, steps=points)

    # 构造波动数据
    for j in range(points):
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]
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
            'f_time': start_time + timedelta(minutes=j * interval_minutes),
            'f_amp': round(amp, 3),
            'f_vol': round(vol, 3),
            'f_temp': round(temp, 3),
            'f_rate': round(rate, 3),
            'f_note': '模拟数据',
            'f_name': f_name,
            'f_alarm': f_alarm
        })

# 写入数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data', con=engine, if_exists='append', index=False)

print(f"✅ 插入完成：{len(df)} 条模拟数据，其中异常条数：{df[df.f_alarm == 1].shape[0]}")
