# 正常 ——> 临近（f_alarm = 1） ——> 异常（f_alarm = 2） ——> 停机维修(f_alarm = 2) ——> 恢复正常
# 假设：一天生成43200个点（每2秒一个），五天数据共216000点
# 设置五个阶段比例：60%正常+10%临近+10%异常+10%停机(无数据)+10%恢复

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import random

# 数据库连接
db_url = 'postgresql://postgres:dz123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 读取正常范围表
range_df = pd.read_csv("F:/Work/NewProject/motorPrediction/normalRange/min_max_ranges_after.csv")

# 获取最新时间
latest_times = pd.read_sql('''
    SELECT f_device, MAX(f_time) as latest_time
    FROM dj_mock_data_ver2
    GROUP BY f_device
''', engine)

name_mapping = pd.read_sql('''
    SELECT f_device, MAX(f_name) as f_name
    FROM dj_mock_data_ver2
    GROUP BY f_device
''', engine).set_index('f_device')['f_name'].to_dict()

sampling_interval = 2  # 秒
points = 24 * 3600 * 5 // sampling_interval  # 5天数据

abnormal_data = []

# 生成波动数据
def generate_wave_series(min_v, max_v, size):
    wave_range = 0.05 * (max_v - min_v)
    base = (max_v + min_v) / 2
    x = np.linspace(0, 2 * np.pi * (size // 30), size)
    wave = wave_range * np.sin(x)
    noise = np.random.uniform(-wave_range * 0.1, wave_range * 0.1, size)
    return np.clip(base + wave + noise, min_v, max_v)

# 生成正常数据
def generate_stage_series(min_v, max_v, stage, size):
    if stage in ['normal', 'near_abnormal', 'recover']:
        return generate_wave_series(min_v, max_v, size)
    elif stage == 'abnormal':
        return np.random.uniform(1.05 * max_v, 1.2 * max_v, size)
    else:
        return np.zeros(size)

# 构造电机1~7异常流程
for i in range(1, 8):
    motor_id = f"电机{i}"
    row = range_df[range_df['f_device'] == motor_id].iloc[0]
    f_name = name_mapping.get(motor_id, motor_id)
    latest_time = pd.to_datetime(latest_times[latest_times['f_device'] == motor_id]['latest_time'].values[0])
    start_time = latest_time + timedelta(minutes=5)

    stage_sizes = [int(points * 0.6), int(points * 0.1), int(points * 0.1), int(points * 0.1), int(points * 0.1)]
    stages = ['normal', 'near_abnormal', 'abnormal', 'stop', 'recover']

    offset = 0
    for idx, stage in enumerate(stages):
        step = stage_sizes[idx]
        if stage == 'stop':
            offset += step
            continue

        amps = generate_stage_series(row['final_min_amp'], row['final_max_amp'], stage, step)
        vols = generate_stage_series(row['final_min_vol'], row['final_max_vol'], stage, step)
        temps = generate_stage_series(row['final_min_temp'], row['final_max_temp'], stage, step)
        rates = generate_stage_series(row['final_min_rate'], row['final_max_rate'], stage, step)

        f_alarm = {'normal': 0, 'near_abnormal': 1, 'abnormal': 2, 'recover': 0}[stage]

        for j in range(step):
            now = start_time + timedelta(seconds=(offset + j) * sampling_interval)
            timestamp = int(now.timestamp() * 1e6)
            f_id = f"{timestamp:017d}{i:02d}{random.randint(0, 999):03d}"

            abnormal_data.append({
                'f_id': f_id,
                'f_device': motor_id,
                'f_err_code': '0',
                'f_run_signal': 4,
                'f_time': now,
                'f_amp': round(amps[j], 2),  # 电流保留两位小数
                'f_vol': round(vols[j], 1),  # 电压保留一位小数
                'f_temp': int(round(temps[j])),  # 温度取整
                'f_rate': int(round(rates[j])),  # 负载率取整
                'f_note': '模拟数据',
                'f_name': f_name,
                'f_alarm': f_alarm
            })

        offset += step

# 写入数据库
df = pd.DataFrame(abnormal_data)
df.to_sql('dj_mock_data_ver2', con=engine, if_exists='append', index=False)
print(f"插入完成：{len(df)} 条，f_alarm=2异常数：{df[df.f_alarm == 2].shape[0]} 条")
