import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine

# 数据库连接
db_url = 'postgresql://postgres:dz123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 读取正常范围表
range_df = pd.read_csv("E:/Work/NewProject/motorPrediction/normalRange/min_max_ranges.csv")

# 从数据库中获取每台电机的最新数据时间
latest_time_query = '''
    SELECT f_device, MAX(f_time) as latest_time
    FROM dj_mock_data
    GROUP BY f_device
'''
latest_times = pd.read_sql(latest_time_query, engine)

# 生成更高频的波动曲线（五段波动）
def generate_fluctuating_series(start, peak1, trough, peak2, exceed_value, steps, min_limit, max_limit):
    # 使用sin/cos函数模拟曲线波动
    x = np.linspace(0, 2 * np.pi, steps)
    series = start + (peak1 - start) * (np.sin(x) + 1) / 2
    series[steps // 5: 2 * steps // 5] = trough + (peak2 - trough) * (np.sin(x[steps // 5: 2 * steps // 5]) + 1) / 2
    series = np.clip(series, min_limit, max_limit)  # 确保波动值不超出正常范围
    series[-1] = np.clip(exceed_value, min_limit, max_limit)  # 最后一条数据确保不超出正常范围
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

    # 基线设置
    base_amp = range_row['final_max_amp'] * 0.9
    base_rate = range_row['final_max_rate'] * 0.9
    base_temp = range_row['final_max_temp'] * 0.9
    base_vol = range_row['final_min_vol'] * 1.05

    # 生成波动数据（每个电机 3 天的模拟数据）
    amps = generate_fluctuating_series(base_amp, base_amp * 1.05, base_amp * 0.98, base_amp * 1.1,
                                       range_row['final_max_amp'] + 0.05, steps=points,
                                       min_limit=range_row['final_min_amp'], max_limit=range_row['final_max_amp'])
    rates = generate_fluctuating_series(base_rate, base_rate * 1.05, base_rate * 0.98, base_rate * 1.1,
                                        range_row['final_max_rate'] + 0.5, steps=points,
                                        min_limit=range_row['final_min_rate'], max_limit=range_row['final_max_rate'])
    temps = generate_fluctuating_series(base_temp, base_temp * 1.05, base_temp * 0.98, base_temp * 1.1,
                                        range_row['final_max_temp'] + 1.0, steps=points,
                                        min_limit=range_row['final_min_temp'], max_limit=range_row['final_max_temp'])
    vols = generate_fluctuating_series(base_vol, base_vol * 0.97, base_vol * 1.01, base_vol * 0.95,
                                       range_row['final_min_vol'] - 1.0, steps=points,
                                       min_limit=range_row['final_min_vol'], max_limit=range_row['final_max_vol'])

    # 构造波动数据，每2秒钟生成一条数据
    for j in range(points):
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]

        # 模拟波动，直到最后一段（几分钟后）开始出现异常
        if j > points - 300:  # 假设最后300个点为异常数据
            # 逐渐超出正常范围并保持异常
            amp = min(amp + np.random.uniform(0.05, 0.1), range_row['final_max_amp'] + 0.1)  # 超过最大值
            rate = min(rate + np.random.uniform(0.05, 0.1), range_row['final_max_rate'] + 1)  # 超过最大值
            temp = min(temp + np.random.uniform(0.1, 0.2), range_row['final_max_temp'] + 1)  # 超过最大值
            vol = max(vol - np.random.uniform(0.05, 0.1), range_row['final_min_vol'] - 0.1)  # 低于最小值
            f_alarm = 1  # 标记为异常
        else:
            f_alarm = 0  # 标记为正常

        abnormal_data.append({
            'f_device': motor_id,
            'f_err_code': str(0),  # 将 f_err_code 转为字符串类型
            'f_run_signal': 4,
            'f_time': start_time + timedelta(seconds=j * sampling_interval),
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
