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

# 波动+异常趋势生成函数（支持多参数强关联）
def generate_trend_series(base_value, min_limit, max_limit, steps, main_param, exceed_ratio=1.2):
    """
    生成稳定波动 + 分段异常的数据序列（支持多参数强关联）
    base_value: 中间基准值（正常范围中间值）
    min_limit/max_limit: 正常范围上下限
    steps: 总数据点数
    main_param: 主参数（如'amp'，触发异常的关键参数）
    exceed_ratio: 主参数超限比例（默认1.2，即超20%）
    """
    # 正常阶段占90%（前90%数据点），异常阶段占10%（后10%）
    normal_steps = int(steps * 0.9)
    abnormal_steps = steps - normal_steps

    # 正常波动参数（±5%正常范围，高频波动）
    wave_range = 0.05 * (max_limit - min_limit)
    fluct_interval = 15  # 波动间隔（30秒，15个2秒采样点）
    x = np.linspace(0, 2 * np.pi * (normal_steps // fluct_interval), normal_steps)
    wave = wave_range * np.sin(x)  # 正弦波波动
    noise = np.random.uniform(-wave_range * 0.1, wave_range * 0.1, normal_steps)  # 随机噪声
    base_series = base_value + wave + noise
    base_series = np.clip(base_series, min_limit, max_limit)  # 限制在正常范围

    # 正常阶段最后5%数据点：波动幅度减小，接近上限（为异常做准备）
    last_normal_end = int(normal_steps * 0.95)  # 正常阶段的95%位置（总steps的85.5%）
    base_series[last_normal_end:] *= 0.98  # 接近上限（例如电流从0.11降至0.1078）

    # 异常阶段（后10%数据点）：分为两部分（前5%缓慢接近，后5%突破）
    # 主参数目标值（超20%）
    main_target = max_limit * exceed_ratio
    # 关联参数目标值（电流超20%时，温度超15%、负载率超15%、电压低20%）
    temp_target = max_limit * 1.15
    rate_target = max_limit * 1.15
    vol_target = min_limit * 0.8

    # 异常阶段分段（前5%缓慢，后5%突破）
    slow_abnormal_steps = int(abnormal_steps * 0.5)  # 前5%异常步骤（总steps的4.5%）
    fast_abnormal_steps = abnormal_steps - slow_abnormal_steps  # 后5%异常步骤（总steps的5.5%）

    # 主参数异常趋势（分段线性增长）
    main_slow_trend = np.linspace(max_limit, main_target * 0.8, slow_abnormal_steps)  # 缓慢接近（到80%目标）
    main_fast_trend = np.linspace(main_target * 0.8, main_target, fast_abnormal_steps)  # 突破（到100%目标）
    main_abnormal_trend = np.concatenate([main_slow_trend, main_fast_trend])

    # 关联参数异常趋势（与主参数同步）
    temp_slow_trend = np.linspace(row['final_max_temp'], temp_target * 0.8, slow_abnormal_steps)
    temp_fast_trend = np.linspace(temp_target * 0.8, temp_target, fast_abnormal_steps)
    temp_abnormal_trend = np.concatenate([temp_slow_trend, temp_fast_trend])

    rate_slow_trend = np.linspace(row['final_max_rate'], rate_target * 0.8, slow_abnormal_steps)
    rate_fast_trend = np.linspace(rate_target * 0.8, rate_target, fast_abnormal_steps)
    rate_abnormal_trend = np.concatenate([rate_slow_trend, rate_fast_trend])

    vol_slow_trend = np.linspace(row['final_min_vol'], vol_target * 0.8, slow_abnormal_steps)
    vol_fast_trend = np.linspace(vol_target * 0.8, vol_target, fast_abnormal_steps)
    vol_abnormal_trend = np.concatenate([vol_slow_trend, vol_fast_trend])

    # 合并所有参数趋势（正常阶段+异常阶段）
    series = np.zeros((steps, 4))  # 4列分别对应amp, rate, temp, vol
    series[:normal_steps, 0] = base_series[:normal_steps]  # 电流正常波动
    series[:normal_steps, 1] = base_series[:normal_steps] * 1.1  # 负载率与电流强相关（正常时10%关联）
    series[:normal_steps, 2] = base_series[:normal_steps] * 1.05  # 温度与电流强相关（正常时5%关联）
    series[:normal_steps, 3] = base_series[:normal_steps] * 0.95  # 电压与电流强相关（正常时5%关联）

    # 异常阶段赋值（同步主参数趋势）
    series[normal_steps:, 0] = main_abnormal_trend  # 电流异常
    series[normal_steps:, 1] = rate_abnormal_trend  # 负载率异常
    series[normal_steps:, 2] = temp_abnormal_trend  # 温度异常
    series[normal_steps:, 3] = vol_abnormal_trend  # 电压异常

    return series

# 参数设置
days = 3
sampling_interval = 2  # 每2秒采样一次
points = 24 * 3600 * days // sampling_interval  # 129600（3天总数据点）
abnormal_data = []

# 生成数据
for i in range(1, 8):
    motor_id = f"电机{i}"
    row = range_df[range_df['f_device'] == motor_id].iloc[0]
    latest_time = pd.to_datetime(latest_times[latest_times['f_device'] == motor_id]['latest_time'].values[0])
    start_time = latest_time + timedelta(minutes=5)

    # 生成各参数趋势序列（以电流为主参数，关联其他参数）
    trend_data = generate_trend_series(
        (row['final_max_amp'] + row['final_min_amp']) / 2,  # 基准值（正常中间值）
        row['final_min_amp'],
        row['final_max_amp'],
        points,
        main_param='amp',  # 主参数为电流
        exceed_ratio=1.2   # 电流超上限20%
    )

    # 解包趋势数据（amp, rate, temp, vol）
    amps = trend_data[:, 0]
    rates = trend_data[:, 1]
    temps = trend_data[:, 2]
    vols = trend_data[:, 3]

    # 构造每一条数据点
    for j in range(points):
        now = start_time + timedelta(seconds=j * sampling_interval)
        amp, rate, temp, vol = amps[j], rates[j], temps[j], vols[j]

        # 添加轻微扰动（保持波动真实性）
        amp += random.uniform(-0.001, 0.001)
        rate += random.uniform(-0.1, 0.1)
        temp += random.uniform(-0.05, 0.05)
        vol += random.uniform(-0.005, 0.005)

        # 判断是否为异常（基于主参数是否超限）
        f_alarm = int(amp > row['final_max_amp'] * 1.2)  # 电流超20%即异常

        # 正常阶段限制不超出范围（前90%数据点）
        if j <= int(points * 0.9) and f_alarm == 0:
            amp = np.clip(amp, row['final_min_amp'], row['final_max_amp'])
            rate = np.clip(rate, row['final_min_rate'], row['final_max_rate'])
            temp = np.clip(temp, row['final_min_temp'], row['final_max_temp'])
            vol = np.clip(vol, row['final_min_vol'], row['final_max_vol'])

        # 生成21位唯一f_id
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