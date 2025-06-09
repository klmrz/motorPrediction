import pandas as pd
from sqlalchemy import create_engine

# 设置数据库连接
db_url = 'postgresql://dbreader:db%40123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 读取最终范围文件
#range_df = pd.read_csv('motor_final_ranges.csv')
range_df = pd.read_csv("E:/Work/NewProject/motorPrediction/normalRange/min_max_ranges.csv")

# 查询历史数据
query = '''
SELECT f_device,
       f_amp AS amp,
       f_vol AS vol,
       f_temp AS temp,
       f_rate AS rate,
       f_time
FROM dj_data2
WHERE f_time > '2025-05-05' AND f_time < '2025-06-05'
'''
df = pd.read_sql(query, engine)

# 参数列表
params = ['amp', 'vol', 'temp', 'rate']
results = []

# 验证每个参数的范围覆盖率
for _, row in range_df.iterrows():
    motor_id = row['f_device']
    motor_data = df[df['f_device'] == motor_id]

    if motor_data.empty:
        continue

    for param in params:
        min_col = f'final_min_{param}'
        max_col = f'final_max_{param}'

        if min_col not in row or max_col not in row:
            continue

        param_min = row[min_col]
        param_max = row[max_col]

        values = motor_data[param].dropna()
        if values.empty:
            continue

        in_range_ratio = values.between(param_min, param_max).mean()

        results.append({
            "f_device": motor_id,
            "parameter": param,
            "min_range": round(param_min, 3),
            "max_range": round(param_max, 3),
            "in_range_percent": round(in_range_ratio * 100, 2)
        })

# 生成验证结果
result_df = pd.DataFrame(results)
result_df.to_csv("check_min_max_inRange.csv", index=False, encoding="utf-8-sig")
print("验证结果已保存为 check_min_max_inRange.csv")