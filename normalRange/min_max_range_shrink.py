import pandas as pd
from sqlalchemy import create_engine

# 设置数据库连接
db_url = 'postgresql://dbreader:db%40123456@117.72.55.146:1012/postgres'
engine = create_engine(db_url)

# 查询原始数据
query_raw = """
SELECT
    f_device,
    f_amp AS amp,
    f_vol AS vol,
    f_temp AS temp,
    f_rate AS rate,
    f_time
FROM dj_data2
WHERE f_time > '2025-05-05' AND f_time < '2025-06-08'
"""
raw_df = pd.read_sql(query_raw, engine)

# 获取初始的最大最小范围
query_bounds = """
SELECT
  f_device,
  MIN(f_amp) AS min_amp,
  MAX(f_amp) AS max_amp,
  MIN(f_vol) AS min_vol,
  MAX(f_vol) AS max_vol,
  MIN(f_temp) AS min_temp,
  MAX(f_temp) AS max_temp,
  MIN(f_rate) AS min_rate,
  MAX(f_rate) AS max_rate
FROM dj_data2
WHERE f_time > '2025-05-05' AND f_time < '2025-06-05'
GROUP BY f_device
"""
bounds_df = pd.read_sql(query_bounds, engine)

# 修改后的内缩逻辑：先缩 min，再缩 max，分别逐步进行
def shrink_ranges(bounds_df, raw_df, step=0.05, threshold=0.95):
    params = ['amp', 'vol', 'temp', 'rate']
    results = []

    for _, row in bounds_df.iterrows():
        device = row['f_device']
        device_data = raw_df[raw_df['f_device'] == device]
        result = {'f_device': device}

        for p in params:
            min_p = row[f'min_{p}']
            max_p = row[f'max_{p}']

            # Step 1：先收缩 min 值
            while True:
                new_min = min_p + step
                if new_min >= max_p:
                    break
                ratio = device_data[p].between(new_min, max_p).mean()
                if ratio >= threshold:
                    min_p = new_min
                else:
                    break

            # Step 2：再收缩 max 值
            while True:
                new_max = max_p - step
                if min_p >= new_max:
                    break
                ratio = device_data[p].between(min_p, new_max).mean()
                if ratio >= threshold:
                    max_p = new_max
                else:
                    break

            result[f'final_min_{p}'] = round(min_p, 3)
            result[f'final_max_{p}'] = round(max_p, 3)

        results.append(result)

    return pd.DataFrame(results)

# 执行迭代内缩
final_df = shrink_ranges(bounds_df, raw_df)

# 保存结果
final_df.to_csv("min_max_ranges.csv", index=False)
print("已保存到 min_max_ranges.csv")