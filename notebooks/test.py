import pandas as pd

# parquet 파일 읽기
df = pd.read_parquet("/home/koopman_pdf_trader/data/raw/binance_monthly/BTCUSDT/2025-04/agg_trades_part121.parquet")

# 앞부분 확인
print(df.head())

# 컬럼 정보
print(df.info())