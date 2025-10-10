import pandas as pd
import numpy as np

# Load Phase 5 submission
df = pd.read_csv('dataset/submission_xgboost_phase5.csv')

print("Before fix:")
print(f"  Negative prices: {(df['price'] < 0).sum()}")
print(f"  Min price: ${df['price'].min():.2f}")
print(f"  Max price: ${df['price'].max():.2f}")
print(f"  Mean price: ${df['price'].mean():.2f}")

# Fix negative and extreme values
# Clip to reasonable range [0, 1000]
df['price'] = np.clip(df['price'], 0.0, 1000.0)

# Save fixed submission
df.to_csv('dataset/submission_xgboost_phase5.csv', index=False)

print("\nAfter fix:")
print(f"  Negative prices: {(df['price'] < 0).sum()}")
print(f"  Min price: ${df['price'].min():.2f}")
print(f"  Max price: ${df['price'].max():.2f}")
print(f"  Mean price: ${df['price'].mean():.2f}")
print(f"\n✅ Fixed and saved: dataset/submission_xgboost_phase5.csv")
