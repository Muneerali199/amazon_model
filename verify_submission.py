import pandas as pd

# Verify final submission
df = pd.read_csv('dataset/test_out.csv')

print("=" * 80)
print("FINAL SUBMISSION VERIFICATION")
print("=" * 80)

print(f"\n📊 File: dataset/test_out.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print(f"\nFirst 10 predictions:")
print(df.head(10))

print(f"\n📈 Price Statistics:")
print(df['price'].describe())

print(f"\n✅ Validation Checks:")
checks = [
    (df.shape[0] == 75000, f"Row count: {df.shape[0]} (expected: 75,000)"),
    (list(df.columns) == ['sample_id', 'price'], f"Columns: {list(df.columns)}"),
    (df.isnull().sum().sum() == 0, f"Missing values: {df.isnull().sum().sum()}"),
    ((df['price'] >= 0).all(), f"Negative prices: {(df['price'] < 0).sum()}"),
    (df['price'].min() >= 0, f"Min price: ${df['price'].min():.2f}"),
    (df['price'].max() < 10000, f"Max price: ${df['price'].max():.2f}")
]

all_pass = True
for check, desc in checks:
    status = "✅" if check else "❌"
    print(f"  {status} {desc}")
    all_pass = all_pass and check

print("\n" + "=" * 80)
if all_pass:
    print("✅ ALL CHECKS PASSED - READY FOR SUBMISSION!")
else:
    print("❌ VALIDATION FAILED - PLEASE REVIEW")
print("=" * 80)
