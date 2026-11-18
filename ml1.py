import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. ساخت داده‌های نمونه (متراژ خانه و قیمت)
data = {
    'metraj': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'price':  [2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600]
}

df = pd.DataFrame(data)
print("📊 داده‌ها:")
print(df)

# 2. جدا کردن ورودی (X) و خروجی (y)
X = df[['metraj']]   # باید دو بعدی باشه
y = df['price']

# 3. تقسیم داده به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. ساخت و آموزش مدل
model = LinearRegression()
model.fit(X_train, y_train)

# 5. پیش‌بینی روی داده تست
y_pred = model.predict(X_test)

# 6. ارزیابی مدل
print("\n✅ نتایج ارزیابی:")
print(f"خطای میانگین مطلق (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"دقت مدل (R²): {r2_score(y_test, y_pred):.2f}")

# 7. پیش‌بینی یه خانه جدید
new_house = [[95]]  # متراژ 95
predicted_price = model.predict(new_house)
print(f"\n🏠 قیمت پیش‌بینی شده برای خانه ۹۵ متری: {predicted_price[0]:.2f}")