# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment in India.csv")

print("First 5 rows:")
print(df.head())

df.columns = df.columns.str.strip()

df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Rate'
}, inplace=True)

print("\nMissing values:")
print(df.isnull().sum())

df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)


df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['unemployment_Rate'])
plt.title("unemployment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")


plt.figure(figsize=(12,5))
sns.barplot(x='Region', y='Unemployment_Rate', data=df)
plt.title("unemployment by state")
plt.xticks(rotation=90)


covid_df = df[df['Year'] == 2020]

print("\nCOVID Data:")
print(covid_df.head())

plt.figure(figsize=(10,5))
plt.plot(covid_df['Date'], covid_df['Unemployment_Rate'])
plt.title("COVID-19 Impact on Unemployment (2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")


monthly_avg = df.groupby('Month')['Unemployment_Rate'].mean()

plt.figure(figsize=(8,4))
monthly_avg.plot(kind='bar')
plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Avg Rate")


plt.figure(figsize=(6,4))
sns.heatmap(df[['Unemployment_Rate','Employed','Labour_Rate']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")

plt.tight_layout()
plt.show()
