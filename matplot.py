import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("crime_dataset_india.csv")


df["Date Reported"] = pd.to_datetime(df["Date Reported"], format='mixed', dayfirst=True)
df["Date of Occurrence"] = pd.to_datetime(df["Date of Occurrence"], format='mixed', dayfirst=True)
df["Date Case Closed"] = pd.to_datetime(df["Date Case Closed"], format='mixed', dayfirst=True)

df["Year"] = df["Date of Occurrence"].dt.year
df["Month"] = df["Date of Occurrence"].dt.month
df["Month Name"] = df["Date of Occurrence"].dt.month_name()
df["Day"] = df["Date of Occurrence"].dt.day
df["Hour"] = pd.to_datetime(df["Time of Occurrence"], format="%H:%M:%S", errors="coerce").dt.hour


crime_counts = df["Crime Description"].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.bar(crime_counts.index, crime_counts.values,color='Green')
plt.title("Top 10 Crime Types")
plt.xlabel("Crime Description")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


city_counts = df["City"].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.barh(city_counts.index, city_counts.values)
plt.title("Top 10 Cities by Number of Crimes")
plt.xlabel("Number of Cases")
plt.ylabel("City")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


gender_counts = df["Victim Gender"].value_counts()

plt.figure(figsize=(8, 5))
plt.pie(
    gender_counts.values,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Victim Gender Distribution")
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(df["Victim Age"], bins=20, edgecolor="black")
plt.title("Distribution of Victim Age",color='Orange')
plt.xlabel("Victim Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


yearly_crimes = df["Year"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(yearly_crimes.index, yearly_crimes.values, marker="o")
plt.title("Crimes by Year")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.grid(True)
plt.tight_layout()
plt.show()


monthly_crimes = df.groupby("Month").size()

plt.figure(figsize=(10, 6))
plt.plot(monthly_crimes.index, monthly_crimes.values, marker="o")
plt.title("Crimes by Month")
plt.xlabel("Month")
plt.ylabel("Number of Cases")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()

crime_domain_counts = df["Crime Domain"].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(crime_domain_counts.index, crime_domain_counts.values)
plt.title("Crimes by Crime Domain")
plt.xlabel("Crime Domain")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

closed_counts = df["Case Closed"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(closed_counts.index, closed_counts.values)
plt.title("Case Closed Status")
plt.xlabel("Case Status")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()


weapon_counts = df["Weapon Used"].fillna("Unknown").value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.bar(weapon_counts.index, weapon_counts.values)
plt.title("Top 10 Weapons Used")
plt.xlabel("Weapon")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


top_cities = df["City"].value_counts().head(5).index
city_year_data = df[df["City"].isin(top_cities)].groupby(["Year", "City"]).size().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
for city in city_year_data.columns:
    plt.plot(city_year_data.index, city_year_data[city], marker="o", label=city)
plt.title("Crime Trends for Top 5 Cities")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


crime_age_data = [
    df.loc[df["Crime Description"] == crime, "Victim Age"].dropna()
    for crime in df["Crime Description"].value_counts().head(5).index
]

crime_names = df["Crime Description"].value_counts().head(5).index

plt.figure(figsize=(12, 6))
plt.boxplot(crime_age_data, labels=crime_names)
plt.title("Victim Age Distribution by Top 5 Crime Types")
plt.xlabel("Crime Description")
plt.ylabel("Victim Age")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


crime_gender = pd.crosstab(
    df["Crime Domain"],
    df["Victim Gender"]
)

crime_gender.plot(
    kind="bar",
    figsize=(12, 6)
)
plt.title("Crime Domain by Victim Gender")
plt.xlabel("Crime Domain")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Victim Gender")
plt.tight_layout()
plt.show()


crime_year_domain = pd.crosstab(
    df["Year"],
    df["Crime Domain"]
)

crime_year_domain.plot(
    kind="area",
    figsize=(12, 6),
    alpha=0.7
)
plt.title("Crime Domains Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.legend(title="Crime Domain")
plt.tight_layout()
plt.show()


police_stats = df.groupby("Crime Domain")["Police Deployed"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(police_stats.index, police_stats.values)
plt.title("Average Police Deployed by Crime Domain")
plt.xlabel("Crime Domain")
plt.ylabel("Average Police Deployed")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


age_by_gender = [
    df.loc[df["Victim Gender"] == gender, "Victim Age"].dropna()
    for gender in df["Victim Gender"].unique()
]

gender_names = df["Victim Gender"].unique()

plt.figure(figsize=(10, 6))
plt.boxplot(age_by_gender, labels=gender_names)
plt.title("Victim Age Distribution by Gender")
plt.xlabel("Victim Gender")
plt.ylabel("Victim Age")
plt.tight_layout()
plt.show()


crime_city = pd.crosstab(
    df["City"].where(df["City"].isin(df["City"].value_counts().head(5).index)),
    df["Crime Domain"]
)

crime_city.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)
plt.title("Crime Domains in Top 5 Cities")
plt.xlabel("City")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Crime Domain")
plt.tight_layout()
plt.show()

crime_counts = df["Crime Description"].value_counts().head(5)

plt.figure(figsize=(10, 6))
plt.bar(
    crime_counts.index,
    crime_counts.values,
    width=0.6
)
plt.title("Top 5 Crime Types")
plt.xlabel("Crime Description")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
