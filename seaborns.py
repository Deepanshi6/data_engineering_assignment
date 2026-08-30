import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("crime_dataset_india.csv")

df["Date Reported"] = pd.to_datetime(
    df["Date Reported"],
    format="mixed",
    dayfirst=True
)

df["Date of Occurrence"] = pd.to_datetime(
    df["Date of Occurrence"],
    format="mixed",
    dayfirst=True
)

df["Date Case Closed"] = pd.to_datetime(
    df["Date Case Closed"],
    format="mixed",
    dayfirst=True
)

df["Year"] = df["Date of Occurrence"].dt.year
df["Month"] = df["Date of Occurrence"].dt.month
df["Month Name"] = df["Date of Occurrence"].dt.month_name()
df["Day"] = df["Date of Occurrence"].dt.day

df["Hour"] = pd.to_datetime(
    df["Time of Occurrence"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour

sns.set_theme(style="whitegrid")


crime_counts = df["Crime Description"].value_counts().head(10)

# Chart Type: Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x=crime_counts.values,
    y=crime_counts.index,
    color="steelblue"
)
plt.title("Top 10 Crime Types")
plt.xlabel("Number of Cases")
plt.ylabel("Crime Description")
plt.tight_layout()
plt.show()


city_counts = df["City"].value_counts().head(10)

# Chart Type: Horizontal Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x=city_counts.values,
    y=city_counts.index,
    color="darkorange"
)
plt.title("Top 10 Cities by Number of Crimes")
plt.xlabel("Number of Cases")
plt.ylabel("City")
plt.tight_layout()
plt.show()


gender_counts = df["Victim Gender"].value_counts()

# Chart Type: Count Plot
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Victim Gender",
    hue="Victim Gender",
    palette="Set2",
    legend=False
)
plt.title("Victim Gender Distribution")
plt.xlabel("Victim Gender")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()


# Chart Type: Histogram
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="Victim Age",
    bins=20,
    kde=True,
    color="mediumpurple"
)
plt.title("Distribution of Victim Age")
plt.xlabel("Victim Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


yearly_crimes = (
    df.groupby("Year")
    .size()
    .reset_index(name="Cases")
)

# Chart Type: Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=yearly_crimes,
    x="Year",
    y="Cases",
    marker="o",
    color="crimson"
)
plt.title("Crimes by Year")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()


monthly_crimes = (
    df.groupby("Month")
    .size()
    .reset_index(name="Cases")
)

# Chart Type: Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=monthly_crimes,
    x="Month",
    y="Cases",
    marker="o",
    color="seagreen"
)
plt.title("Crimes by Month")
plt.xlabel("Month")
plt.ylabel("Number of Cases")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


domain_counts = (
    df["Crime Domain"]
    .value_counts()
    .reset_index()
)

domain_counts.columns = ["Crime Domain", "Cases"]

# Chart Type: Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=domain_counts,
    x="Cases",
    y="Crime Domain",
    color="teal"
)
plt.title("Crimes by Crime Domain")
plt.xlabel("Number of Cases")
plt.ylabel("Crime Domain")
plt.tight_layout()
plt.show()


closed_counts = (
    df["Case Closed"]
    .value_counts()
    .reset_index()
)

closed_counts.columns = ["Case Closed", "Cases"]

# Chart Type: Count Plot
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Case Closed",
    hue="Case Closed",
    palette="Set1",
    legend=False
)
plt.title("Case Closed Status")
plt.xlabel("Case Status")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()


weapon_counts = (
    df["Weapon Used"]
    .fillna("Unknown")
    .value_counts()
    .head(10)
)

# Chart Type: Horizontal Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x=weapon_counts.values,
    y=weapon_counts.index,
    color="slateblue"
)
plt.title("Top 10 Weapons Used")
plt.xlabel("Number of Cases")
plt.ylabel("Weapon Used")
plt.tight_layout()
plt.show()


top_cities = df["City"].value_counts().head(5).index

city_year_data = (
    df[df["City"].isin(top_cities)]
    .groupby(["Year", "City"])
    .size()
    .reset_index(name="Cases")
)

# Chart Type: Multi-Line Plot
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=city_year_data,
    x="Year",
    y="Cases",
    hue="City",
    marker="o",
    palette="tab10"
)
plt.title("Crime Trends for Top 5 Cities")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()


top_crimes = df["Crime Description"].value_counts().head(5).index

# Chart Type: Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df[df["Crime Description"].isin(top_crimes)],
    x="Crime Description",
    y="Victim Age",
    hue="Crime Description",
    palette="Set3",
    legend=False
)
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

# Chart Type: Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    crime_gender,
    annot=True,
    fmt="d",
    cmap="YlOrRd"
)
plt.title("Crime Domain vs Victim Gender")
plt.xlabel("Victim Gender")
plt.ylabel("Crime Domain")
plt.tight_layout()
plt.show()


numeric_columns = [
    "Victim Age",
    "Crime Code",
    "Police Deployed",
    "Year",
    "Month",
    "Hour"
]

correlation = df[numeric_columns].corr()

# Chart Type: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.show()


# Chart Type: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x="Victim Age",
    y="Police Deployed",
    hue="Crime Domain",
    palette="tab10",
    alpha=0.6
)
plt.title("Victim Age vs Police Deployed")
plt.xlabel("Victim Age")
plt.ylabel("Police Deployed")
plt.tight_layout()
plt.show()


# Chart Type: Regression Plot
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x="Victim Age",
    y="Police Deployed",
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)
plt.title("Victim Age vs Police Deployed with Regression Line")
plt.xlabel("Victim Age")
plt.ylabel("Police Deployed")
plt.tight_layout()
plt.show()


# Chart Type: Grouped Bar Plot
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df,
    x="Crime Domain",
    hue="Victim Gender",
    palette="Set2"
)
plt.title("Crime Domain by Victim Gender")
plt.xlabel("Crime Domain")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()