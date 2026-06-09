import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# Load Dataset
df = pd.read_csv("amazon.csv")

# Required Columns
df = df[['reviewText', 'overall']]

# Remove Missing Values
df.dropna(inplace=True)

# Sentiment Function
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Sentiment"] = df["reviewText"].apply(get_sentiment)

# Show First Rows
print(df.head())

# Sentiment Counts
print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

# -------------------------
# 1. Bar Chart
# -------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.savefig("sentiment_bar_chart.png")
plt.show()

# -------------------------
# 2. Pie Chart
# -------------------------

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%"
)
plt.title("Sentiment Analysis Pie Chart")
plt.savefig("sentiment_pie_chart.png")
plt.show()

# -------------------------
# 3. Word Cloud
# -------------------------

all_reviews = " ".join(df["reviewText"].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(all_reviews)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Amazon Reviews Word Cloud")
plt.savefig("wordcloud.png")
plt.show()

# -------------------------
# 4. Rating Distribution
# -------------------------

plt.figure(figsize=(7,4))
sns.countplot(x="overall", data=df)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

# -------------------------
# Summary
# -------------------------

print("\nProject Completed Successfully!")
print("Generated Files:")
print("1. sentiment_bar_chart.png")
print("2. sentiment_pie_chart.png")
print("3. wordcloud.png")
print("4. rating_distribution.png")

