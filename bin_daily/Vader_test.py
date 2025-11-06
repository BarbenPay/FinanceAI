from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Créer l'analyseur
analyzer = SentimentIntensityAnalyzer()

# Phrase très positive avec un jargon financier ("to the moon") et des emojis
phrase1 = "I love TSLA, this stock is going to the moon! 🚀💰🤑"
score1 = analyzer.polarity_scores(phrase1)
print(f"Phrase : '{phrase1}'\nScore : {score1}\n")

# Phrase négative
phrase2 = "AMC is crashing, selling all my shares. Terrible performance."
score2 = analyzer.polarity_scores(phrase2)
print(f"Phrase : '{phrase2}'\nScore : {score2}\n")

# Phrase plus neutre
phrase3 = "Looking at the NVDA chart."
score3 = analyzer.polarity_scores(phrase3)
print(f"Phrase : '{phrase3}'\nScore : {score3}\n")