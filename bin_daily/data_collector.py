import yfinance as yf
import pandas as pd
import os

# Liste des tickers (symboles boursiers) des entreprises sélectionnées
tickers_list = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "SQ", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE"
]

# Période pour les données historiques (ici, 5 ans)
start_date = "2019-01-01"
end_date = "2023-12-31"

# Créer un dossier pour stocker les fichiers CSV s'il n'existe pas
output_dir = 'donnees_bourse'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Boucle pour télécharger les données de chaque entreprise
for ticker in tickers_list:
    try:
        print(f"Téléchargement des données pour {ticker}...")
        # Télécharge les données historiques
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"Aucune donnée trouvée pour {ticker}. Peut-être que le ticker est incorrect ou n'existait pas pour la période donnée.")
            continue

        # Sauvegarder les données dans un fichier CSV
        file_path = os.path.join(output_dir, f"{ticker}_data.csv")
        data.to_csv(file_path)
        print(f"Données pour {ticker} sauvegardées dans {file_path}")

    except Exception as e:
        print(f"Une erreur est survenue lors du téléchargement des données pour {ticker}: {e}")

print("\n--- Téléchargement terminé ---")