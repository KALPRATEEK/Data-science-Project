import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei laden
df = pd.read_csv("AAPL_stock_data.csv")  # z. B. "AAPL_stock_data.csv"
df['Date'] = pd.to_datetime(df['Date'])

# Diagramm mit zwei Achsen: Kursdaten und Volumen
fig, ax1 = plt.subplots(figsize=(16, 8))

# Kursdaten plotten (linke y-Achse)
ax1.plot(df['Date'], df['Open'], label='Open', color='blue', alpha=0.7)
ax1.plot(df['Date'], df['High'], label='High', color='green', alpha=0.7)
ax1.plot(df['Date'], df['Low'], label='Low', color='red', alpha=0.7)
ax1.plot(df['Date'], df['Close'], label='Close', color='black', alpha=0.7)
ax1.set_xlabel('Datum')
ax1.set_ylabel('Kurs (in USD)')
ax1.tick_params(axis='x', rotation=45)

# Zweite y-Achse für Volumen
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Volume'], label='Volume', color='orange', alpha=0.4)
ax2.set_ylabel('Volumen')

# Legenden kombinieren
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Titel & Layout
plt.title('AAPL')
plt.grid(True)
plt.tight_layout()

# Diagramm speichern
plt.savefig("AAPL_stock_data_diagramm.png", dpi=300)
plt.show()  # Optional: Direkt anzeigen
