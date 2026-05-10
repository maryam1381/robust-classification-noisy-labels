import pandas as pd
import matplotlib.pyplot as plt

# read excel file
df = pd.read_csv("results/results.csv")

# x axis
noise = df["Noise Level"]

# create plot
plt.figure(figsize=(8,6))

plt.plot(noise, df["Logistic Regression"], marker='o', label="Logistic Regression")
plt.plot(noise, df["Random Forest"], marker='o', label="Random Forest")
plt.plot(noise, df["MLP"], marker='o', label="MLP")

# labels and title
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.title("Model Robustness Under Label Noise")

plt.legend()
plt.grid(True)

# save figure
plt.savefig("results/model_noise_comparison.png", dpi=300)

# show plot
plt.show()