from src.data import load_mnist_arff, split_data
from src.utils import add_label_noise
from src.train import *
from src.evaluate import evaluate
import pandas as pd

DATA_PATH = "mnist_784.arff"


X, y = load_mnist_arff(DATA_PATH)

X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

results = []

for noise in noise_levels:

    print(f"\nRunning experiment for noise level: {noise}")

    y_train_noisy = add_label_noise(y_train.values, noise)

    # Train models
    model_LR = train_model_LR(X_train, y_train_noisy)
    # model_SVC = train_model_SVC(X_train, y_train_noisy)
    model_RF = train_model_RF(X_train, y_train_noisy)
    model_MLP = train_model_MLP(X_train, y_train_noisy)

    # Evaluate models
    acc_lr = evaluate(model_LR, X_test, y_test)
    # acc_svc = evaluate(model_SVC, X_test, y_test)
    acc_rf = evaluate(model_RF, X_test, y_test)
    acc_mlp = evaluate(model_MLP, X_test, y_test)

    # Save results
    results.append({
        "Noise Level": noise,
        "Logistic Regression": acc_lr,
        # "Linear SVC": acc_svc,
        "Random Forest": acc_rf,
        "MLP": acc_mlp
    })

# Create dataframe
results_df = pd.DataFrame(results)

# Print results table
print("\nResults Table:")
print(results_df)

# Save results
results_df.to_csv("results/results.csv", index=False)




