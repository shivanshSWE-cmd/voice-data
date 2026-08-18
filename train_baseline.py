import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "voice_classifier.pkl")
PLOT_PATH = os.path.join(BASE_DIR, "training_results.png")

def main():
    if not os.path.exists(FEATURES_PATH):
        print(f"Error: {FEATURES_PATH} not found. Run extract_features.py first.")
        return
        
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")
    
    X = df.drop(columns=['filename', 'label'])
    y = df['label']
    
    # 5-fold Cross-Validation evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=skf, scoring='accuracy')
    svm_scores = cross_val_score(SVC(kernel='rbf', probability=True, random_state=42), X, y, cv=skf, scoring='accuracy')
    
    print("\n=== Stratified 5-Fold Cross-Validation Accuracy ===")
    print(f"Random Forest CV Accuracy: {np.mean(rf_scores)*100:.2f}% (+/- {np.std(rf_scores)*100:.2f}%)")
    print(f"SVM CV Accuracy:           {np.mean(svm_scores)*100:.2f}% (+/- {np.std(svm_scores)*100:.2f}%)")
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train final Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n=== Test Set Performance (25% Holdout) ===")
    print(f"Test Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance Analysis
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = X.columns[indices[:10]]
    top_importances = importances[indices[:10]]
    
    print("Top 10 Most Important Features:")
    for f_name, imp in zip(top_features, top_importances):
        print(f"  - {f_name:20s}: {imp:.4f}")
        
    # Save Model & Scaler
    model_payload = {
        'model': clf,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }
    joblib.dump(model_payload, MODEL_PATH)
    print(f"\nTrained model and scaler saved to: {MODEL_PATH}")
    
    # Plot Feature Importance & Confusion Matrix
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=top_importances, y=top_features, palette='Blues_r')
    plt.title('Top 10 Acoustic Feature Importances')
    plt.xlabel('Importance Score')
    
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred, labels=['human', 'ai'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['human', 'ai'], yticklabels=['human', 'ai'])
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Evaluation plot saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()
