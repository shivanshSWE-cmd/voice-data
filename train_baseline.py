import os
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
feat_path = os.path.join(BASE_DIR, "features.csv")

df = pd.read_csv(feat_path)
print(f"Loaded feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop(columns=['filename', 'label', 'language'])
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Stratified 10-Fold Cross Validation
rf_clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
et_clf = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('et', et_clf), ('gb', gb_clf)],
    voting='soft'
)

cv_scores = cross_val_score(ensemble_clf, X_scaled, y, cv=10, scoring='accuracy', n_jobs=-1)
print(f"\n=== Stratified 10-Fold Cross-Validation Accuracy ===")
print(f"Ensemble Model CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# 2. Holdout Test Performance (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

ensemble_clf.fit(X_train, y_train)
y_pred = ensemble_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n=== Holdout Test Set Accuracy (20% Holdout / {len(y_test)} samples) ===")
print(f"Test Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Feature importances via Random Forest
rf_clf.fit(X_scaled, y)
importances = rf_clf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Top 10 Most Important Acoustic Features:")
for feat, imp in feat_importances.head(10).items():
    print(f"  - {feat:<20} : {imp:.4f}")

# Save Model Payload
model_payload = {
    'scaler': scaler,
    'model': ensemble_clf,
    'feature_names': list(X.columns)
}

pkl_path = os.path.join(BASE_DIR, "voice_classifier.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(model_payload, f)
print(f"\nTrained ensemble model saved to: {pkl_path}")

# Plot Confusion Matrix and Feature Importances
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, y_pred, labels=['ai', 'human'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'], ax=axes[0])
axes[0].set_title('Confusion Matrix (4,000 English Samples)')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

top_feat = feat_importances.head(10)
sns.barplot(x=top_feat.values, y=top_feat.index, ax=axes[1], hue=top_feat.index, palette='Blues_r', legend=False)
axes[1].set_title('Top 10 Acoustic Features Importance')
axes[1].set_xlabel('Importance Weight')

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "training_results.png")
plt.savefig(plot_path, dpi=300)
plt.close(fig)
print(f"Evaluation plot saved to: {plot_path}")
