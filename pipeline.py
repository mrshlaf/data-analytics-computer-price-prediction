"""
Dasar Analitik Data (DAD) - Final Project Pipeline
Mahasiswa : Marshal Aufa Diliyana  |  NPM: 2406346913
Kelas     : Dasar Analitik-08     |  Dosen: Dr. Taufiq Alif Kurniawan
Dataset   : computer_prices_all.csv  (100,000 rows x 33 columns)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

DIR_DATA = "data"
DIR_FIGURES = "figures"
DIR_REPORTS = "reports"
os.makedirs(DIR_DATA, exist_ok=True)
os.makedirs(DIR_FIGURES, exist_ok=True)
os.makedirs(DIR_REPORTS, exist_ok=True)

DIVIDER = "=" * 80
print(DIVIDER)
print("  DAD FINAL PROJECT PIPELINE | Marshal Aufa Diliyana (2406346913)")
print(DIVIDER)


# ==========================================================================
# PHASE 1: BUSINESS & DATA UNDERSTANDING (Topic 01-04)
# ==========================================================================
print("\n" + "="*80)
print("  PHASE 1 | TOPICS 01-04: BUSINESS & DATA UNDERSTANDING")
print("="*80)

print("\n[STEP 0] Business Scope Analysis:")
business_scope = {
    "Problem Statement": "Manual computer pricing leads to inconsistent quotes and margin leaks.",
    "Business Objective": "Build a predictive system for fair-market custom computer pricing.",
    "Analytical Task": "Supervised Machine Learning - Regression (Target Variable: 'price')",
    "Technical Metric Target": "R-squared Score >= 85% with minimized Mean Absolute Error (MAE)."
}
for key, val in business_scope.items():
    print(f"  - {key}: {val}")

print("\n[STEP 1] Loading Dataset...")
csv_path = "computer_prices_all.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing dataset at {csv_path}. Please place computer_prices_all.csv in workspace.")

df = pd.read_csv(csv_path)
n_rows, n_cols = df.shape
print(f"  -> Dataset Loaded: {n_rows:,} rows, {n_cols} columns")

print("\n  - [Q1] Extracting tail(10) rows for inspection...")
df_tail_10 = df.tail(10)
df_tail_10.to_csv(os.path.join(DIR_DATA, "q1_bottom_10_rows.csv"))
print(f"  -> [CSV] Bottom 10 rows exported.")

print("\n[STEP 2] Attribute Profiling & Integrity Audit:")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
missing_counts = df.isnull().sum()
total_missing = missing_counts.sum()

print(f"  - Total Attributes: {n_cols}")
print(f"  - Total Records: {n_rows:,}")
print(f"  - Total Null Values: {total_missing}")
print(f"  - Categorical Attributes ({len(categorical_cols)}): {categorical_cols}")
print(f"  - Numerical Attributes ({len(numerical_cols)}): {numerical_cols}")

print("\n  - Evaluating text attributes unique values:")
obj_cardinality = df.select_dtypes(include=['object']).nunique()
for col in obj_cardinality.index:
    print(f"    * Attribute '{col}': {obj_cardinality[col]} unique values (sample: {df[col].dropna().unique()[:3]})")

print("\n[STEP 3] Summary Statistics Calculation:")
desc_specific = df[['weight_kg', 'warranty_months']].describe()
desc_specific.to_csv(os.path.join(DIR_DATA, "q4_specific_describe.csv"))
print(f"  -> [CSV] Descriptive metrics for weight & warranty saved.")

stats_summary = {}
key_stats_cols = ['price', 'ram_gb', 'storage_gb', 'weight_kg']
for col in key_stats_cols:
    col_data = df[col]
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR_val = Q3 - Q1
    stats_summary[col] = {
        "Mean": col_data.mean(),
        "Median": col_data.median(),
        "Mode": col_data.mode()[0],
        "Variance": col_data.var(),
        "Std Dev": col_data.std(),
        "Min": col_data.min(),
        "Max": col_data.max(),
        "Range": col_data.max() - col_data.min(),
        "IQR": IQR_val,
        "Skewness": skew(col_data),
        "Kurtosis": kurtosis(col_data)
    }
df_stats = pd.DataFrame(stats_summary)
df_stats.to_csv(os.path.join(DIR_DATA, "comprehensive_numerical_statistics.csv"))
print(f"  -> [CSV] Detailed statistical matrix exported.")

corr_matrix = df[numerical_cols].corr(method='pearson')

print("\n[STEP 4] Plotting Phase 1 Visuals...")

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, color='#2c3e50', bins=40, line_kws={'linewidth': 2})
plt.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: ${df['price'].mean():.2f}")
plt.axvline(df['price'].median(), color='blue', linestyle='-', linewidth=1.5, label=f"Median: ${df['price'].median():.2f}")
plt.title("Distribution of Computer Price Target", fontsize=14, weight='bold', pad=15)
plt.xlabel("Price ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_01_price_distribution.png"), dpi=200)
plt.close()
print("  -> [IMAGE 01] Saved: image_01_price_distribution.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(x='device_type', data=df, ax=axes[0, 0], palette='viridis', hue='device_type', legend=False)
axes[0, 0].set_title("Device Type Proportion", fontsize=12, weight='bold')
sns.countplot(y='brand', data=df, ax=axes[0, 1], palette='crest', order=df['brand'].value_counts().index, hue='brand', legend=False)
axes[0, 1].set_title("Brand Volume Distribution", fontsize=12, weight='bold')
sns.countplot(x='os', data=df, ax=axes[1, 0], palette='magma', order=df['os'].value_counts().index, hue='os', legend=False)
axes[1, 0].set_title("OS Market Share", fontsize=12, weight='bold')
sns.countplot(x='storage_type', data=df, ax=axes[1, 1], palette='rocket', order=df['storage_type'].value_counts().index, hue='storage_type', legend=False)
axes[1, 1].set_title("Storage Interface Prevalence", fontsize=12, weight='bold')
plt.suptitle("Univariate Categorical Frequency Audits", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_02_categorical_distributions.png"), dpi=200)
plt.close()
print("  -> [IMAGE 02] Saved: image_02_categorical_distributions.png")

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Pearson Correlation Heatmap (Continuous Attributes)", fontsize=14, weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_03_correlation_heatmap.png"), dpi=200)
plt.close()
print("  -> [IMAGE 03] Saved: image_03_correlation_heatmap.png")

plt.figure(figsize=(12, 6))
sns.boxplot(x='brand', y='price', data=df, palette='Set2', hue='brand', legend=False)
plt.title("Price Distribution Across Computer Brands", fontsize=14, weight='bold', pad=15)
plt.xlabel("Brand", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_04_brand_vs_price_boxplot.png"), dpi=200)
plt.close()
print("  -> [IMAGE 04] Saved: image_04_brand_vs_price_boxplot.png")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='ram_gb', y='price', data=df.sample(5000, random_state=42), alpha=0.3, color='#8e44ad')
plt.title("Scatter Plot: RAM Capacity vs Computer Price (N=5000 Sample)", fontsize=14, weight='bold', pad=15)
plt.xlabel("RAM Size (GB)", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_05_ram_vs_price_scatter.png"), dpi=200)
plt.close()
print("  -> [IMAGE 05] Saved: image_05_ram_vs_price_scatter.png")

plt.figure(figsize=(10, 6))
plt.hexbin(df['weight_kg'], df['price'], gridsize=30, cmap='inferno', mincnt=1)
cb = plt.colorbar(shrink=0.8)
cb.set_label('Point Density Count')
plt.title("Hexbin Bivariate Density: Physical Weight vs Price", fontsize=14, weight='bold', pad=15)
plt.xlabel("Weight (kg)", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_06_weight_vs_price_density.png"), dpi=200)
plt.close()
print("  -> [IMAGE 06] Saved: image_06_weight_vs_price_density.png")

report_1 = f"""================================================================================
  DASAR ANALITIK DATA (DAD) - MODULE REPORT: DATA UNDERSTANDING
  Syllabus Coverage: Topics 01 - 04 (Business & Data Understanding, EDA, Summary Stats)
================================================================================

1. BUSINESS SCOPE DEFINITIONS:
   - Problem Statement: {business_scope["Problem Statement"]}
   - Analytical Method: {business_scope["Analytical Task"]}
   - Target Objective: {business_scope["Technical Metric Target"]}

2. DATASET SUMMARY METRICS:
   - File Name: {csv_path}
   - Rows: {n_rows:,}
   - Columns: {n_cols}
   - Missing Elements Count: {total_missing}

3. CENTRAL TENDENCY & DISPERSION METRICS:
{df_stats.to_string()}

4. SKEWNESS & KURTOSIS (Target Price):
   - Skewness: {stats_summary["price"]["Skewness"]:.4f} (Positive skew / Right-tailed)
   - Kurtosis: {stats_summary["price"]["Kurtosis"]:.4f} (Leptokurtic curve peak)
================================================================================
"""
with open(os.path.join(DIR_REPORTS, "handson_data_understanding_report.txt"), "w") as f:
    f.write(report_1)
print("  -> [REPORT] handson_data_understanding_report.txt written.")


# ==========================================================================
# PHASE 2: DATA PREPARATION (Topic 05-06)
# ==========================================================================
print("\n" + "="*80)
print("  PHASE 2 | TOPICS 05-06: DATA PREPARATION (CLEANING & ENCODING)")
print("="*80)

outlier_candidates = ['price', 'ram_gb', 'storage_gb', 'weight_kg']
outlier_audit_before = {}
outlier_audit_after = {}

df_clean = df.copy()

for col in outlier_candidates:
    q75, q25 = np.percentile(df_clean[col], [75, 25])
    iqr = q75 - q25
    upper_bound = q75 + (1.5 * iqr)
    lower_bound = q25 - (1.5 * iqr)

    n_outliers_before = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].shape[0]
    outlier_audit_before[col] = n_outliers_before

    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    n_outliers_after = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].shape[0]
    outlier_audit_after[col] = n_outliers_after

print("  - Outliers BEFORE Winsorization Clamping:")
for col, val in outlier_audit_before.items():
    print(f"    * '{col}': {val:,} outlier points")
print("  - Outliers AFTER Winsorization Clamping:")
for col, val in outlier_audit_after.items():
    print(f"    * '{col}': {val} outlier points")

print("\n  - Processing pivot table aggregations...")
pivot_price = df_clean.pivot_table(index='brand', columns='device_type', values='price', aggfunc='mean')
pivot_count = df_clean.pivot_table(index='brand', columns='device_type', values='price', aggfunc='count')

print("\n  - Categorizing Target Variable (Price Binning):")
dividers = np.linspace(df_clean['price'].min(), df_clean['price'].max(), 4)
print(f"    * Linear Bins limits: {dividers}")
df_clean['price-binned'] = pd.cut(df_clean['price'], bins=dividers, labels=['Low', 'Medium', 'High'], include_lowest=True)
binned_counts = df_clean['price-binned'].value_counts()
print(binned_counts)

print("\n  - Performing Feature Encodings...")
ord_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df_clean['price-binned-numeric'] = df_clean['price-binned'].map(ord_mapping).astype(int)

nominal_features = ['brand', 'os', 'storage_type', 'cpu_brand', 'gpu_brand', 'wifi', 'device_type']
df_encoded = pd.get_dummies(df_clean, columns=nominal_features, drop_first=True, dtype=int)

columns_to_drop = ['model', 'cpu_model', 'gpu_model', 'form_factor', 'resolution', 'display_type', 'price-binned']
df_modelling = df_encoded.drop(columns=columns_to_drop)

df_modelling.to_csv(os.path.join(DIR_DATA, "clean_df.csv"), index=False)
print(f"  -> [CSV] Cleaned and Encoded dataset saved.")

print("\n[STEP 4] Plotting Phase 2 Visuals...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.boxplot(y=df['price'], ax=axes[0, 0], color='#e74c3c')
axes[0, 0].set_title("Price Distribution (Before Clamping)", weight='bold')
sns.boxplot(y=df_clean['price'], ax=axes[0, 1], color='#2ecc71')
axes[0, 1].set_title("Price Distribution (After Clamping)", weight='bold')
sns.boxplot(y=df['ram_gb'], ax=axes[1, 0], color='#e74c3c')
axes[1, 0].set_title("RAM Capacity (Before Clamping)", weight='bold')
sns.boxplot(y=df_clean['ram_gb'], ax=axes[1, 1], color='#2ecc71')
axes[1, 1].set_title("RAM Capacity (After Clamping)", weight='bold')
plt.suptitle("Outliers Audit (IQR Winsorization Clamping)", fontsize=15, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_07_outliers_before_after.png"), dpi=200)
plt.close()
print("  -> [IMAGE 07] Saved: image_07_outliers_before_after.png")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_price, cmap='YlOrRd', annot=True, fmt=".2f", cbar_kws={'label': 'Average Price ($)'})
plt.title("Pivot Heatmap: Average Price by Brand & Device Type", fontsize=14, weight='bold', pad=15)
plt.xlabel("Device Type", fontsize=12)
plt.ylabel("Brand", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_08_pivot_average_price.png"), dpi=200)
plt.close()
print("  -> [IMAGE 08] Saved: image_08_pivot_average_price.png")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_count, cmap='mako', annot=True, fmt="d", cbar_kws={'label': 'Unit Stock Count'})
plt.title("Pivot Heatmap: Stock Inventory by Brand & Device Type", fontsize=14, weight='bold', pad=15)
plt.xlabel("Device Type", fontsize=12)
plt.ylabel("Brand", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_09_pivot_distribution_count.png"), dpi=200)
plt.close()
print("  -> [IMAGE 09] Saved: image_09_pivot_distribution_count.png")

plt.figure(figsize=(10, 6))
sns.countplot(x='price-binned', data=df_clean, palette='crest', hue='price-binned', legend=False)
plt.title("Inventory Distribution by Price Tier Category", fontsize=14, weight='bold', pad=15)
plt.xlabel("Budget Tier", fontsize=12)
plt.ylabel("Volume Counts", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_10_binned_price_distribution.png"), dpi=200)
plt.close()
print("  -> [IMAGE 10] Saved: image_10_binned_price_distribution.png")

report_2 = f"""================================================================================
  DASAR ANALITIK DATA (DAD) - MODULE REPORT: DATA PREPARATION
  Syllabus Coverage: Topics 05 - 06 (Outliers, Pivot Tables, Feature Binning & Encoding)
================================================================================

1. OUTLIERS AUDIT (IQR CLAMPING METHOD):
   - Outliers Detected (Before winsorization):
     * price: {outlier_audit_before["price"]:,} points
     * ram_gb: {outlier_audit_before["ram_gb"]:,} points
     * storage_gb: {outlier_audit_before["storage_gb"]:,} points
     * weight_kg: {outlier_audit_before["weight_kg"]:,} points
   - Outliers Post Clamping: 0 (All features clamped safely)

2. FEATURE BINNING (Target variable price):
   - Price category counts:
{binned_counts.to_string()}

3. PIVOT TABLE SUMMARIES (Average Price):
{pivot_price.to_string()}

4. PIVOT TABLE SUMMARIES (Units Frequencies):
{pivot_count.to_string()}
================================================================================
"""
with open(os.path.join(DIR_REPORTS, "handson_data_preparation_report.txt"), "w") as f:
    f.write(report_2)
print("  -> [REPORT] handson_data_preparation_report.txt written.")


# ==========================================================================
# PHASE 3: FEATURE SELECTION (Topic 07)
# ==========================================================================
print("\n" + "="*80)
print("  PHASE 3 | TOPIC 07: FEATURE SELECTION (FILTER / WRAPPER / EMBEDDED)")
print("="*80)

X = df_modelling.drop(columns=['price'])
y = df_modelling['price']

scaler = StandardScaler()
X_scaled_arr = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns)

print("\n[STEP 1] Executing Filter Method...")
corr_with_price = X_scaled.corrwith(y).abs()
filter_selected_features = corr_with_price[corr_with_price >= 0.1].index.tolist()
print(f"  - Selected {len(filter_selected_features)} features with Correlation to Price >= 0.1")

filter_corr_matrix = X_scaled[filter_selected_features].corr().abs()
high_corr_pairs = []
for i in range(len(filter_selected_features)):
    for j in range(i + 1, len(filter_selected_features)):
        f1 = filter_selected_features[i]
        f2 = filter_selected_features[j]
        if filter_corr_matrix.loc[f1, f2] >= 0.8:
            high_corr_pairs.append((f1, f2))

print("  - Multicollinear High-Correlation Pairs detected:")
for p1, p2 in high_corr_pairs:
    print(f"    * Pair ('{p1}', '{p2}'): {filter_corr_matrix.loc[p1, p2]:.4f}")

cols_to_drop = ['cpu_cores', 'cpu_threads', 'cpu_boost_ghz', 'gpu_tier', 'charger_watts', 'psu_watts']
filter_final_features = [f for f in filter_selected_features if f not in cols_to_drop]
print(f"  -> Final Filtered dataset retains {len(filter_final_features)} independent features.")

df_filtered = X[filter_final_features].copy()
df_filtered['price'] = y
df_filtered.to_csv(os.path.join(DIR_DATA, "df_filtered.csv"), index=False)
print("  -> [CSV] Filter dataset saved.")

print("\n[STEP 2] Executing Wrapper Method (RFE Ridge)...")
X_train_rfe, _, y_train_rfe, _ = train_test_split(X_scaled, y, train_size=15000, random_state=42)
rfe_estimator = Ridge(alpha=1.0)
rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=15, step=1)
rfe_selector.fit(X_train_rfe, y_train_rfe)
wrapper_features = X.columns[rfe_selector.support_].tolist()
print(f"  -> Top 15 RFE Selected features: {wrapper_features}")

df_wrapper = X[wrapper_features].copy()
df_wrapper['price'] = y
df_wrapper.to_csv(os.path.join(DIR_DATA, "df_wrapper.csv"), index=False)
print("  -> [CSV] Wrapper dataset saved.")

print("\n[STEP 3] Executing Embedded Method (Random Forest Importance)...")
X_train_rf, _, y_train_rf, _ = train_test_split(X, y, train_size=15000, random_state=42)
rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_rf, y_train_rf)
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]

embedded_features = []
print("  - Feature Importance Rankings:")
for f_idx in indices:
    score = importances[f_idx]
    f_name = X.columns[f_idx]
    if score >= 0.005:
        embedded_features.append(f_name)
        print(f"    * Feature '{f_name}': {score:.4f} [RETAINED]")
    else:
        if len(embedded_features) < 10:
            print(f"    * Feature '{f_name}': {score:.4f} [REJECTED]")

df_embedded = X[embedded_features].copy()
df_embedded['price'] = y
df_embedded.to_csv(os.path.join(DIR_DATA, "df_embedded.csv"), index=False)
print("  -> [CSV] Embedded dataset saved.")

print("\n[STEP 4] Plotting Phase 3 Visuals...")

plt.figure(figsize=(16, 14))
sns.heatmap(X_scaled.corr(), cmap='rocket', annot=False, fmt=".2f", square=True, linewidths=0.1)
plt.title("Correlation Matrix Heatmap (All 51 Encoded Attributes)", fontsize=16, weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_11_full_encoded_correlation.png"), dpi=200)
plt.close()
print("  -> [IMAGE 11] Saved: image_11_full_encoded_correlation.png")

plt.figure(figsize=(12, 6))
top_15_features = [X.columns[i] for i in indices[:15]]
top_15_scores = importances[indices[:15]]
sns.barplot(x=top_15_scores, y=top_15_features, palette='viridis', hue=top_15_features, legend=False)
plt.axvline(x=0.005, color='red', linestyle='--', linewidth=1.5, label="Ambang Batas (0.005)")
plt.title("Embedded Method: Feature Importance (Top 15 Columns)", fontsize=14, weight='bold', pad=15)
plt.xlabel("MDI Feature Importance Score", fontsize=12)
plt.ylabel("Hardware Attribute", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_12_random_forest_importances.png"), dpi=200)
plt.close()
print("  -> [IMAGE 12] Saved: image_12_random_forest_importances.png")

print("\n  - Auditing performance of all feature subsets using baseline KNN...")
subsets = {
    "Filter Method": (df_filtered.drop(columns=['price']), df_filtered['price']),
    "Wrapper Method": (df_wrapper.drop(columns=['price']), df_wrapper['price']),
    "Embedded Method": (df_embedded.drop(columns=['price']), df_embedded['price'])
}

fs_results = []
for name, (sub_x, sub_y) in subsets.items():
    sub_x_scaled = StandardScaler().fit_transform(sub_x)
    X_tr, X_te, y_tr, y_te = train_test_split(sub_x_scaled, sub_y, train_size=15000, test_size=5000, random_state=42)
    knn_eval = KNeighborsRegressor(n_neighbors=5)
    knn_eval.fit(X_tr, y_tr)
    pred_y = knn_eval.predict(X_te)
    r2_score_val = r2_score(y_te, pred_y)
    mae_val = mean_absolute_error(y_te, pred_y)
    fs_results.append({
        "Method": name,
        "R2 Score": r2_score_val,
        "MAE": mae_val,
        "Feature Count": sub_x.shape[1]
    })
df_fs_comparison = pd.DataFrame(fs_results)
df_fs_comparison.to_csv(os.path.join(DIR_DATA, "feature_selection_comparison.csv"), index=False)
print(df_fs_comparison)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
sns.barplot(x='Method', y='R2 Score', data=df_fs_comparison, ax=ax1, palette='Blues', alpha=0.8, hue='Method', legend=False)
sns.lineplot(x='Method', y='MAE', data=df_fs_comparison, ax=ax2, color='red', marker='o', linewidth=2.5)
ax1.set_ylabel("R2 Score (Bar, Higher is Better)", fontsize=12)
ax2.set_ylabel("MAE (Line, Lower is Better)", fontsize=12)
ax1.set_ylim(0.7, 0.9)
plt.title("Performance Validation (KNN Test Set) by Feature Selection Methods", fontsize=14, weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_13_feature_selection_comparison.png"), dpi=200)
plt.close()
print("  -> [IMAGE 13] Saved: image_13_feature_selection_comparison.png")

winner_idx = df_fs_comparison['R2 Score'].idxmax()
winning_method_name = df_fs_comparison.loc[winner_idx, 'Method']
winning_features = list(subsets[winning_method_name][0].columns)
print(f"  -> Winning feature selection subset: '{winning_method_name}' with {len(winning_features)} features.")

plt.figure(figsize=(10, 6))
ridge_coefs = rfe_selector.estimator_.coef_
ridge_coef_series = pd.Series(ridge_coefs, index=wrapper_features).abs().sort_values(ascending=False).head(15)
sns.barplot(x=ridge_coef_series.values, y=ridge_coef_series.index, palette='crest', hue=ridge_coef_series.index, legend=False)
plt.title("Wrapper Method: Top 15 Feature Coefficient Weights (Ridge OLS)", fontsize=14, weight='bold', pad=15)
plt.xlabel("Absolute Coefficients Weight value", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_14_rfe_ridge_coefficients.png"), dpi=200)
plt.close()
print("  -> [IMAGE 14] Saved: image_14_rfe_ridge_coefficients.png")

report_3 = f"""================================================================================
  DASAR ANALITIK DATA (DAD) - MODULE REPORT: FEATURE SELECTION
  Syllabus Coverage: Topic 07 (Data Prep III - Feature Selection Methods)
================================================================================

1. FILTER METHOD HIGHLIGHTS:
   - Selected features count (Correlation >= 0.1): {len(filter_selected_features)}
   - Multicollinearity dropped columns: {cols_to_drop}
   - Final independent variables count: {len(filter_final_features)}

2. WRAPPER METHOD (RFE RIDGE) HIGHLIGHTS:
   - Top 15 RFE features kept: {wrapper_features}

3. EMBEDDED METHOD (RANDOM FOREST IMPORTANCE) HIGHLIGHTS:
   - Features exceeding threshold (>= 0.005): {embedded_features}

4. BENCHMARK COMPARISONS & DYNAMIC WINNER:
{df_fs_comparison.to_string()}
   - Selected Dataset Winner: '{winning_method_name}' (Features: {winning_features})
================================================================================
"""
with open(os.path.join(DIR_REPORTS, "handson_data_preparation_3_report.txt"), "w") as f:
    f.write(report_3)
print("  -> [REPORT] handson_data_preparation_3_report.txt written.")


# ==========================================================================
# PHASE 4: REGRESSION MODELLING & EVALUATION (Topic 09-12)
# ==========================================================================
print("\n" + "="*80)
print("  PHASE 4 | TOPICS 09-10-12: MODELLING & PERFORMANCE EVALUATION")
print("="*80)

X_model = subsets[winning_method_name][0]
y_model = subsets[winning_method_name][1]

X_model_scaled = StandardScaler().fit_transform(X_model)

X_train, X_test, y_train, y_test = train_test_split(X_model_scaled, y_model, train_size=15000, test_size=5000, random_state=42)
print(f"  - Model Train shape: {X_train.shape}")
print(f"  - Model Test shape: {X_test.shape}")

models = {
    "K-Nearest Neighbors (KNN)": KNeighborsRegressor(),
    "Multivariable Linear Regression": LinearRegression(),
    "Support Vector Regressor (SVR)": LinearSVR(max_iter=5000, random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
}

print("\n[STEP 1] Tuning KNN parameters using GridSearchCV...")
knn_grid = GridSearchCV(
    estimator=KNeighborsRegressor(),
    param_grid={'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    cv=5,
    n_jobs=-1
)
knn_grid.fit(X_train, y_train)
best_knn_params = knn_grid.best_params_
print(f"  -> Best KNN Hyperparameters found: {best_knn_params}")
models["K-Nearest Neighbors (KNN)"] = KNeighborsRegressor(**best_knn_params)

model_metrics = []
trained_models = {}

print("\n[STEP 2] Training all 5 Regressors...")
for name, clf in models.items():
    print(f"  - Training '{name}'...")
    clf.fit(X_train, y_train)
    trained_models[name] = clf

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    model_metrics.append({
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse
    })

df_metrics = pd.DataFrame(model_metrics)
df_metrics.to_csv(os.path.join(DIR_DATA, "model_performance_comparison.csv"), index=False)
print("\n  - ML Regressors Performance Comparison Matrix:")
print(df_metrics[['Model', 'Train R2', 'Test R2', 'Test MAE', 'Test RMSE']])

best_model_idx = df_metrics['Test R2'].idxmax()
best_model_name = df_metrics.loc[best_model_idx, 'Model']
best_model_clf = trained_models[best_model_name]
print(f"\n  -> Winning Model: '{best_model_name}' (Test R2: {df_metrics.loc[best_model_idx, 'Test R2']:.4f})")

print("\n[STEP 3] Plotting Phase 4 Visuals...")

plt.figure(figsize=(12, 6))
df_melt_r2 = df_metrics.melt(id_vars="Model", value_vars=["Train R2", "Test R2"], var_name="Dataset", value_name="R2")
sns.barplot(x="R2", y="Model", hue="Dataset", data=df_melt_r2, palette="muted")
plt.title("Model Accuracy Comparison (Train R-squared vs Test R-squared)", fontsize=14, weight='bold', pad=15)
plt.xlabel("R-squared score", fontsize=12)
plt.xlim(0.7, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_15_model_performance_comparison.png"), dpi=200)
plt.close()
print("  -> [IMAGE 15] Saved: image_15_model_performance_comparison.png")

plt.figure(figsize=(12, 6))
df_melt_mae = df_metrics.melt(id_vars="Model", value_vars=["Train MAE", "Test MAE"], var_name="Dataset", value_name="MAE")
sns.barplot(x="MAE", y="Model", hue="Dataset", data=df_melt_mae, palette="Set1")
plt.title("Model Error Comparison (Train MAE vs Test MAE)", fontsize=14, weight='bold', pad=15)
plt.xlabel("Mean Absolute Error ($)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_16_model_mae_comparison.png"), dpi=200)
plt.close()
print("  -> [IMAGE 16] Saved: image_16_model_mae_comparison.png")

y_best_pred = best_model_clf.predict(X_test)
residuals = y_test - y_best_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_best_pred, alpha=0.4, color='#3498db')
min_val = min(y_test.min(), y_best_pred.min())
max_val = max(y_test.max(), y_best_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Perfect Fit (y=x)")
plt.title(f"Best Model Diagnostic: Actual vs Predicted Price\n(Algorithm: {best_model_name})", fontsize=14, weight='bold', pad=15)
plt.xlabel("Actual Price ($)", fontsize=12)
plt.ylabel("Predicted Price ($)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_17_best_model_actual_vs_predicted.png"), dpi=200)
plt.close()
print("  -> [IMAGE 17] Saved: image_17_best_model_actual_vs_predicted.png")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_best_pred, y=residuals, alpha=0.4, color='#e67e22')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title(f"Best Model Residuals Plot: Error Scatter vs Prediction\n(Algorithm: {best_model_name})", fontsize=14, weight='bold', pad=15)
plt.xlabel("Predicted Price ($)", fontsize=12)
plt.ylabel("Residuals Error (Actual - Predicted) ($)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_18_best_model_residuals.png"), dpi=200)
plt.close()
print("  -> [IMAGE 18] Saved: image_18_best_model_residuals.png")

plt.figure(figsize=(10, 6))
dt_preds = trained_models["Decision Tree Regressor"].predict(X_test)
rf_preds = trained_models["Random Forest Regressor"].predict(X_test)
sns.scatterplot(x=dt_preds, y=rf_preds, alpha=0.3, color='#1abc9c')
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1)
plt.title("Bivariate Comparison: Decision Tree vs Random Forest Ensemble Predictions", fontsize=14, weight='bold', pad=15)
plt.xlabel("Decision Tree Predictions ($)", fontsize=12)
plt.ylabel("Random Forest Predictions ($)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_19_decision_tree_vs_rf.png"), dpi=200)
plt.close()
print("  -> [IMAGE 19] Saved: image_19_decision_tree_vs_rf.png")

plt.figure(figsize=(10, 6))
df_metrics['R2_Gap'] = (df_metrics['Train R2'] - df_metrics['Test R2']).abs()
sns.barplot(x="R2_Gap", y="Model", data=df_metrics, palette="plasma", hue="Model", legend=False)
plt.title("Model Overfitting Diagnosis: Absolute R2 Score Gap (Train vs Test)", fontsize=14, weight='bold', pad=15)
plt.xlabel("Absolute Score Difference (Lower is Better)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "image_20_model_learning_generalization.png"), dpi=200)
plt.close()
print("  -> [IMAGE 20] Saved: image_20_model_learning_generalization.png")

report_4 = f"""================================================================================
  DASAR ANALITIK DATA (DAD) - MODULE REPORT: REGRESSION MODELLING
  Syllabus Coverage: Topics 09, 10, 12 (Modelling, Supervised Learning, Evaluation)
================================================================================

1. DATASET CHARACTERISTICS FOR MODELLING:
   - Feature Selection Winner: '{winning_method_name}'
   - Attributes Count: {X_model.shape[1]}
   - Train Sample Size: {X_train.shape[0]}
   - Test Sample Size: {X_test.shape[0]}

2. K-NEAREST NEIGHBORS (KNN) GRIDSEARCH TUNING:
   - Best Parameters: {best_knn_params}

3. COMPREHENSIVE PERFORMANCE EVALUATION METRICS:
{df_metrics.to_string()}

4. WINNING PREDICTIVE MODEL SELECTION:
   - Model Name: '{best_model_name}'
   - Accuracy (R2): {df_metrics.loc[best_model_idx, 'Test R2']*100:.2f}%
   - Mean Absolute Error (MAE): ${df_metrics.loc[best_model_idx, 'Test MAE']:.2f}
   - Root Mean Squared Error (RMSE): ${df_metrics.loc[best_model_idx, 'Test RMSE']:.2f}
   - Overfitting Gap (R2 Train - R2 Test): {df_metrics.loc[best_model_idx, 'R2_Gap']:.4f}
================================================================================
"""
with open(os.path.join(DIR_REPORTS, "handson_modelling_report.txt"), "w") as f:
    f.write(report_4)
print("  -> [REPORT] handson_modelling_report.txt written.")

print("\n" + "="*80)
print("  PIPELINE COMPLETE. 20 figures, CSV datasets, and reports generated.")
print("="*80)
