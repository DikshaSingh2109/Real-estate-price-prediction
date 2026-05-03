"""
PROJECT 6: Real Estate Price Prediction — Regression Modelling
Author: Diksha Singh
Tools: Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
Dataset: Simulated Ames Housing-style dataset (2,900+ records)
Models: Linear Regression vs Random Forest — with comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. GENERATE REALISTIC HOUSING DATASET ────────────────────────────────────
n = 2930

neighborhoods = {
    'NridgHt':  {'premium': 1.45, 'label': 'Northridge Heights'},
    'NoRidge':  {'premium': 1.40, 'label': 'Northridge'},
    'StoneBr':  {'premium': 1.38, 'label': 'Stone Brook'},
    'Timber':   {'premium': 1.20, 'label': 'Timberland'},
    'Veenker':  {'premium': 1.15, 'label': 'Veenker'},
    'Somerst':  {'premium': 1.10, 'label': 'Somerset'},
    'ClearCr':  {'premium': 1.05, 'label': 'Clear Creek'},
    'CollgCr':  {'premium': 1.00, 'label': 'College Creek'},
    'Crawfor':  {'premium': 0.98, 'label': 'Crawford'},
    'Gilbert':  {'premium': 0.95, 'label': 'Gilbert'},
    'NWAmes':   {'premium': 0.90, 'label': 'Northwest Ames'},
    'Mitchel':  {'premium': 0.85, 'label': 'Mitchell'},
    'NAmes':    {'premium': 0.80, 'label': 'North Ames'},
    'BrkSide':  {'premium': 0.72, 'label': 'Brookside'},
    'IDOTRR':   {'premium': 0.65, 'label': 'Iowa DOT RR'},
}
neigh_list = list(neighborhoods.keys())
neigh_probs = [0.08, 0.06, 0.06, 0.07, 0.05, 0.08, 0.06, 0.10,
               0.07, 0.07, 0.07, 0.06, 0.07, 0.05, 0.05]

neighborhood   = np.random.choice(neigh_list, size=n, p=neigh_probs)
year_built     = np.random.randint(1900, 2011, size=n)
year_remod     = np.array([max(yb, np.random.randint(yb, 2011))
                            for yb in year_built])
overall_qual   = np.random.randint(1, 11, size=n)
overall_cond   = np.random.randint(1, 10, size=n)
gr_liv_area    = np.random.randint(500, 4500, size=n)
total_bsmt_sf  = np.maximum(0, gr_liv_area * np.random.uniform(0.3, 0.85, n)
                             + np.random.normal(0, 150, n)).astype(int)
garage_cars    = np.random.choice([0, 1, 2, 3, 4], size=n,
                                   p=[0.05, 0.15, 0.55, 0.20, 0.05])
full_bath      = np.random.choice([0, 1, 2, 3], size=n, p=[0.02, 0.25, 0.60, 0.13])
bedroom_abvgr  = np.random.choice([1, 2, 3, 4, 5], size=n,
                                   p=[0.03, 0.18, 0.50, 0.23, 0.06])
fireplaces     = np.random.choice([0, 1, 2], size=n, p=[0.45, 0.45, 0.10])
lot_area       = np.random.randint(1500, 25000, size=n)
house_age      = 2010 - year_built
years_since_remod = 2010 - year_remod

bldg_type      = np.random.choice(['1Fam','2fmCon','Duplex','TwnhsE','Twnhs'],
                                   size=n, p=[0.75, 0.05, 0.05, 0.10, 0.05])
sale_type      = np.random.choice(['WD','New','COD','ConLD','Oth'],
                                   size=n, p=[0.80, 0.08, 0.06, 0.04, 0.02])

# ── Price construction ──
base_price = 100000
neigh_premium  = np.array([neighborhoods[nb]['premium'] for nb in neighborhood])
price = (
    base_price
    + gr_liv_area    * 55
    + total_bsmt_sf  * 18
    + overall_qual   * 12000
    + overall_cond   * 2500
    + garage_cars    * 8000
    + full_bath      * 6000
    + fireplaces     * 5000
    + bedroom_abvgr  * 3000
    + lot_area       * 0.8
    - house_age      * 400
    - years_since_remod * 200
) * neigh_premium

new_mask = (sale_type == 'New')
price[new_mask] *= np.random.uniform(1.05, 1.15, new_mask.sum())
price += np.random.normal(0, 12000, n)
price = np.maximum(price, 50000).astype(int)

df = pd.DataFrame({
    'Neighborhood':     neighborhood,
    'Year_Built':       year_built,
    'Year_Remod_Add':   year_remod,
    'Overall_Qual':     overall_qual,
    'Overall_Cond':     overall_cond,
    'Gr_Liv_Area':      gr_liv_area,
    'Total_Bsmt_SF':    total_bsmt_sf,
    'Garage_Cars':      garage_cars,
    'Full_Bath':        full_bath,
    'Bedroom_AbvGr':    bedroom_abvgr,
    'Fireplaces':       fireplaces,
    'Lot_Area':         lot_area,
    'House_Age':        house_age,
    'Years_Since_Remod':years_since_remod,
    'Bldg_Type':        bldg_type,
    'Sale_Type':        sale_type,
    'Sale_Price':       price
})

df.to_csv('/home/claude/projects/real_estate/housing_data.csv', index=False)
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Price range: ${df['Sale_Price'].min():,} — ${df['Sale_Price'].max():,}")
print(f"Mean price: ${df['Sale_Price'].mean():,.0f}")

# ── 2. EDA ───────────────────────────────────────────────────────────────────
print("\n========== EDA: DATA SUMMARY ==========")
print(df.describe()[['Sale_Price','Gr_Liv_Area','Overall_Qual','House_Age']].round(1))

print("\n========== TOP 5 NEIGHBORHOODS BY MEDIAN PRICE ==========")
neigh_stats = df.groupby('Neighborhood')['Sale_Price'].agg(
    Median='median', Mean='mean', Count='count').sort_values('Median', ascending=False)
neigh_stats['Median'] = neigh_stats['Median'].apply(lambda x: f'${x:,.0f}')
neigh_stats['Mean']   = neigh_stats['Mean'].apply(lambda x: f'${x:,.0f}')
print(neigh_stats.head(5).to_string())

print("\n========== CORRELATION WITH SALE PRICE ==========")
num_cols = ['Gr_Liv_Area','Total_Bsmt_SF','Overall_Qual','Overall_Cond',
            'Garage_Cars','Full_Bath','Fireplaces','Lot_Area',
            'House_Age','Years_Since_Remod','Sale_Price']
corr_price = df[num_cols].corr()['Sale_Price'].drop('Sale_Price').sort_values(ascending=False)
print(corr_price.round(3).to_string())

# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
df['Total_SF']          = df['Gr_Liv_Area'] + df['Total_Bsmt_SF']
df['Qual_x_Area']       = df['Overall_Qual'] * df['Gr_Liv_Area']
df['Has_Garage']        = (df['Garage_Cars'] > 0).astype(int)
df['Has_Fireplace']     = (df['Fireplaces'] > 0).astype(int)
df['Is_New']            = (df['Sale_Type'] == 'New').astype(int)
df['Price_Per_SqFt']    = df['Sale_Price'] / df['Gr_Liv_Area']

le = LabelEncoder()
for col in ['Neighborhood', 'Bldg_Type', 'Sale_Type']:
    df[col + '_enc'] = le.fit_transform(df[col])

features = ['Gr_Liv_Area', 'Total_Bsmt_SF', 'Overall_Qual', 'Overall_Cond',
            'Garage_Cars', 'Full_Bath', 'Bedroom_AbvGr', 'Fireplaces',
            'Lot_Area', 'House_Age', 'Years_Since_Remod', 'Total_SF',
            'Qual_x_Area', 'Has_Garage', 'Has_Fireplace', 'Is_New',
            'Neighborhood_enc', 'Bldg_Type_enc']

X = df[features]
y = df['Sale_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 4. MODEL TRAINING ─────────────────────────────────────────────────────────
models = {
    'Linear Regression':   LinearRegression(),
    'Ridge Regression':    Ridge(alpha=10),
    'Random Forest':       RandomForestRegressor(n_estimators=150, max_depth=15,
                                                  min_samples_split=5, random_state=42),
    'Gradient Boosting':   GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                                      max_depth=4, random_state=42),
}

results = {}
predictions = {}
print("\n========== MODEL PERFORMANCE ==========")
print(f"{'Model':<22} {'MAE':>12} {'RMSE':>12} {'R² Score':>10} {'CV R²':>10}")
print("-" * 70)

for name, model in models.items():
    if 'Regression' in name:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv_r2 = cv_scores.mean()

    results[name]     = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv_r2}
    predictions[name] = y_pred
    print(f"{name:<22} ${mae:>10,.0f} ${rmse:>10,.0f} {r2:>10.4f} {cv_r2:>10.4f}")

best_model_name = max(results, key=lambda x: results[x]['R2'])
print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

# ── 5. VISUALISATIONS ─────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# ── FIGURE 1: EDA Dashboard ───────────────────────────────────────────────
fig1 = plt.figure(figsize=(18, 12))
fig1.suptitle('Real Estate Price Prediction — EDA Dashboard\nDiksha Singh | Python (Pandas, Scikit-learn)',
              fontsize=15, fontweight='bold', y=0.98)
gs1 = GridSpec(2, 3, figure=fig1, hspace=0.40, wspace=0.35)

# Price distribution
ax = fig1.add_subplot(gs1[0, 0])
ax.hist(df['Sale_Price'] / 1000, bins=40, color='#2E86AB', edgecolor='white', alpha=0.85)
ax.axvline(df['Sale_Price'].mean() / 1000, color='#C73E1D', linewidth=2,
           linestyle='--', label=f"Mean: ${df['Sale_Price'].mean()/1000:.0f}K")
ax.axvline(df['Sale_Price'].median() / 1000, color='#F18F01', linewidth=2,
           linestyle='--', label=f"Median: ${df['Sale_Price'].median()/1000:.0f}K")
ax.set_title('Sale Price Distribution', fontweight='bold')
ax.set_xlabel('Sale Price ($000s)'); ax.set_ylabel('Frequency')
ax.legend(fontsize=9)

# Price vs Living Area
ax2 = fig1.add_subplot(gs1[0, 1])
qual_colors = {q: plt.cm.RdYlGn(q / 10) for q in range(1, 11)}
for qual in range(3, 11, 2):
    mask = df['Overall_Qual'] == qual
    ax2.scatter(df[mask]['Gr_Liv_Area'], df[mask]['Sale_Price'] / 1000,
                alpha=0.4, s=15, label=f'Qual {qual}', color=qual_colors[qual])
ax2.set_title('Living Area vs Sale Price\n(colored by Overall Quality)', fontweight='bold')
ax2.set_xlabel('Above Ground Living Area (sq ft)')
ax2.set_ylabel('Sale Price ($000s)')
ax2.legend(fontsize=7, ncol=2)

# Quality vs Price boxplot
ax3 = fig1.add_subplot(gs1[0, 2])
qual_data = [df[df['Overall_Qual'] == q]['Sale_Price'].values / 1000
             for q in range(3, 11)]
bp = ax3.boxplot(qual_data, patch_artist=True, medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], plt.cm.RdYlGn(np.linspace(0.2, 0.9, 8))):
    patch.set_facecolor(color)
ax3.set_xticklabels(range(3, 11))
ax3.set_title('Sale Price by Overall Quality', fontweight='bold')
ax3.set_xlabel('Overall Quality (1–10)'); ax3.set_ylabel('Sale Price ($000s)')

# Neighborhood median prices (horizontal bar)
ax4 = fig1.add_subplot(gs1[1, 0:2])
neigh_med = df.groupby('Neighborhood')['Sale_Price'].median().sort_values(ascending=True) / 1000
colors_bar = ['#C73E1D' if v > 200 else '#2E86AB' for v in neigh_med.values]
bars = ax4.barh(neigh_med.index, neigh_med.values, color=colors_bar, edgecolor='white', height=0.7)
for bar, val in zip(bars, neigh_med.values):
    ax4.text(val + 2, bar.get_y() + bar.get_height()/2,
             f'${val:.0f}K', va='center', fontsize=8)
ax4.set_title('Median Sale Price by Neighborhood', fontweight='bold')
ax4.set_xlabel('Median Sale Price ($000s)')
ax4.axvline(df['Sale_Price'].median() / 1000, color='#F18F01',
            linewidth=1.5, linestyle='--', label='Overall Median')
ax4.legend(fontsize=9)

# Correlation heatmap
ax5 = fig1.add_subplot(gs1[1, 2])
corr_cols = ['Sale_Price', 'Gr_Liv_Area', 'Total_Bsmt_SF', 'Overall_Qual',
             'Garage_Cars', 'Full_Bath', 'House_Age']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, ax=ax5, linewidths=0.5, annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8})
ax5.set_title('Feature Correlation Heatmap', fontweight='bold')
ax5.tick_params(axis='x', rotation=45)

plt.savefig('/home/claude/projects/real_estate/figure1_eda_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1 (EDA) saved.")

# ── FIGURE 2: Model Performance Dashboard ─────────────────────────────────
fig2 = plt.figure(figsize=(18, 11))
fig2.suptitle('Real Estate Price Prediction — Model Performance Comparison\nLinear Regression vs Ridge vs Random Forest vs Gradient Boosting',
              fontsize=14, fontweight='bold', y=0.98)
gs2 = GridSpec(2, 3, figure=fig2, hspace=0.42, wspace=0.35)

# R² Comparison bar chart
ax = fig2.add_subplot(gs2[0, 0])
model_names_short = ['Linear\nRegression', 'Ridge\nRegression', 'Random\nForest', 'Gradient\nBoosting']
r2_vals  = [results[m]['R2']  for m in models]
cv_vals  = [results[m]['CV_R2'] for m in models]
x = np.arange(len(model_names_short))
w = 0.38
bars1 = ax.bar(x - w/2, r2_vals, w, label='Test R²',    color='#2E86AB', edgecolor='white')
bars2 = ax.bar(x + w/2, cv_vals,  w, label='CV R² (5-fold)', color='#F18F01', edgecolor='white')
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_title('R² Score Comparison', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(model_names_short, fontsize=9)
ax.set_ylabel('R² Score'); ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.axhline(0.9, color='green', linewidth=1, linestyle=':', alpha=0.7)
ax.text(3.5, 0.91, 'Target: 0.90', fontsize=8, color='green', ha='right')

# MAE Comparison
ax2 = fig2.add_subplot(gs2[0, 1])
mae_vals = [results[m]['MAE'] / 1000 for m in models]
bars = ax2.bar(model_names_short, mae_vals, color=COLORS, edgecolor='white', width=0.6)
for bar, val in zip(bars, mae_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'${val:.1f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_title('Mean Absolute Error (MAE)', fontweight='bold')
ax2.set_ylabel('MAE ($000s)')
ax2.tick_params(axis='x', labelsize=9)

# Best model: Actual vs Predicted
ax3 = fig2.add_subplot(gs2[0, 2])
best_pred = predictions[best_model_name]
ax3.scatter(y_test / 1000, best_pred / 1000, alpha=0.35, s=18,
            color='#2E86AB', edgecolors='white', linewidth=0.3)
mn = min(y_test.min(), best_pred.min()) / 1000
mx = max(y_test.max(), best_pred.max()) / 1000
ax3.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
ax3.set_title(f'{best_model_name}\nActual vs Predicted ($000s)', fontweight='bold')
ax3.set_xlabel('Actual Price ($000s)'); ax3.set_ylabel('Predicted Price ($000s)')
ax3.text(0.05, 0.92, f"R² = {results[best_model_name]['R2']:.4f}",
         transform=ax3.transAxes, fontsize=11, fontweight='bold', color='#2E86AB')
ax3.legend(fontsize=9)

# Residuals plot
ax4 = fig2.add_subplot(gs2[1, 0])
residuals = (y_test - best_pred) / 1000
ax4.scatter(best_pred / 1000, residuals, alpha=0.35, s=18,
            color='#A23B72', edgecolors='white', linewidth=0.3)
ax4.axhline(0, color='red', linewidth=2, linestyle='--')
ax4.set_title(f'Residuals Plot — {best_model_name}', fontweight='bold')
ax4.set_xlabel('Predicted Price ($000s)'); ax4.set_ylabel('Residuals ($000s)')

# Feature Importance (Random Forest)
ax5 = fig2.add_subplot(gs2[1, 1:])
rf_model = models['Random Forest']
feat_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True).tail(12)
colors_fi = ['#C73E1D' if v > 0.08 else '#2E86AB' for v in feat_imp.values]
feat_imp.plot(kind='barh', ax=ax5, color=colors_fi, edgecolor='white')
ax5.set_title('Top 12 Feature Importances — Random Forest', fontweight='bold')
ax5.set_xlabel('Importance Score')
feature_labels = {
    'Gr_Liv_Area': 'Above Ground Living Area',
    'Total_SF': 'Total Square Footage',
    'Overall_Qual': 'Overall Quality',
    'Qual_x_Area': 'Quality × Area (Interaction)',
    'Total_Bsmt_SF': 'Basement Area',
    'Neighborhood_enc': 'Neighborhood',
    'House_Age': 'House Age',
    'Lot_Area': 'Lot Area',
    'Years_Since_Remod': 'Years Since Remodel',
    'Garage_Cars': 'Garage Capacity',
    'Full_Bath': 'Full Bathrooms',
    'Overall_Cond': 'Overall Condition',
    'Bedroom_AbvGr': 'Bedrooms',
    'Fireplaces': 'Fireplaces',
    'Has_Garage': 'Has Garage',
    'Has_Fireplace': 'Has Fireplace',
    'Is_New': 'Is New Sale',
    'Bldg_Type_enc': 'Building Type',
}
ax5.set_yticklabels([feature_labels.get(f, f) for f in feat_imp.index], fontsize=9)
for i, (val, label) in enumerate(zip(feat_imp.values, feat_imp.index)):
    ax5.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

plt.savefig('/home/claude/projects/real_estate/figure2_model_performance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 (Model Performance) saved.")

# ── FIGURE 3: Business Insights ────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle('Real Estate: Business Insights for Investment Decision-Making',
              fontweight='bold', fontsize=13)

# Price per sqft by neighborhood
ax = axes3[0]
ppsf = df.groupby('Neighborhood')['Price_Per_SqFt'].median().sort_values(ascending=False)
colors_ppsf = ['#C73E1D' if i < 3 else '#2E86AB' for i in range(len(ppsf))]
bars = ax.bar(ppsf.index, ppsf.values, color=colors_ppsf, edgecolor='white', width=0.7)
for bar, val in zip(bars, ppsf.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'${val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('Median Price per Sq Ft by Neighborhood\n(Red = Premium Zones)', fontweight='bold')
ax.set_xlabel('Neighborhood'); ax.set_ylabel('Price per Sq Ft ($)')
ax.tick_params(axis='x', rotation=45)

# ROI: Quality score vs price premium
ax2 = axes3[1]
qual_median = df.groupby('Overall_Qual')['Sale_Price'].median() / 1000
ax2.plot(qual_median.index, qual_median.values, marker='o', color='#2E86AB',
         linewidth=2.5, markersize=8, markerfacecolor='#F18F01', markeredgecolor='#2E86AB')
for x_val, y_val in zip(qual_median.index, qual_median.values):
    ax2.text(x_val, y_val + 5, f'${y_val:.0f}K', ha='center', fontsize=8.5, fontweight='bold')
ax2.fill_between(qual_median.index, qual_median.values, alpha=0.12, color='#2E86AB')
ax2.set_title('Overall Quality Score vs Median Sale Price\n(Investment Insight)', fontweight='bold')
ax2.set_xlabel('Overall Quality (1–10)'); ax2.set_ylabel('Median Sale Price ($000s)')
ax2.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig('/home/claude/projects/real_estate/figure3_business_insights.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 (Business Insights) saved.")

# ── 6. BUSINESS SUMMARY ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("           KEY INSIGHTS — BUSINESS SUMMARY")
print("="*65)
best_r2 = results[best_model_name]['R2']
best_mae = results[best_model_name]['MAE']
lr_r2   = results['Linear Regression']['R2']
print(f"\n1. Best Model: {best_model_name}")
print(f"   R² Score: {best_r2:.4f} — explains {best_r2*100:.1f}% of price variance")
print(f"   MAE: ${best_mae:,.0f} — avg prediction error")
print(f"\n2. Linear Regression R² = {lr_r2:.4f} vs {best_model_name} R² = {best_r2:.4f}")
print(f"   → Ensemble models outperform linear by {(best_r2-lr_r2)*100:.1f} percentage points")
top_features = pd.Series(models['Random Forest'].feature_importances_,
                          index=features).nlargest(3)
print(f"\n3. Top 3 Price Drivers:")
for feat, imp in top_features.items():
    print(f"   • {feature_labels.get(feat, feat)}: {imp:.3f} importance score")
top_neigh = df.groupby('Neighborhood')['Sale_Price'].median().nlargest(3)
print(f"\n4. Premium Neighborhoods (highest median price):")
for nb, price_v in top_neigh.items():
    print(f"   • {nb}: ${price_v:,.0f}")
print(f"\n5. Each quality point increase → approx. "
      f"${df.groupby('Overall_Qual')['Sale_Price'].median().diff().mean():,.0f} price increase")

print("\n✅ All outputs saved to: /home/claude/projects/real_estate/")
print("   • housing_data.csv")
print("   • figure1_eda_dashboard.png")
print("   • figure2_model_performance.png")
print("   • figure3_business_insights.png")
