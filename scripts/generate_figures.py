"""
Script de réalisation de l'analyse exploratoire et génération des figures
Palmer Penguins - Laboratoire de Recherche Écologique
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration esthétique globale
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'figure.autolayout': False
})

# 1. Chargement des données
print("Chargement des données...")
df = sns.load_dataset('penguins')

# Sauvegarde locale pour autonomie
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
df.to_csv("data/penguins.csv", index=False)
print("Données sauvegardées sous data/penguins.csv")

# Palette de couleurs constante pour les espèces
species_colors = {
    'Adelie': '#FF8C00',    # Orange foncé
    'Chinstrap': '#9932CC', # Violet / Pourpre
    'Gentoo': '#008080'     # Sarcelle / Bleu-vert
}

# FIGURE 1: Histogrammes des variables numériques
print("Génération Figure 1 : Histogrammes...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
col_labels = {
    'bill_length_mm': 'Longueur du bec (mm)',
    'bill_depth_mm': 'Profondeur du bec (mm)',
    'flipper_length_mm': 'Longueur des nageoires (mm)',
    'body_mass_g': 'Masse corporelle (g)'
}

for ax, col in zip(axes.flatten(), num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#2b5c8f', bins=25, edgecolor='black', alpha=0.6)
    mean_val = df[col].mean()
    med_val = df[col].median()
    ax.axvline(mean_val, color='#d9534f', linestyle='--', linewidth=2, label=f'Moyenne : {mean_val:.1f}')
    ax.axvline(med_val, color='#5cb85c', linestyle=':', linewidth=2.5, label=f'Médiane : {med_val:.1f}')
    ax.set_title(f"Distribution : {col_labels[col]}", fontweight='bold')
    ax.set_xlabel(col_labels[col])
    ax.set_ylabel("Effectif (Fréquence)")
    ax.legend(loc='upper right')

plt.suptitle("Distribution des caractéristiques biométriques (Ensemble de la population)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/01_histograms_distributions.png", dpi=300)
plt.close()

# FIGURE 2: Boxplots comparatifs par espèce
print("Génération Figure 2 : Boxplots par espèce...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flatten(), num_cols):
    sns.boxplot(data=df, x='species', y=col, ax=ax, palette=species_colors, hue='species', legend=False, width=0.5, flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
    sns.stripplot(data=df, x='species', y=col, ax=ax, color='black', alpha=0.25, jitter=0.2, size=4)
    ax.set_title(f"Dispersion : {col_labels[col]} par Espèce", fontweight='bold')
    ax.set_xlabel("Espèce")
    ax.set_ylabel(col_labels[col])

plt.suptitle("Comparaison morphologique inter-espèces (Boxplots & Points individuels)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/02_boxplots_by_species.png", dpi=300)
plt.close()

# FIGURE 3: Nuage de points Bec (Longueur vs Profondeur) - Paradoxe de Simpson
print("Génération Figure 3 : Nuage de points Bec (Simpson)...")
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='species',
    style='species',
    palette=species_colors,
    s=85,
    alpha=0.85,
    edgecolor='black'
)
# Droites de tendance par espèce
for sp, color in species_colors.items():
    sub = df[df['species'] == sp].dropna(subset=['bill_length_mm', 'bill_depth_mm'])
    sns.regplot(
        data=sub,
        x='bill_length_mm',
        y='bill_depth_mm',
        scatter=False,
        ax=plt.gca(),
        color=color,
        line_kws={'linestyle': '-', 'linewidth': 2}
    )

# Tendance globale (pointillé noir)
sub_all = df.dropna(subset=['bill_length_mm', 'bill_depth_mm'])
sns.regplot(
    data=sub_all,
    x='bill_length_mm',
    y='bill_depth_mm',
    scatter=False,
    ax=plt.gca(),
    color='grey',
    line_kws={'linestyle': '--', 'linewidth': 1.8, 'label': 'Tendance globale (toutes espèces)'}
)

plt.title("Relation Longueur vs Profondeur du Bec : Illustration du Paradoxe de Simpson", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Longueur du bec (mm)", fontsize=12)
plt.ylabel("Profondeur du bec (mm)", fontsize=12)
plt.legend(title="Espèces & Tendances", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures/03_scatter_bill_dimensions.png", dpi=300)
plt.close()

# FIGURE 4: Matrice de corrélation
print("Génération Figure 4 : Matrice de corrélation...")
plt.figure(figsize=(9, 7))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=1.5,
    cbar_kws={"shrink": 0.8, "label": "Coefficient de corrélation linéaire de Pearson (r)"},
    xticklabels=[col_labels[c] for c in num_cols],
    yticklabels=[col_labels[c] for c in num_cols]
)
plt.title("Matrice de Corrélation Linéaire des Variables Numériques", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/04_correlation_matrix.png", dpi=300)
plt.close()

# FIGURE 5: Comparaison de la masse corporelle entre espèces
print("Génération Figure 5 : Masse corporelle...")
plt.figure(figsize=(10, 6.5))
ax = sns.violinplot(
    data=df,
    x='species',
    y='body_mass_g',
    palette=species_colors,
    hue='species',
    legend=False,
    inner='quartile',
    cut=0
)
sns.stripplot(
    data=df,
    x='species',
    y='body_mass_g',
    color='white',
    edgecolor='gray',
    linewidth=0.5,
    alpha=0.6,
    jitter=0.2,
    size=5
)
# Ajout des moyennes textuelles
means = df.groupby('species')['body_mass_g'].mean()
counts = df.groupby('species')['body_mass_g'].count()
for i, sp in enumerate(means.index):
    m = means[sp]
    n = counts[sp]
    ax.text(i, m + 180, f"Moyenne: {m:.0f} g\n(n={n})", horizontalalignment='center', size=11, color='darkblue', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='navy'))

plt.title("Distribution comparée de la Masse Corporelle par Espèce", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Espèce de manchot", fontsize=12)
plt.ylabel("Masse corporelle (g)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/05_body_mass_by_species.png", dpi=300)
plt.close()

# FIGURE 6: Répartition par île et espèce (Contexte d'échantillonnage)
print("Génération Figure 6 : Répartition géographique...")
plt.figure(figsize=(9, 6))
ct = pd.crosstab(df['island'], df['species'])
ax = ct.plot(kind='bar', stacked=False, color=[species_colors[s] for s in ct.columns], figsize=(9, 6), edgecolor='black', width=0.7)
plt.title("Répartition des effectifs d'espèces par île de l'Archipel Palmer", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Île d'observation", fontsize=12)
plt.ylabel("Nombre d'individus recensés", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Espèce")
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height() + 1.5), ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.ylim(0, max([p.get_height() for p in ax.patches]) + 15)
plt.tight_layout()
plt.savefig("figures/06_species_by_island.png", dpi=300)
plt.close()

# FIGURE 7: Nageoires vs Masse corporelle
print("Génération Figure 7 : Nageoires vs Masse corporelle...")
plt.figure(figsize=(10, 6.5))
sns.scatterplot(
    data=df,
    x='flipper_length_mm',
    y='body_mass_g',
    hue='species',
    palette=species_colors,
    s=80,
    alpha=0.8,
    edgecolor='black'
)
sns.regplot(
    data=df.dropna(subset=['flipper_length_mm', 'body_mass_g']),
    x='flipper_length_mm',
    y='body_mass_g',
    scatter=False,
    color='black',
    line_kws={'linestyle': '--', 'linewidth': 1.8, 'label': f'Régression globale (r = {df["flipper_length_mm"].corr(df["body_mass_g"]):.2f})'}
)
plt.title("Allométrie : Longueur des Nageoires vs Masse Corporelle", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Longueur des nageoires (mm)", fontsize=12)
plt.ylabel("Masse corporelle (g)", fontsize=12)
plt.legend(title="Espèce & Régression", loc='upper left')
plt.tight_layout()
plt.savefig("figures/07_flipper_vs_bodymass.png", dpi=300)
plt.close()

print("Toutes les figures ont été générées avec succès dans figures/")
