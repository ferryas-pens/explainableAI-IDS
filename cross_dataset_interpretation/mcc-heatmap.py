# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:16:00 2026

@author: hduser
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Baca file CSV
mcc = pd.read_csv("academic_table_mcc_matrix.csv")

# 2. Jadikan kolom 'Reference' sebagai index (kalau ada)
if "Reference" in mcc.columns:
    mcc = mcc.set_index("Reference")

# 3. Plot heatmap
plt.figure(figsize=(6, 4))
ax = sns.heatmap(
    mcc,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"label": "MCC (%)"},
)
ax.set_xlabel("Test dataset")
ax.set_ylabel("Train dataset")
plt.tight_layout()

# 4. Simpan ke PDF
plt.savefig("mcc_matrix_heatmap.pdf")
plt.close()
print("Saved to mcc_matrix_heatmap.pdf")