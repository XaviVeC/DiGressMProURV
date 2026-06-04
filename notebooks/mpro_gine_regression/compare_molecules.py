import pandas as pd

FILE_A = "artifacts/external_predictions_14Apr2026.csv"
FILE_B = "artifacts/external_predictions_21Apr2026.csv"

df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)

smiles_a = set(df_a["SMILES"])
smiles_b = set(df_b["SMILES"])

common = smiles_a & smiles_b

print(f"Molecules in 14Apr2026: {len(smiles_a)}")
print(f"Molecules in 21Apr2026: {len(smiles_b)}")
print(f"Repeated (exact SMILES match): {len(common)}")

if common:
    print("\nRepeated SMILES:")
    for smi in sorted(common):
        pic50_a = df_a.loc[df_a["SMILES"] == smi, "pIC50"].values[0]
        pic50_b = df_b.loc[df_b["SMILES"] == smi, "pIC50"].values[0]
        print(f"  {smi}  |  14Apr pIC50={pic50_a:.4f}  |  21Apr pIC50={pic50_b:.4f}")
else:
    print("\nNo repeated molecules found.")
