from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/raw/dataset.csv"
if not DATA.exists():
    DATA.parent.mkdir(parents=True,exist_ok=True); df=pd.read_csv("https://raw.githubusercontent.com/tanishq21/Mall-Customers/main/Mall_Customers.csv")
else: df=pd.read_csv(DATA)
cols=[c for c in df.columns if c.lower() in {"age","annual income (k$)","annual_income","spending score (1-100)","spending_score"}]
if len(cols)<2: raise ValueError(f"Expected customer behavior columns: {df.columns.tolist()}")
X=df[cols].copy(); X.columns=[c.lower().replace(" ","_").replace("(","").replace(")","").replace("$","") for c in X.columns]
scaled=StandardScaler().fit_transform(X); scores={}
for k in range(2,9): scores[k]=silhouette_score(scaled,KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(scaled))
best_k=max(scores,key=scores.get); labels=KMeans(n_clusters=best_k,n_init=10,random_state=42).fit_predict(scaled)
df["cluster"]=labels; df.to_csv(ROOT/"data/clustered_customers.csv",index=False)
z=PCA(n_components=2,random_state=42).fit_transform(scaled)
plt.figure(figsize=(8,5)); plt.scatter(z[:,0],z[:,1],c=labels); plt.title(f"Customer Segments (K={best_k})"); plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.tight_layout(); plt.savefig(ROOT/"customer_segments.png",dpi=160); plt.close()
print("Silhouette scores:",scores); print("Selected k:",best_k)
