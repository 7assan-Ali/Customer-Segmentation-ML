from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; df=pd.read_csv(ROOT/"data/raw/dataset.csv")
cols=[c for c in df.columns if c.lower() in {"age","annual income (k$)","annual_income","spending score (1-100)","spending_score"}]
X=df[cols]; labels=KMeans(n_clusters=5,n_init=10,random_state=42).fit_predict(StandardScaler().fit_transform(X))
print(pd.Series(labels,name="cluster").value_counts().sort_index())
