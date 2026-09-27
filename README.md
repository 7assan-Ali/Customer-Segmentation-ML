# Customer Segmentation with K-Means

Unsupervised learning project that groups customers using age, annual income, and spending behavior.

## Workflow
- Standardize numeric features
- Test K values from 2 to 8
- Select K using silhouette score
- Fit K-Means
- Use PCA for visualization
- Export clustered customers

## Dataset
https://raw.githubusercontent.com/tanishq21/Mall-Customers/main/Mall_Customers.csv

There is no target variable because this is unsupervised learning.

## Run
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

Outputs: `data/clustered_customers.csv` and `customer_segments.png`.

## Author
Hassan Ali — Computer Science student focused on Machine Learning and AI Engineering.
