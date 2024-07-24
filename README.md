# Machine Learning Algorithms

- [Linear Regression](./notebooks/linear-regression.ipynb)
- [Logistic Regression](./notebooks/logistic-regression.ipynb)
- Decision Trees
- Ensemble
  - Random Forest
  - XGBoost
- [K-Means](./notebooks/k-means.ipynb)
- Anomaly Detection
  - Gaussian Mixture Model (GMM)
- Recommender Systems
  - Collaborative Filtering
  - Content-Based Filtering
- Principle Component Analysis (PCA)
- Reinforcement Learning

## Usage

Install Poetry

```bash
pipx install poetry
```

Clone Repo

```bash
git clone https://github.com/wuihee/machine-learning-algorithms.git
cd machine-learning-algorithms
```

Install Dependencies

```bash
poetry install
```

Example Usage

```python
from mlagos import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)
pred = clf.pred(X_test)
```
