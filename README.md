# Machine Learning Algorithms

1. [Linear Regression](./notebooks/linear-regression.ipynb)
2. [Logistic Regression](./notebooks/logistic-regression.ipynb)
3. Decision Trees
4. Random Forest
5. XGBoost
6. K-Means

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
