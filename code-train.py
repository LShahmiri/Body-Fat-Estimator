import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("bodyfat.csv")

target_col = "BodyFat"

features = ["Density", "Abdomen", "Chest", "Weight", "Hip"]

X = df[features]
y = df[target_col]

numeric_features = X.columns.tolist()


# -----------------------------
# 2. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 3. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)


# -----------------------------
# 4. Candidate models
# -----------------------------
pipelines = {
    "LinearRegression": Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression())
    ]),
    
    "DecisionTree": Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),
    
    "RandomForest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]),
    
    "GradientBoosting": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])
}


param_grids = {
    "LinearRegression": {},
    
    "DecisionTree": {
        "model__max_depth": [3,5,7,None],
        "model__min_samples_split":[2,5,10]
    },
    
    "RandomForest": {
        "model__n_estimators":[100,200],
        "model__max_depth":[None,5,10]
    },
    
    "GradientBoosting":{
        "model__n_estimators":[100,200],
        "model__learning_rate":[0.05,0.1]
    }
}


best_models = {}
results = []


# -----------------------------
# 5. Hyperparameter tuning
# -----------------------------
for name, pipe in pipelines.items():

    print(f"\nTuning {name}...")

    grid = GridSearchCV(
        pipe,
        param_grid=param_grids[name],
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_

    preds = grid.best_estimator_.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# -----------------------------
# 6. Compare models
# -----------------------------
results_df = pd.DataFrame(results).sort_values("RMSE")

print("\nModel comparison:")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = best_models[best_model_name]

print(f"\nBest model: {best_model_name}")


# -----------------------------
# 7. Cross validation
# -----------------------------
cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

cv_rmse = -cv_scores

print("\nCross validation RMSE:", cv_rmse)
print("Mean CV RMSE:", cv_rmse.mean())


# -----------------------------
# 8. Save model
# -----------------------------
with open("bodyfatmodel.pkl","wb") as f:
    pickle.dump(best_model,f)

print("\nModel saved as bodyfatmodel.pkl")