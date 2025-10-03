import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris_dt")

# Data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model/params
max_depth = 15
random_state = 42

# with mlflow.start_run(run_name="dt_1"):  # run 1
with mlflow.start_run(run_name="dt_2"):  # run 2
    # log params
    mlflow.log_params(
        {
            "max_depth": max_depth,
            "random_state": random_state,
            "dataset": "iris",
        }
    )

    # train
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)

    # metrics
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    print("accuracy", acc)

    # confusion matrix as a figure (cleaner than saving then logging)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix")
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()

    # signature + input example
    signature = infer_signature(X_train, dt.predict(X_train))  # input and output schema
    input_example = X_train[:5]  # first 5 rows

    # log the model
    mlflow.sklearn.log_model(
        sk_model=dt,
        name="decision_tree_model",
        signature=signature,
        input_example=input_example,
    )

    # log the training script and tags
    try:
        mlflow.log_artifact(__file__)  # with Pathlib can also log a dir
    except NameError:
        pass  # __file__ not defined if run interactively

    mlflow.set_tags(
        {
            "author": "rutuja",
            "project": "iris-classification",
            "algorithm": "decision-tree",
        }
    )
