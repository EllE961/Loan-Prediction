"""Re-run the loan model, compare it against baselines with cross-validation, and write the README figures.

Run from the repo root:  python scripts/make_figures.py   (several minutes, most of it nested CV)
Prints every number quoted in the README and writes docs/assets/*.png.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
GRID_PARAMS = {"n_estimators": [50, 100], "max_depth": [5, 10, None], "min_samples_split": [2, 5, 10]}
PCT = ([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote docs/assets/{name}")


def prepare():
    """The preprocessing from loan_prediction.py, applied to data/train.csv."""
    df = pd.read_csv(ROOT / "data" / "train.csv")
    for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
        df[col] = LabelEncoder().fit_transform(df[col].fillna("Unknown"))
    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History", "ApplicantIncome", "CoapplicantIncome"]:
        df[col] = SimpleImputer(strategy="median").fit_transform(df[[col]]).ravel()
    df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_to_Loan"] = df["Total_Income"] / (df["LoanAmount"] + 1)
    X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
    return X, df["Loan_Status"]


def rate_bars(ax, labels, rates, sizes, color):
    bars = ax.bar(labels, rates, color=color, width=0.62)
    for b, r, n in zip(bars, rates, sizes):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.0%}", ha="center", va="bottom", color=INK2, fontsize=8.5)
        ax.text(b.get_x() + b.get_width() / 2, 0.025, f"n={n}", ha="center", va="bottom", color=BG, fontsize=7.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(*PCT)


def eda_figure(raw):
    y = (raw["Loan_Status"] == "Y").astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [0.8, 1.05, 1.45]})

    ax = axes[0]
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    bars = ax.bar(["rejected", "approved"], counts, color=[INK3, EMBER], width=0.6)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}  ({v / len(y):.1%})", ha="center", va="bottom",
                color=INK2, fontsize=8.5)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_ylabel("applications")
    ax.set_title(f"Outcome ({len(y)} applications)")

    ax = axes[1]
    ch = raw["Credit_History"].map({1.0: "meets\nguidelines", 0.0: "does not\nmeet"}).fillna("missing")
    order = ["meets\nguidelines", "does not\nmeet", "missing"]
    g = y.groupby(ch).agg(["mean", "size"]).loc[order]
    rate_bars(ax, order, g["mean"].values, g["size"].values, [EMBER, TEMPER, INK3])
    ax.set_title("Approval rate by credit history")

    ax = axes[2]
    total = raw["ApplicantIncome"] + raw["CoapplicantIncome"]
    q = pd.qcut(total, 5)
    g = y.groupby(q, observed=True).agg(["mean", "size"])
    labels = [f"{iv.left / 1000:.1f}k to\n{iv.right / 1000:.1f}k" for iv in g.index]
    labels[0] = f"under\n{g.index[0].right / 1000:.1f}k"
    labels[-1] = f"over\n{g.index[-1].left / 1000:.1f}k"
    rate_bars(ax, labels, g["mean"].values, g["size"].values, PURPLE)
    ax.set_xlabel("household income quintile (applicant + co-applicant)")
    ax.set_title("Approval rate by income")
    fig.tight_layout(w_pad=2.5)
    save(fig, "eda.png")


def model_figure(names, scores, highlight):
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    means = np.array([s.mean() for s in scores])
    stds = np.array([s.std() for s in scores])
    y = np.arange(len(names))[::-1]
    colors = [EMBER if n == highlight else (INK3 if i < 2 else TEMPER) for i, n in enumerate(names)]
    ax.barh(y, means, xerr=stds, color=colors, height=0.6, error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 3})
    for yi, m, s in zip(y, means, stds):
        ax.text(m + s + 0.012, yi, f"{m:.3f} ± {s:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy (mean ± std over 50 folds)")
    ax.set_title("Cross-validated accuracy on 614 applications")
    fig.text(0.012, -0.015, "Repeated stratified 10-fold CV (5 repeats). The tuned forest is nested: the grid search "
             "runs inside each training fold.", color=INK3, fontsize=8.5, ha="left", va="top")
    save(fig, "models.png")


def confusion_importance_figure(cm, imp_names, imp_mean, imp_std):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    ax.imshow(cm, cmap=HEAT, vmin=0, vmax=cm.max())
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n{v / total:.1%}", ha="center", va="center", fontsize=11,
                    color=BG if v > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["rejected", "approved"])
    ax.set_yticks([0, 1], ["rejected", "approved"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Tuned forest, validation split (123)")

    ax = axes[1]
    order = np.argsort(imp_mean)[-7:]
    y = np.arange(len(order))
    ax.barh(y, imp_mean[order], xerr=imp_std[order], height=0.6,
            color=[EMBER if imp_names[k] == "Credit_History" else TEMPER for k in order],
            error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 3})
    for yi, k in zip(y, order):
        ax.text(max(imp_mean[k], 0) + imp_std[k] + 0.004, yi, f"{imp_mean[k]:+.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [imp_names[k] for k in order])
    ax.set_xlim(min(0, (imp_mean[order] - imp_std[order]).min() - 0.005), imp_mean.max() + imp_std.max() + 0.04)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy drop when the feature is shuffled (top 7 of 13 features)")
    ax.set_title("Permutation importance, validation split")
    fig.tight_layout(w_pad=3)
    save(fig, "confusion_importance.png")


def main():
    raw = pd.read_csv(ROOT / "data" / "train.csv")
    X, y = prepare()
    print(f"applications: {len(y)}, approved: {y.sum()} ({y.mean():.3f})")
    ch = raw["Credit_History"]
    for label, mask in [("meets", ch == 1), ("does not meet", ch == 0), ("missing", ch.isna())]:
        print(f"credit history {label:14s} n={mask.sum():3d} approval {y[mask].mean():.3f}")

    # the script's own split and grid search
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    search = GridSearchCV(RandomForestClassifier(random_state=42), GRID_PARAMS, cv=5, scoring="accuracy", n_jobs=4)
    search.fit(X_train, y_train)
    pred = search.best_estimator_.predict(X_val)
    cm = confusion_matrix(y_val, pred)
    print(f"best params {search.best_params_}, inner CV {search.best_score_:.4f}")
    print(f"validation accuracy {accuracy_score(y_val, pred):.4f} ({(pred == y_val).sum()}/{len(y_val)}), "
          f"confusion {cm.tolist()}")
    rule = (X_val["Credit_History"] == 1).astype(int)
    print(f"credit-history rule on the same validation split: {accuracy_score(y_val, rule):.4f}, "
          f"agrees with forest on {(rule.values == pred).mean():.3f}")

    pi = permutation_importance(search.best_estimator_, X_val, y_val, n_repeats=30, random_state=0, n_jobs=4)
    for k in np.argsort(-pi.importances_mean)[:5]:
        print(f"permutation importance {X.columns[k]:18s} {pi.importances_mean[k]:+.4f} +/- {pi.importances_std[k]:.4f}")

    credit_only = ColumnTransformer([("credit", "passthrough", ["Credit_History"])])
    tuned = GridSearchCV(RandomForestClassifier(random_state=42), GRID_PARAMS, cv=5, scoring="accuracy", n_jobs=4)
    models = {
        "Majority class (approve all)": DummyClassifier(strategy="most_frequent"),
        "Credit history only (stump)": make_pipeline(credit_only, DecisionTreeClassifier(max_depth=1)),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Gradient boosting": GradientBoostingClassifier(random_state=0),
        "Random forest (defaults)": RandomForestClassifier(random_state=42),
        "Random forest, tuned (this repo)": tuned,
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    scores = []
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=1 if m is tuned else 4)
        scores.append(s)
        print(f"{name:34s} {s.mean():.4f} +/- {s.std():.4f}")

    eda_figure(raw)
    model_figure(list(models), scores, highlight="Random forest, tuned (this repo)")
    confusion_importance_figure(cm, list(X.columns), pi.importances_mean, pi.importances_std)


if __name__ == "__main__":
    main()
