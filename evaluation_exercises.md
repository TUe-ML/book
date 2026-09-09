(eval_exercises)=
# Exercises


## Evaluating a Regression Model from an Actual-vs-Predicted Plot
Consider a machine learning model that predicts the prices of houses based on various features
such as size, number of bedrooms, and location. The graph below shows the actual prices of houses
($y$-axis) versus the predicted prices by the model ($x$-axis) for a test dataset of 20 houses.

```{figure} images/regression/exercises_actual_vs_predicted.png
---
width: 420px
name: fig-regression-actual-vs-predicted
align: center
---
Actual vs. predicted house prices for 20 test houses.
```

Which of the following metrics can be directly derived from the given graph to evaluate the
model's performance?

Group of answer choices
- Mean Absolute Error (MAE)
- Precision
- Accuracy
- R-squared (R²)

````{dropdown} Solution
**Mean Absolute Error (MAE) — yes.** For every point in the plot we can read off both the actual
value $y_i$ and the predicted value $\hat y_i$ (its vertical distance from the diagonal $y=\hat y$
line is exactly the absolute error $\lvert y_i-\hat y_i\rvert$). Averaging these 20 absolute
errors gives the MAE directly from the plotted data.

**R-squared (R²) — yes.** R² is computed from the same actual/predicted pairs (it compares the
residual sum of squares around the diagonal to the total sum of squares of the actual values
around their mean), so it is equally derivable from this data. Visually, the more tightly the
points cluster around the diagonal $y=\hat y$ line, the higher the R².

**Precision — no.** Precision is defined for classification (the fraction of predicted-positive
instances that are actually positive). House price prediction here is a regression task with
continuous predictions, not class labels, so precision is not defined for this output without
first imposing an arbitrary classification threshold, which the graph does not provide.

**Accuracy — no.** For the same reason, accuracy (fraction of exactly correct predictions) is a
classification metric. Since the predictions are continuous prices, expecting them to exactly
match the actual price is not a meaningful notion of "correctness" here, so accuracy cannot be
derived from this graph either.
````

## ROC Curves
The graph below shows the receiver operating characteristic (ROC) curve for a logistic regression model that classifies whether an email is spam or not spam.

```{figure} images/evaluation/exercises_roc_curve_spam.png
---
width: 480px
name: fig-eval-roc-spam
align: center
---
ROC curve of a logistic regression model for spam classification.
```

Which of the following statements about the model's performance is correct, based on the given ROC curve?

- The model has perfect accuracy.
- The model has poor performance and should not be used.
- The model has a high false positive rate and a low true positive rate.
- The model is overfitting the training data.
- The model has a high true positive rate and a low false positive rate.

````{dropdown} Solution
**The model has a high true positive rate and a low false positive rate.**

The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) as the classification threshold is varied. A curve that lies close to the top-left corner, as in the plot above, indicates that the model can achieve a high TPR while keeping the FPR low, which is characteristic of a well-performing classifier. The curve also lies far above the diagonal "Reference" line, which represents random guessing (TPR = FPR).

The other options are incorrect:
- *Perfect accuracy* would require the curve to pass exactly through the top-left corner (TPR = 1, FPR = 0). Also, accuracy is not directly readable from an ROC curve, since it depends on the class balance and on the classification threshold that is chosen.
- *Poor performance* is inconsistent with a curve lying well above the diagonal. A classifier with performance close to random would tend to lie close to the diagonal instead.
- *High FPR and low TPR* describes the opposite of what the curve shows.
- *Overfitting* cannot be diagnosed from a single ROC curve computed on one dataset. This would require comparing the model's performance on training data versus test data.
````


## Cross-Validation
You are training a machine learning model on a dataset with 1000 samples. You decide to use 5-fold cross-validation to assess the model's performance. How many subsets is the dataset divided into during the cross-validation process?

- 10
- 20
- 5

````{dropdown} Solution
**5**

In $k$-fold cross-validation, the dataset is split into $k$ subsets of about equal size, called folds. For $k=5$, the 1000 samples are split into **5** folds of 200 samples each.

The model is then trained and tested 5 times. Each time, 4 folds (800 samples) are used for training and the remaining fold (200 samples) is used for validation. This way, every sample is used exactly once for validation over the 5 runs.
````


## Confusion Matrices
You have trained a binary classification model and compare the predicted classes $\hat{y}_j$ with the true labels $y_j$ on the test data:

| $y_j$ | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
|---|---|---|---|---|---|---|---|---|
| $\hat{y}_j$ | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 |

What does the confusion matrix look like for this result? That is, compute $TP$, $FN$, $FP$, and $TN$, where the class with label 1 is the positive class and the class with label 0 is the negative class.

````{dropdown} Solution
Going through the eight test points one by one and comparing the true label $y_j$ with the predicted label $\hat{y}_j$:

| $j$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $y_j$ | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| $\hat{y}_j$ | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 |
| outcome | TP | FP | TN | FN | TP | TN | TP | FP |

Counting up the outcomes gives:
$$TP = 3, \qquad FN = 1, \qquad FP = 2, \qquad TN = 2.$$

So the confusion matrix looks like this:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | $TP=3$ | $FN=1$ |
| **Actual Negative** | $FP=2$ | $TN=2$ |

As a check, the four counts add up to $3+1+2+2=8$, the total number of test points.
````


## Precision-Recall Tradeoff
A logistic regression model is trained to classify sentiment as positive or negative. The model's performance can be evaluated using precision and recall. Which of the following statements about the precision-recall tradeoff is correct?

- A higher threshold for classification will result in higher precision but lower recall.
- Increasing the threshold for classification will always increase both precision and recall.
- Precision and recall can never be optimized simultaneously; improving one will always harm the other.
- Precision and recall are independent of each other; changing one will not affect the other.

````{dropdown} Solution
**A higher threshold for classification will result in higher precision but lower recall.**

A logistic regression model outputs a predicted probability that a sample belongs to the positive class. It classifies a sample as positive only if this probability is above a chosen threshold. Raising the threshold makes the model more careful about predicting the positive class:
- Only samples where the model is very confident get predicted as positive. So the fraction of predicted positives that are actually positive tends to go up. This means **precision increases**.
- At the same time, some true positives that the model was less confident about now fall below the threshold and get missed. So fewer of the actual positives are found. This means **recall decreases**.

The other statements are false:
- Increasing the threshold does **not** increase both metrics at once. This tradeoff is exactly the point: one usually improves while the other gets worse.
- The tradeoff is a general tendency, not a strict rule. Saying that improving one metric will *always* harm the other is too strong a claim.
- Precision and recall are both computed from the same confusion matrix, and both change as the threshold changes. So they are not independent of each other.
````


## Precision and Recall from a Confusion Matrix
A machine learning classifier was developed on a dataset with 1000 samples. The confusion matrix for the classifier's predictions is as follows:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | 180 | 20 |
| **Actual Negative** | 30 | 770 |

Compute the precision and recall of this model (rounded to three decimals).

````{dropdown} Solution
From the confusion matrix we read off $TP = 180$, $FN = 20$, $FP = 30$, and $TN = 770$.

**Precision** is the fraction of predicted positives that are actually positive:
$$\text{Precision} = \frac{TP}{TP+FP} = \frac{180}{180+30} = \frac{180}{210} \approx 0.857.$$

**Recall** is the fraction of actual positives that were correctly found:
$$\text{Recall} = \frac{TP}{TP+FN} = \frac{180}{180+20} = \frac{180}{200} = 0.900.$$

As a check, the four numbers in the confusion matrix add up to $180+20+30+770=1000$, which matches the total number of samples.
````


## Practical Exercise: Verifying SecureBank's Fraud Detection Model
You are starting a new project at work: as the ML Engineer on the team, your job is to verify SecureBank's existing AI model for fraud detection before it is trusted for wider use. The team that built the model reports: *"It's 98% accurate on our test set, so it's ready to go!"* (It's actually a `DummyClassifier`.)

**Question 1.** Before looking at the data, does 98% accuracy alone convince you that this model is ready to deploy? What is it about a fraud-detection dataset that could make accuracy a misleading metric to verify a model with?

To carry out your verification, the cell below generates SecureBank's (synthetic) transaction history: 5000 transactions, about 2% of which are fraud. You hold out part of this data as a **validation set** that you will use throughout this exercise to inspect the model's behavior and choose a classification threshold. This is deliberately not called a test set: in practice, a test set should stay untouched until the very end, so that the number you finally report is not biased by having been used to make decisions about the model.



```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=5,
    weights=[0.98, 0.02], flip_y=0.01, random_state=0,
)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)
print(f"Fraud rate in training data: {y_train.mean():.3%}")
print(f"Fraud rate in validation data: {y_val.mean():.3%}")

```

**Task 1.** Implement `confusion_counts(y_true, y_pred)` below, returning `(TP, FP, FN, TN)`, where fraud (label `1`) is the positive class. You will reuse this function throughout the exercise, so implement the counting yourself rather than calling `sklearn.metrics.confusion_matrix`.



```python
def confusion_counts(y_true, y_pred):
    '''Return (TP, FP, FN, TN), treating class 1 (fraud) as positive.'''
    # Your implementation here
    TP = FP = FN = TN = 0

    return TP, FP, FN, TN

# Sanity check with the tiny example from the "Confusion Matrices" exercise above
y_true_demo = np.array([1, 0, 0, 1, 1, 0, 1, 0])
y_pred_demo = np.array([1, 1, 0, 0, 1, 0, 1, 1])
print(confusion_counts(y_true_demo, y_pred_demo))  # should print (3, 2, 1, 2)

```

**Task 2.** Train a `LogisticRegression` on `(X_train, y_train)`. Using a classification threshold of 0.5, predict on the validation set and use your `confusion_counts` function to compute the accuracy, precision, and recall. One of the most basic model-verification steps is to compare a model against a trivial baseline: fit a `DummyClassifier(strategy="most_frequent")` from `sklearn.dummy` on `(X_train, y_train)` (it simply always predicts the majority class) and compute the same three metrics for it. Compare the two. (When a classifier never predicts positive, its precision has a $0/0$ denominator; treat it as $0.0$ in that case.)



```python
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
val_scores = model.predict_proba(X_val)[:, 1]  # predicted fraud probability

def accuracy_precision_recall(y_true, y_pred):
    TP, FP, FN, TN = confusion_counts(y_true, y_pred)
    # Your implementation here
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    return accuracy, precision, recall

# Your implementation here: threshold `val_scores` at 0.5 to get predictions
y_pred_default = np.zeros_like(y_val)
model_accuracy, model_precision, model_recall = accuracy_precision_recall(y_val, y_pred_default)

# Your implementation here: fit a DummyClassifier(strategy="most_frequent") on
# (X_train, y_train), then predict on X_val
baseline_pred = np.zeros_like(y_val)
baseline_accuracy, baseline_precision, baseline_recall = accuracy_precision_recall(y_val, baseline_pred)

print(f"Logistic regression:      accuracy={model_accuracy:.3f}, precision={model_precision:.3f}, recall={model_recall:.3f}")
print(f"Majority-class baseline:  accuracy={baseline_accuracy:.3f}, precision={baseline_precision:.3f}, recall={baseline_recall:.3f}")

```

**Question 2.** Explain, in terms of the confusion matrix, why the baseline achieves high accuracy despite never catching a single fraudulent transaction. Which metric, precision or recall, exposes this failure most clearly, and why?

**Task 3.** A single threshold of 0.5 hides how the logistic regression model behaves as you make it more or less conservative about flagging fraud. Implement `sweep_thresholds(y_true, y_scores, thresholds)` below, which should return the precision, recall, and false positive rate at each threshold in `thresholds`. Each point of the resulting curves belongs to one specific threshold: sweeping over the whole range of `thresholds` traces out the entire curve, this is not about picking a single "best" threshold yet. Use your function to plot (a) precision and recall as direct functions of the threshold, so you can read off which threshold gives which trade-off, together with (b) the precision-recall curve and (c) the ROC curve of the model on the validation set. Report the AUC as well (you may use `sklearn.metrics.roc_auc_score` directly for this last part).



```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

thresholds = np.linspace(0.01, 0.99, 50)

def sweep_thresholds(y_true, y_scores, thresholds):
    '''For each threshold, return (precisions, recalls, fprs) as arrays of the
    same length as `thresholds`, obtained by predicting positive whenever
    y_scores >= threshold.
    '''
    precisions, recalls, fprs = [], [], []
    for t in thresholds:
        # Your implementation here: threshold y_scores at t, then use
        # confusion_counts to get precision, recall, and the false positive rate
        precisions.append(0.0)
        recalls.append(0.0)
        fprs.append(0.0)
    return np.array(precisions), np.array(recalls), np.array(fprs)

precisions, recalls, fprs = sweep_thresholds(y_val, val_scores, thresholds)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(thresholds, precisions, marker=".", label="precision")
axes[0].plot(thresholds, recalls, marker=".", label="recall")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Score")
axes[0].set_title("Precision & recall vs. threshold")
axes[0].legend()

axes[1].plot(recalls, precisions, marker=".")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall curve")

axes[2].plot(fprs, recalls, marker=".")
axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC curve")
plt.tight_layout()
plt.show()

auc_score = roc_auc_score(y_val, val_scores)
print(f"AUC = {auc_score:.3f}")

```

**Question 3.** SecureBank estimates that manually reviewing a flagged transaction costs `$5`, while letting a fraudulent transaction slip through costs `$200` on average. This defines an expected cost for operating at a given threshold $\tau$:
$$\text{Cost}(\tau) = 5\cdot FP(\tau) + 200\cdot FN(\tau).$$
Using `confusion_counts` again, compute this cost at each threshold in `thresholds`, plot the cost against the threshold, and identify the threshold that minimizes the expected cost on the validation set. Would you recommend a low or a high classification threshold at SecureBank? Beyond the dollar figures in the formula above, what else might be worth raising with your team before recommending this threshold?



```python
def expected_cost(y_true, y_scores, thresholds, cost_fp=5, cost_fn=200):
    '''Return an array of expected costs, one per threshold in `thresholds`.'''
    costs = []
    for t in thresholds:
        # Your implementation here: threshold y_scores at t, get FP and FN via
        # confusion_counts, and compute cost_fp * FP + cost_fn * FN
        costs.append(0.0)
    return np.array(costs)

costs = expected_cost(y_val, val_scores, thresholds)

plt.plot(thresholds, costs, marker=".")
plt.xlabel("Threshold")
plt.ylabel("Expected cost ($)")
plt.show()

best_threshold = thresholds[np.argmin(costs)]
print(f"Threshold minimizing expected cost: {best_threshold:.2f} (cost = {costs.min():.2f})")

```

**Task 4.** A single train/validation split gives you only one AUC estimate, which could get lucky or unlucky depending on which transactions ended up in the validation set. Use 5-fold cross-validation (`StratifiedKFold` together with `cross_val_score`, scoring `"roc_auc"`) on the full dataset `(X, y)` to obtain 5 AUC estimates for a fresh logistic regression model. Report their mean and standard deviation.

**Question 4.** How does the spread of the 5 cross-validated AUC scores compare to the single validation-set AUC you computed in Task 3? What does this tell your team about how much to trust a performance number computed from a single train/validation split?



```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Your implementation here: use cross_val_score with the `cv` object above and
# scoring="roc_auc" to get one AUC score per fold for a fresh LogisticRegression model
cv_auc_scores = np.zeros(5)

print(f"Per-fold AUC: {np.round(cv_auc_scores, 3)}")
print(f"Mean AUC = {cv_auc_scores.mean():.3f}, Std = {cv_auc_scores.std():.3f}")

```

````{dropdown} Solution
This exercise will be solved together with the TA during the exercise session.
````

