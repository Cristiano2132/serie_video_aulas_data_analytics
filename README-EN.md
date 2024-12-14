
# Video Lesson Plan: Data Science and Analytics

## Introductory Module: Fundamentals of Statistics

### 1. Descriptive Statistics

**Objective:** Introduce measures of central tendency, dispersion, and charts.  
**Business Problem:** Summarize a sales dataset to better understand customer characteristics.  
**Background:**  
Descriptive statistics are essential for understanding the basic characteristics of a dataset before moving on to complex analyses. This step is crucial in any data analysis pipeline. Measures such as mean, median, variance, and standard deviation help to summarize and visualize the data in a comprehensible manner.  
**Data:** Iris Dataset from Scikit-Learn.  
**Exercise:** Calculate mean, median, variance, standard deviation, and create a histogram.

---

### 2. Relationship Between Variables

**Objective:** Explore relationships between qualitative and quantitative variables.  
**Business Problem:** Identify relationships between customer characteristics and their purchasing decisions.  
**Background:**  
The relationship between variables helps to identify important associations in data. The correlation coefficient measures the strength and direction of a linear relationship between numerical variables, while the chi-square test evaluates associations between categories.  
**Exercise:**

- Use `scipy.stats` to calculate the Pearson correlation coefficient.

```python
from scipy.stats import pearsonr

# Example data
x = [10, 20, 30, 40]
y = [15, 25, 35, 45]

# Correlation coefficient
corr, _ = pearsonr(x, y)
print(f"Correlation coefficient: {corr:.2f}")
```

**Data Sources:** Kaggle, UCI Machine Learning Repository

### 3. Probability Distributions

**Objective:** Present discrete and continuous distributions.  
**Business Problem:** Simulate daily sales of a store using binomial and normal distributions.  
**Background:**  
Understanding distributions is essential for modeling uncertainties in data. The normal distribution is widely used due to its application in natural phenomena, while the binomial distribution is useful for modeling counts of successes in trials.  
**Exercise:**

- Use numpy to simulate data from a normal distribution.

```python
import numpy as np

# Generate normally distributed data
data = np.random.normal(loc=50, scale=10, size=1000)
```

**Data Sources:** Data simulation with NumPy

### 4. Hypothesis Testing

**Objective:** Introduce statistical tests for means, variances, and associations.  
**Business Problems:**

1. Verify if the average sales of a store differ from an expected value.
2. Compare the average sales of two stores to assess performance differences.  
   **Background:**  
   Hypothesis tests help to validate assumptions about data. The Z-test is used for large samples, while the t-test is appropriate for smaller samples. The chi-square test evaluates associations in frequency tables.  
   **Common Tests and Examples:**

3. Z-test for means (one sample): Verify if the average sales equal 500 units.

```python
from statsmodels.stats.weightstats import ztest

# Fictional data
sales = [510, 490, 505, 515, 480]

# Z-test
z_stat, p_val = ztest(sales, value=500)
print(f"Z-statistic: {z_stat:.2f}, p-value: {p_val:.4f}")
```

2. t-test for means (one sample): Compare the average sales with 500 units.

```python
from scipy.stats import ttest_1samp

# t-test
t_stat, p_val = ttest_1samp(sales, 500)
print(f"t-statistic: {t_stat:.2f}, p-value: {p_val:.4f}")
```

3. t-test for two independent samples: Compare average sales between two stores.

```python
from scipy.stats import ttest_ind

# Fictional data
store1 = [510, 520, 530, 500]
store2 = [480, 490, 495, 500]

# t-test
t_stat, p_val = ttest_ind(store1, store2)
print(f"t-statistic: {t_stat:.2f}, p-value: {p_val:.4f}")
```

4. F-test for variances: Compare sales variances between two stores.

```python
from scipy.stats import f_oneway

# F-test
f_stat, p_val = f_oneway(store1, store2)
print(f"F-statistic: {f_stat:.2f}, p-value: {p_val:.4f}")
```

5. Chi-square test for frequency tables: Check the association between categories.

```python
from scipy.stats import chi2_contingency

# Contingency table
table = [[10, 20], [30, 40]]

# Chi-square test
chi2, p_val, dof, expected = chi2_contingency(table)
print(f"Chi-square: {chi2:.2f}, p-value: {p_val:.4f}")
```

6. Confidence Interval: Construct a confidence interval for the mean.

```python
import scipy.stats as stats

mean = np.mean(sales)
sem = stats.sem(sales)
ci = stats.t.interval(alpha=0.95, df=len(sales)-1, loc=mean, scale=sem)
print(f"Confidence Interval: {ci}")
```

**Data Sources:** Fictional data or store sales datasets (Kaggle, UCI Machine Learning Repository)

## Module: Supervised and Unsupervised Machine Learning Models
```

Se precisar de mais alguma coisa, estou à disposição! 😊