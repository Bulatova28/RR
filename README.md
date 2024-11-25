# Special programming languages

Topic: "Visualization of statistical tests"
Bulatova Viktoriia, Popp Sofia, SA-32

## RR_PROJECT_BULATOVA_POPP

A Python statistical test visualization project designed to help analyze and visualize the results of statistical methods such as t-tests, ANOVA, and others.

## Description

RR_PROJECT_BULATOVA_POPP is a comprehensive Python library designed for automated statistical hypothesis testing and data visualization. The package simplifies the decision-making process by automatically selecting the appropriate parametric or non-parametric test based on the data characteristics, including normality and sample size. It also provides robust visualizations to ensure clear and insightful presentation of results.

Main features:
1. Automatic Test Selection
- Automatically chooses between parametric and non-parametric tests based on:
    - Normality of data, determined using Shapiro-Wilk or D’Agostino-Pearson tests depending on the sample size.
    - Homogeneity of variances for multi-group tests.
    - Supports a wide range of statistical tests:
    - Paired samples: Paired t-test, Wilcoxon signed-rank test.
    - Independent samples: Independent t-test, Mann-Whitney U test.
    - Multi-group comparisons: One-way ANOVA, Alexander-Govern test, Kruskal-Wallis test.
2. Assumption Checks
- Normality Testing:
    - Shapiro-Wilk test for small samples.
    - D’Agostino-Pearson test for larger datasets.
    - Homoscedasticity (Equal Variances):
    - Bartlett’s test for verifying variance equality across groups.
3. Descriptive Statistics
- Provides detailed summaries for each sample:
    - Mean, median, mode, variance, standard deviation.
    - Skewness, kurtosis, range, minimum, and maximum.
4. Advanced Visualizations
- Visualizations are integrated into the analysis to ensure intuitive understanding:
    - Histograms with KDE: Displays data distributions and overlays the ideal normal curve.
    - Q-Q Plots: For assessing normality.
    - Boxplots: Highlights variability and spread across groups.
    - Joyplots: Overlays group distributions, emphasizing mean or median values for quick interpretation.
    - Custom Visualization for Differences: Specialized plots for paired tests, showing distributions and differences between samples.

## Installation

To install this package, use pip