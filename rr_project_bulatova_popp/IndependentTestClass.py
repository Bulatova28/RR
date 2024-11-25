import scipy.stats as st
from scipy.stats import shapiro, normaltest, ttest_rel, wilcoxon, ttest_ind, mannwhitneyu, f_oneway, kruskal, bartlett, alexandergovern
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joypy import joyplot
from scipy.stats import norm
from typing import Callable, Optional
from abc import ABC, abstractmethod
from .ABCTest import StatisticalTest

class IndependentTtest(StatisticalTest):
    """
    A class to perform an independent t-test or Mann-Whitney U test on two samples 
    and provide related descriptive statistics, normality checks, and visualizations.

    Attributes:
        data (pd.DataFrame): The dataset containing the two samples.
        columns (list[str]): The names of the two columns to compare.
        alpha (float): The significance level for hypothesis testing (default: 0.05).
    """

    def __init__(self, data: pd.DataFrame, columns: list[str], alpha: float = 0.05) -> None:
        """
        Initializes the IndependentTtest class.

        Args:
            data (pd.DataFrame): The dataset containing the samples.
            columns (list[str]): The names of the two columns to compare.
            alpha (float): The significance level for hypothesis testing (default: 0.05).

        Raises:
            ValueError: If the number of columns is not exactly 2 or if specified columns are missing.
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input 'data' must be a pandas DataFrame.")
            
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise ValueError("The 'columns' must be a list of strings.")
            
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f'The following columns are missing in the dataset: {missing_columns}')
            
        for col in columns:
            col_data = data[col].dropna()

            if not np.issubdtype(col_data.dtype, np.number):
                raise ValueError(f"Column '{col}' must contain numeric values.")
            
            if not np.isfinite(col_data).all():
                raise ValueError(f"Column '{col}' contains non-finite values (inf or NaN).")
            
            if len(col_data) < 2:
                raise ValueError(f"Column '{col}' must have at least two non-NaN values.")
            
        if len(columns) != 2:
            raise ValueError(f'The number of samples must be 2.')
            
        if not isinstance(alpha, (float, int)) or not (0 < alpha < 1):
            raise ValueError("The 'alpha' must be a float between 0 and 1.")

        super().__init__(data, columns, alpha)

    
    def check_normality(self) -> bool:
        """
        Performs a normality test on the specified column.

        Args:
            column (str): The column name for which the normality test will be performed.

        Returns:
            bool: True if the data is normally distributed, False otherwise.

        Raises:
            ValueError: If the column name is invalid or not provided.
        """
        
        not_lurge_length = all(len(self.data[column]) < 5000 for column in self.columns)
        if not_lurge_length:
            print('---The Shapiro-Wilk normality test has been performed---\n')
        else:
            print('---The D’Agostino and Pearson’s normality test has been performed---\n')  
        results = {}
    
        for column in self.columns:
            if not_lurge_length:
                stat, p_shapiro = st.shapiro(self.data[column])
                is_normal = p_shapiro > self.alpha
                results[column] = {
                    'stat': stat,
                    'p_value': p_shapiro,
                    'is_normal': is_normal 
                }
            else: 
                stat, p_normal = st.normaltest(self.data[column])
                is_normal = p_normal > self.alpha
                results[column] = {
                    'stat': stat,
                    'p_value': p_normal,
                    'is_normal':  is_normal
                }

        norm_df = pd.DataFrame.from_dict(results)
        
        print(norm_df)
        
        normality = norm_df.loc['is_normal'].all()        
        
        if  normality:
            print(f'All THE SAMPLES ARE NORMALLY DISTRIBUTED\n')
        else:
            print(f'Any of the samples is not normally distributed\n')
        return normality
        
    def normality_visualization(self, vis_func='histogram') -> None:
        """
        Visualizes the distribution of the two samples using histograms or Q-Q plots.

        Args:
            vis_func (str): Type of visualization to create ('histogram' or 'probplot').

        Raises:
            ValueError: If the visualization type is invalid.
        """
   
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.flatten()
    
        for i, column in enumerate(self.columns):
            original_data = self.data[column].dropna() 
            mean = original_data.mean()
            std = original_data.std()
            x = np.linspace(original_data.min(), original_data.max(), 1000)
            ideal_pdf = norm.pdf(x, loc=mean, scale=std)
    
            if vis_func == 'histogram':
                sns.histplot(original_data, kde=True, color='green', bins=30, label=f'{column} Original', stat='density', alpha=0.6, ax=axes[i])
                sns.lineplot(x=x, y=ideal_pdf, color='red', lw=2, label="Ideal Normal (PDF)", ax=axes[i])
                axes[i].set_title(f"{column} Distribution", fontsize=12)
                axes[i].set_xlabel("Value", fontsize=10)
                axes[i].set_ylabel("Density", fontsize=10)
                axes[i].legend()
                
            elif vis_func == 'probplot':
                st.probplot(original_data, dist="norm", plot=axes[i])
                axes[i].set_title(f"{column} Q-Q Plot", fontsize=12)
                axes[i].set_xlabel("Theoretical Quantiles", fontsize=10)
                axes[i].set_ylabel("Ordered Values", fontsize=10)
            else:
                raise ValueError("vis_func must be 'histogram' or 'probplot'")
            
        
    def visualize_joyplot(self, test_name="t-test") -> None:
        """
        Creates a joyplot for the two samples with mean or median lines.

        Args:
            test_name (str): Type of test ('t-test' or 'mannwhitneyu').

        Raises:
            ValueError: If the test name is invalid.
        """

        if test_name not in ["t-test", "mannwhitneyu"]:
            raise ValueError(f"Invalid test_name: {test_name}. Choose 't-test' or 'mannwhitneyu'.")

        sample_1 = self.data[self.columns[0]]
        sample_2 = self.data[self.columns[1]]

        samples = {"Sample 1": sample_1, "Sample 2": sample_2}
        fig, axes = joyplot(
            samples,
            figsize=(8, 6),
            colormap=plt.cm.plasma if test_name == "t-test" else plt.cm.viridis,
            overlap=0.3,
            fade=True
        )

        for ax, (label, sample) in zip(axes, samples.items()):
            if test_name == "t-test":
                stat_value = np.mean(sample)
                color = 'red'
                label_text = f"Mean: {stat_value:.2f}"
            else:  
                stat_value = np.median(sample)
                color = 'blue'
                label_text = f"Median: {stat_value:.2f}"

            ax.axvline(stat_value, color=color, linestyle="--", label=label_text)
            ax.legend()

        plt.title(f"Joyplot with {'Mean' if test_name == 't-test' else 'Median'} Values")
        plt.show()

    def run_test(self, alternative = "two-sided", equal_var = False, nan_policy="omit") -> dict[str, float]:
        """
        Performs an independent t-test or Mann-Whitney U test depending on normality.

        Args:
            alternative (str): Type of alternative hypothesis ('two-sided', 'less', or 'greater').
            equal_var (bool): Assumes equal variance if True (default: False).
            nan_policy (str): Defines how to handle NaNs ('omit', 'propagate', or 'raise').

        Returns:
            dict: A dictionary with the test statistic and p-value.

        Raises:
            ValueError: If arguments are invalid.
        """

        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError(f'Invalid alternative hypothesis: {alternative}. Choose "two-sided", "less", or "greater".')
        
        if not isinstance(equal_var, bool):
            raise ValueError(f'Invalid value for equal_var: {equal_var}. It must be a boolean (True or False).')
    
        if nan_policy not in ["omit", "propagate", "raise"]:
            raise ValueError(f'Invalid nan_policy: {nan_policy}. Choose "omit", "propagate", or "raise".')

        
        sample_1 = self.data[self.columns[0]]
        sample_2 = self.data[self.columns[1]]

        is_all_normal = self.check_normality()
        
        samples = [self.data[column] for column in self.columns]

        if is_all_normal:
            ttest_ind = st.ttest_ind(
                    *samples,
                    equal_var=equal_var,
                    nan_policy= nan_policy,
                    alternative= alternative)
            print(f'Independent t-test:')
            p_value = ttest_ind.pvalue
            if p_value > self.alpha:
                print(f'pvalue {p_value} > alpha: failed to reject H0 (mean values can be equal)')
            else:
                print(f'pvalue {p_value} =< alpha: reject H0 (mean values are not equal)')

            self.visualize_joyplot(test_name="t-test")
            return {'pvalue': ttest_ind.pvalue, 'statistic': ttest_ind.statistic}
        else:
            mannwhitneyu = st.mannwhitneyu(sample_1, 
                                           sample_2,
                                           alternative=alternative)
            print(f'Mannwhitneyu test:')
            p_value=mannwhitneyu.pvalue
            if p_value > self.alpha:
                print(f'pvalue {p_value} > alpha: failed to reject H0 (distributions of samples are not significantly different)')
            else:
                print(f'pvalue {p_value} =< alpha: reject H0 (distributions of samples are significantly different)')

            self.visualize_joyplot(test_name="mannwhitneyu")
            return {'pvalue': mannwhitneyu.pvalue, 'statistic': mannwhitneyu.statistic}