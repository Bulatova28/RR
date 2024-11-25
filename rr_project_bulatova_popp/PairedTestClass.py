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

class PairedTtest(StatisticalTest):
    """
    A class for conducting paired t-tests and non-parametric Wilcoxon tests on two related samples.

    Attributes:
        data (pd.DataFrame): The dataset containing the paired samples.
        columns (list[str]): The names of the two columns representing the paired samples.
        alpha (float): The significance level for hypothesis testing. Default is 0.05.
    """
    
    def __init__(self, data: pd.DataFrame, columns: list[str], alpha: float = 0.05) -> None:
        """
        Initializes the PairedTtest class with the dataset, columns, and significance level.

        Args:
            data (pd.DataFrame): The dataset containing the paired samples.
            columns (list[str]): The names of the two columns representing the paired samples.
            alpha (float, optional): The significance level for hypothesis testing. Default is 0.05.

        Raises:
            ValueError: If the specified columns are missing, if the number of columns is not 2, 
                        or if the lengths of the two columns are not equal.
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input 'data' must be a pandas DataFrame.")     
            
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise ValueError("The 'columns' must be a list of strings.")
            
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f'The following columns are missing in the dataset: {missing_columns}')

        if len(columns) != 2:
            raise ValueError(f'The number of samples must be 2.')

        for col in columns:
            col_data = data[col].dropna()

            if not np.issubdtype(col_data.dtype, np.number):
                raise ValueError(f"Column '{col}' must contain numeric values.")
            
            if not np.isfinite(col_data).all():
                raise ValueError(f"Column '{col}' contains non-finite values (inf or NaN).")
            
            if len(col_data) < 2:
                raise ValueError(f"Column '{col}' must have at least two non-NaN values.")
                
        if len(data[columns[0]]) != len(data[columns[1]]):
            raise ValueError(f'The lenght of samples {columns[0]} and {columns[1]} is different. Must be the same size of samples.')
        

        if not isinstance(alpha, (float, int)) or not (0 < alpha < 1):
            raise ValueError("The 'alpha' must be a float between 0 and 1.") 

        super().__init__(data, columns, alpha)

    @property
    def diff(self) -> pd.Series:
        """
        Calculates the differences between the paired samples.

        Returns:
            pd.Series: The differences between the two samples (column1 - column2).
        """

        return self.data[self.columns[0]] - self.data[self.columns[1]]
    
    def check_normality(self) -> bool:
        """
        Performs a normality test on the differences between paired samples.

        Returns:
            bool: True if the differences follow a normal distribution (parametric test is appropriate),
                  False otherwise (non-parametric test is appropriate).
        """      

        if len(self.diff) < 5000:
            print('---The Shapiro-Wilk normality test has been performed---\n')
            t_stat, p_value = st.shapiro(self.diff)
            is_normal = p_value > self.alpha
            results = {
                    'stat': t_stat,
                    'p_value': p_value,
                    'is_normal': is_normal 
                }
        else:
            print('---The D’Agostino and Pearson’s normality test has been performed---\n')
            t_stat, p_value = st.normaltest(self.diff)
            is_normal = p_value > self.alpha
            results = {
                    'stat': t_stat,
                    'p_value': p_value,
                    'is_normal': is_normal 
                
                }
        norm_df = pd.DataFrame.from_dict(results, orient='index')
        norm_df.rename(columns={0: 'Diff'}, inplace=True)
        print(norm_df)
        if is_normal:
            print(f'pvalue > alpha: failed to reject H0, so it will be a parametric version of T-test')
        else:
            print(f'pvalue =< alpha: reject H0, so it will be a non-parametric version of T-test')

        return is_normal
    
    def visualize_plot(self, test_name="t-test") -> None:
        """
        Visualizes the data distribution and test statistics for the selected test.

        Args:
            test_name (str, optional): The name of the test ("t-test" or "wilcoxon"). Default is "t-test".

        Raises:
            ValueError: If the test name is not valid.
        """
        
        if test_name not in ["t-test", "wilcoxon"]:
            raise ValueError(f"Invalid test_name: {test_name}. Choose 't-test' or 'wilcoxon'.")

        sample_1 = self.data[self.columns[0]]
        sample_2 = self.data[self.columns[1]]

        if test_name == "t-test":
            samples = {
                "Sample 1": sample_1,
                "Sample 2": sample_2
            }

            fig, axes = joyplot(
                samples,
                figsize=(8, 6),
                colormap=plt.cm.plasma,
                overlap=0.3,
                fade=True
            )

            mean_sample_1 = np.mean(sample_1)
            mean_sample_2 = np.mean(sample_2)

            axes[0].axvline(mean_sample_1, color="red", linestyle="--", label=f"Mean: {mean_sample_1:.2f}")
            axes[1].axvline(mean_sample_2, color="green", linestyle="--", label=f"Mean: {mean_sample_2:.2f}")

            axes[0].legend()
            axes[1].legend()

        else:
            fig, axes = plt.subplots(figsize=(8, 6))
            sns.histplot(self.diff, kde=True, color="skyblue", bins=30, ax=axes, stat="density")

            for line in axes.lines:
                if isinstance(line, plt.Line2D):  
                    line.set_color("purple")  

            median_diff = self.diff.median()

            axes.axvline(median_diff, color="blue", linestyle="--", label=f"Median Difference: {median_diff:.2f}")

            axes.set_xlabel(f"Difference ({self.columns[0]} - {self.columns[1]})", fontsize=12)
            axes.set_ylabel("Density", fontsize=12)
            axes.set_title(f"Histogram with KDE of Differences (Median: {np.median(self.diff):.2f})")

        plt.title(f"{'Joyplot' if test_name == 't-test' else 'Histogram with KDE'} of Differences with {'Mean' if test_name == 't-test' else 'Median'} Value")
        plt.tight_layout()
        plt.show()


    def normality_visualization(self, vis_func='histogram') -> None:
        """
        Visualizes the normality of the differences using a histogram or Q-Q plot.

        Args:
            vis_func (str, optional): Visualization method ("histogram" or "probplot"). Default is "histogram".

        Raises:
            ValueError: If an invalid visualization function is provided.
        """
        
        mean_diff = self.diff.mean()
        std_diff = self.diff.std()
        x = np.linspace(self.diff.min(), self.diff.max(), 1000)
        ideal_pdf = norm.pdf(x, loc=mean_diff, scale=std_diff)
        fig, ax = plt.subplots(figsize=(10, 6))
    
        if vis_func == 'histogram':
            sns.histplot(self.diff, kde=True, bins=30, color='blue', stat='density', alpha=0.6, label="Differences")
            sns.lineplot(x=x, y=ideal_pdf, color='red', lw=2, label="Ideal Normal (PDF)")
            plt.title("Distribution of Differences", fontsize=14)
            plt.xlabel("Difference", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend()
        elif vis_func == 'probplot':
            st.probplot(self.diff, dist="norm", plot=plt)
            plt.title("Q-Q Plot of Differences", fontsize=14)
            plt.xlabel("Theoretical Quantiles", fontsize=12)
            plt.ylabel("Ordered Differences", fontsize=12)
        else:
            raise ValueError("vis_func must be 'histogram' or 'probplot'")
    
        plt.tight_layout()
        plt.show()
    

    def run_test(self, alternative = "two-sided", nan_policy="omit", is_normal=None) -> dict[str, float]:
        """
        Runs the paired t-test or Wilcoxon test based on the normality of the differences.

        Args:
            alternative (str, optional): The alternative hypothesis ("two-sided", "less", or "greater"). Default is "two-sided".
            nan_policy (str, optional): Policy for handling NaN values ("omit", "propagate", or "raise"). Default is "omit".
            is_normal (bool, optional): Whether the differences follow a normal distribution. If None, the normality check is run.

        Returns:
            dict[str, float]: A dictionary containing the test statistic and p-value.

        Raises:
            ValueError: If an invalid alternative hypothesis or NaN policy is provided.
        """
        
        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError(f'Invalid alternative hypothesis: {alternative}. Choose "two-sided", "less", or "greater".')
        
        if nan_policy not in ["omit", "propagate", "raise"]:
            raise ValueError(f'Invalid nan_policy: {nan_policy}. Choose "omit", "propagate", or "raise".')
        
        if is_normal is None:
            is_normal = self.check_normality()

        if is_normal:
            ttest_rel=st.ttest_rel(
                    self.data[self.columns[0]],
                    self.data[self.columns[1]],
                    nan_policy=nan_policy,
                    alternative=alternative)
            print(f't-test:')
            p_value = ttest_rel.pvalue
            if p_value > self.alpha:
                print(f'pvalue {p_value} > alpha: failed to reject H0 (mean values can be equal)')
            else:
                print(f'pvalue {p_value} =< alpha: reject H0 (mean values are not equal)')

            self.visualize_plot(test_name="t-test")
            return {'pvalue': ttest_rel.pvalue, 'statistic': ttest_rel.statistic}
        else:
            diff_rounded = np.round(self.diff, decimals=5)
            wilcoxon = st.wilcoxon(diff_rounded,
                                   alternative=alternative)
            print(f'Wilcoxon test:')
            p_value = wilcoxon.pvalue
            if p_value > self.alpha:
                print(f'pvalue {p_value} > alpha: failed to reject H0 (median values can be equal)')
            else:
                print(f'pvalue {p_value} =< alpha: reject H0 (median values are not equal)')
            
            self.visualize_plot(test_name="wilcoxon")
            return {'pvalue': wilcoxon.pvalue, 'statistic': wilcoxon.statistic}