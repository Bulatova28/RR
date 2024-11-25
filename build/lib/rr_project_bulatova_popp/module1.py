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

class StatisticalTest(ABC):
    def __init__(self, alpha: float = 0.05) -> None:
       self.alpha=alpha 
    @abstractmethod
    def descriptive_stats(self) -> pd.DataFrame:
        pass
    @abstractmethod
    def check_normality(self) -> bool:
        pass
    @abstractmethod
    def normality_visualization(self, vis_func) -> None:
        pass
    @abstractmethod
    def run_test(self) -> dict[str, float]|None:
        pass
        

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

        self.data = data[columns] 
        self.columns=columns
        self.alpha=alpha

    @property
    def diff(self) -> pd.Series:
        """
        Calculates the differences between the paired samples.

        Returns:
            pd.Series: The differences between the two samples (column1 - column2).
        """

        return self.data[self.columns[0]] - self.data[self.columns[1]]

    def descriptive_stats(self) -> pd.DataFrame:
        """
        Computes descriptive statistics for the paired samples.

        Returns:
            pd.DataFrame: A DataFrame containing descriptive statistics for each sample, 
                          including mean, standard deviation, variance, minimum, maximum, range, mode, median, kurtosis, and skewness.
        """

        data_frame = {}
        for col_name in self.columns:
            arr = self.data[col_name]
            data_frame[col_name] = {
                'mean': np.mean(arr),
                'std': np.std(arr, ddof = 1),
                'var': np.var(arr, ddof = 1),
                'min': np.min(arr),
                'max': np.max(arr),
                'range': np.max(arr) - np.min(arr),
                'mode': st.mode(arr),
                'median': np.median(arr),
                'kurtosis': st.kurtosis(arr),
                'skewness': st.skew(arr)
            }
        return pd.DataFrame(data_frame)
    
    def check_normality(self) -> bool:
        """
        Performs a normality test on the differences between paired samples.

        Returns:
            bool: True if the differences follow a normal distribution (parametric test is appropriate),
                  False otherwise (non-parametric test is appropriate).
        """

        if len(self.diff) < 5000:
            p_value = st.shapiro(self.diff).pvalue
            t_stat = st.shapiro(self.diff).statistic
            test_name = 'Shapiro-Wilk test'
        else:
            p_value = st.normaltest(self.diff).pvalue
            t_stat = st.normaltest(self.diff).statistic 
            test_name = 'Normal test'

        self.p_value = p_value

        print(f'{test_name} results for sample: p-value = {p_value}, stat = {t_stat}')
        if p_value > self.alpha:
            print(f'pvalue > alpha: failed to reject H0, so it will be a parametric version of T-test')
        else:
            print(f'pvalue =< alpha: reject H0, so it will be a non-parametric version of T-test')

        return p_value > self.alpha
    
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
    

    def run_paired_test(self, alternative = "two-sided", nan_policy="omit", is_normal=None) -> dict[str, float]:
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

        self.data = data[columns] 
        self.columns=columns
        self.alpha=alpha

    def descriptive_stats(self) -> pd.DataFrame:
        """
        Computes descriptive statistics for the two samples.

        Returns:
            pd.DataFrame: A dataframe containing statistics like mean, standard deviation, variance, 
                          minimum, maximum, range, mode, median, kurtosis, and skewness for each sample.
        """

        data_frame = {}
        for col_name in self.columns:
            arr = self.data[col_name] 
            data_frame[col_name] = {
                'mean': np.mean(arr),
                'std': np.std(arr, ddof = 1),
                'var': np.var(arr, ddof = 1),
                'min': np.min(arr),
                'max': np.max(arr),
                'range': np.max(arr) - np.min(arr),
                'mode': st.mode(arr), 
                'median': np.median(arr),
                'kurtosis': st.kurtosis(arr),
                'skewness': st.skew(arr)
            }
        return pd.DataFrame(data_frame)
    
    def check_normality(self, column: str=None) -> bool:
        """
        Performs a normality test on the specified column.

        Args:
            column (str): The column name for which the normality test will be performed.

        Returns:
            bool: True if the data is normally distributed, False otherwise.

        Raises:
            ValueError: If the column name is invalid or not provided.
        """

        if column:
            if column not in self.columns:
                raise ValueError(f'Invalid column name. Available columns: {self.columns}')
            sample = self.data[column]
        else:
            raise ValueError('Column name must be provided for normality check.')

        if len(sample) < 5000:
            p_value = st.shapiro(sample).pvalue
            t_stat = st.shapiro(sample).statistic
            test_name = 'Shapiro-Wilk test'
        else:
            p_value = st.normaltest(sample).pvalue
            t_stat = st.normaltest(sample).statistic 
            test_name = 'Normal test'

        print(f'{test_name} results for column "{column}": p-value = {p_value}, stat = {t_stat}')
        if p_value > self.alpha:
            print(f'pvalue > alpha: failed to reject H0, so it will be a parametric version of T-test')
            return True
        else:
            print(f'pvalue =< alpha: reject H0, so it will be a non-parametric version of T-test')
            return False
        

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

    def run_ind_test(self, alternative = "two-sided", equal_var = False, nan_policy="omit") -> dict[str, float]:
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

        is_normal_1 = self.check_normality(self.columns[0])
        is_normal_2 = self.check_normality(self.columns[1])

        if is_normal_1 and is_normal_2:
            ttest_ind = st.ttest_ind(
                    sample_1,
                    sample_2,
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


class MultiGroupTest(StatisticalTest):
    """
    A class for performing multi-group statistical tests and visualizations on datasets.

    This class provides:
    - Normality checks using Shapiro-Wilk or D’Agostino and Pearson’s tests.
    - Homoscedasticity checks using Bartlett's test.
    - Visualization of distributions using histograms, Q-Q plots, joyplots, and boxplots.
    - Execution of multi-group comparison tests (ANOVA, Alexander-Govern, Kruskal-Wallis).

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame containing the data for testing.
    - columns (list[str]): List of column names in the DataFrame to be used for testing.
    - alpha (float): The significance level for hypothesis testing. Default is 0.05.
    
    """
    def __init__(self, data: pd.DataFrame, columns: list[str], alpha: float = 0.05):
        """
        Initialize the MultiGroupTest class with data and parameters.
    
        Parameters:
        - data (pd.DataFrame): A pandas DataFrame containing the data for testing.
        - columns (list[str]): List of column names in the DataFrame to be used for testing.
        - alpha (float): The significance level for hypothesis testing. Default is 0.05.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input 'data' must be a pandas DataFrame.")
            
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise ValueError("The 'columns' must be a list of strings.")
            
        if len(columns) < 2:
            raise ValueError(f'At least two columns must be provided for multi-group testing.')
            
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
            
        if not isinstance(alpha, (float, int)) or not (0 < alpha < 1):
            raise ValueError("The 'alpha' must be a float between 0 and 1.")
            
        self.data = data
        self.columns = columns
        self.alpha = alpha

    def descriptive_stats(self) -> pd.DataFrame:
        """
        Computes descriptive statistics for multiple groups in the dataset.

        Returns:
            pd.DataFrame: A dataframe containing statistics like mean, standard deviation, variance,
                        minimum, maximum, range, mode, median, kurtosis, and skewness for each group.
        """
        data_frame = {}
        for col_name in self.columns:
            arr = self.data[col_name].dropna()  # Уникаємо NaN значень
            data_frame[col_name] = {
                'mean': np.mean(arr),
                'std': np.std(arr, ddof=1),
                'var': np.var(arr, ddof=1),
                'min': np.min(arr),
                'max': np.max(arr),
                'range': np.max(arr) - np.min(arr),
                'mode': st.mode(arr),
                'median': np.median(arr),
                'kurtosis': st.kurtosis(arr, fisher=True, bias=False),
                'skewness': st.skew(arr, bias=False)
            }
        return pd.DataFrame(data_frame)
        
    def check_normality(self) -> bool:
        """
        Checks the normality of the samples in the dataset using Shapiro-Wilk 
        or D’Agostino and Pearson’s normality test based on sample size.
    
        Returns:
        - bool: True if all samples are normally distributed, otherwise False.
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
                results[column] = p_shapiro > self.alpha
                print(f'The test results: p-value = {p_shapiro}, stat = {stat}\n')
                print(f'The sample "{column}": {'is normally distributed' if results[column] else 'is not normally distributed'}\n')
            else: 
                stat, p_normal = st.normaltest(self.data[column])
                results[column] = p_normal > self.alpha 
                print(f'The test results: p-value = {p_normal}, stat = {stat}\n')
                print(f'The sample "{column}": {'is normally distributed' if results[column] else 'is not normally distributed'}\n')
    
        normality = all(results.values())        
        
        if  normality:
            print(f'All THE SAMPLES ARE NORMALLY DISTRIBUTED\n')
        else:
            print(f'Any of the samples is not normally distributed\n')
        return normality
        

    def normality_visualization(self, vis_func: str = 'histogram') -> None:
        """
        Visualizes the normality of data columns using either histograms or Q-Q plots.
    
        Parameters:
        - vis_func (str): Type of visualization. Options are:
            - 'histogram': Displays a histogram of the data with the ideal normal PDF.
            - 'probplot': Displays a Q-Q plot for the data.
        
        Raises:
        - ValueError: If an unsupported value for vis_func is provided.
        """
        num_columns = len(self.columns)
        cols_per_row = 2
        num_rows = int(np.ceil(num_columns / cols_per_row))
    
        fig, axs = plt.subplots(num_rows, cols_per_row, figsize=(14, 5 * num_rows))
        axs = axs.flatten()
    
        for i, column in enumerate(self.columns):
            original_data = self.data[column].dropna()  
            mean = original_data.mean()
            std = original_data.std()
            x = np.linspace(original_data.min(), original_data.max(), 1000)
            ideal_pdf = norm.pdf(x, loc=mean, scale=std)
            if vis_func == 'histogram':
                sns.histplot(original_data, kde=True, color='blue', bins=30, label=f'{column} Original', stat='density', alpha=0.6, ax=axs[i])
                sns.lineplot(x=x, y=ideal_pdf, color='red', lw=2, label="Ideal Normal (PDF)", ax=axs[i])
                axs[i].set_title(f"{column} Distribution", fontsize=12)
                axs[i].set_xlabel("Value", fontsize=10)
                axs[i].set_ylabel("Density", fontsize=10)
                axs[i].legend()
            elif vis_func == 'probplot':
                st.probplot(original_data, dist="norm", plot=axs[i])
                axs[i].set_title(f"{column} Q-Q Plot", fontsize=12)
                axs[i].set_xlabel("Quantiles", fontsize=10)
                axs[i].set_ylabel("Values", fontsize=10)
            else:
                raise ValueError("vis_func must be 'histogram' or 'probplot'")     
        for j in range(num_columns, len(axs)):
            fig.delaxes(axs[j])
    
        plt.tight_layout()
        plt.show()
        
    def check_homoscedasticity(self) -> bool:
        """
        Checks the homoscedasticity of the samples in the dataset using Bartlett's test.
        Returns:
        - bool: True if the data is homoscedastic (variances are equal), otherwise False.
        """
        samples = [self.data[column] for column in self.columns]
        stat, p_bartlett = bartlett(*samples)
        print('---The Bartlett test has been performed---\n')
        print('H0: all input samples are from populations with equal variances\n')
        print(f'The test results: p-value = {p_bartlett}, stat = {stat}\n')
        homoscedasticity = p_bartlett > self.alpha
        if homoscedasticity:
            print("Large p-value suggests that our data is homogeneous (the population standard deviations of the groups are all equal)\n")
        else:
            print("Small p-value suggests that populations do not have equal variances\n")
        return homoscedasticity

    def homoscedasticity_visualization1(self) -> None:
        """
        Visualizes the spread of data across the specified columns using boxplots.
    
        Returns:
            None: This function is used for visualization purposes only.
        """
       
      
        melted_data = self.data.melt(var_name="Columns", value_name="Values")
        
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.boxplot(x="Columns", y="Values", data=melted_data)
        
        ax.set_title("Boxplots for Selected Columns", fontsize=16)
        ax.set_xlabel("Samples", fontsize=12)
        ax.set_ylabel("Values", fontsize=12)
        plt.tight_layout()
        plt.show()

    def joyplot_visualize(self, test_type: str) -> None:
        """
        Visualizes the distribution of columns using a joyplot and highlights the mean or median
        depending on the type of statistical test performed.
    
        Parameters:
        - test_type (str): The type of statistical test. Accepted values are:
            - 'f_oneway': Highlights the mean for each column.
            - 'alexandergovern': Highlights the mean for each column.
            - 'kruskal': Highlights the median for each column.
    
        Raises:
        - ValueError: If an unsupported test type is provided.
        """
        if test_type in ['f_oneway', 'alexandergovern']:
            stat_name = "Mean"
            stat_func = lambda x: x.mean()
        elif test_type == 'kruskal':
            stat_name = "Median"
            stat_func = lambda x: np.median(x)
        else:
            raise ValueError("Unsupported test type. Use 'f_oneway', 'alexandergovern', or 'kruskal'.")

   
        fig, axs = joyplot(
            data=self.data[self.columns],
            figsize=(14, 2.5 * len(self.columns)),
            overlap=0.4,
            fade=True,
            title=f"Distribution of Samples with Highlighted {stat_name}s"
        )
        
        for i, column in enumerate(self.columns):
            stat_value = stat_func(self.data[column].dropna())
            axs[i].axvline(stat_value, color="red", zorder=5, linestyle="--", label=f"{stat_name} ({column}): {stat_value:.2f}") 
            axs[i].legend(loc="upper right")

        plt.show()
   

    def perform_test(self, test_func: Callable, visualize: bool) -> None:
        """
        Performs a specified statistical test on the dataset columns and visualizes the results if needed.
        
        Parameters:
        - test_func (Callable): A statistical test function (e.g., f_oneway, alexandergovern, kruskal).
        - visualize (bool): If True, visualizes the distribution of groups using a joyplot.
        
        Returns:
        - None: This function does not return a value but prints the results and conclusions.
        """
        samples = [self.data[column] for column in self.columns]
        result = test_func(*samples)
        if hasattr(result, "statistic") and hasattr(result, "pvalue"):
            stat, p_value = result.statistic, result.pvalue
        else:
            raise TypeError(f"Unexpected return type from {test_func.__name__}: {type(result)}")
        
        if test_func.__name__ == "f_oneway":
            if any(len(self.data[column]) == 0 for column in self.columns):
                raise ValueError("All groups must have at least one value for ANOVA.")
            if all(len(self.data[col]) <= 1 for col in self.columns):
                raise ValueError("At least one group must have more than one value for ANOVA.")
            unique_values = [np.unique(self.data[col]) for col in self.columns]
            if all(len(values) == 1 for values in unique_values) and len(set(val[0] for val in unique_values)) == 1:
                raise ValueError("All groups contain identical values; ANOVA cannot be performed.")
            print('---The one-way ANOVA test has been performed---\n')
            print('H0: Two or more groups have the same population mean\n')
            if visualize:
                self.joyplot_visualize(test_type='f_oneway')
        
        elif test_func.__name__ == "alexandergovern":
            print('---The Alexander-Govern test has been performed---\n')
            print('H0: The population means of all the groups are equal (relaxing the assumption of equal variances)\n')
            if visualize:
                self.joyplot_visualize(test_type='alexandergovern')
        
        elif test_func.__name__ == "kruskal":
            # Перевірка для Kruskal-Wallis
            if any(len(self.data[column]) < 5 for column in self.columns):
                raise ValueError("All groups must have at least 5 measurements for Kruskal-Wallis.")
            print('---The Kruskal-Wallis H-test has been performed---\n')
            print('H0: The population medians of all the groups are equal\n')
            if visualize:
                self.joyplot_visualize(test_type='kruskal')
    
        print(f'Test results: p-value = {p_value:.4f}, statistic = {stat:.4f}\n')
        if p_value > self.alpha:
            print("Conclusion: Failed to reject the null hypothesis, which means there is no statistically significant difference between the groups.\n")
        else:
            print("Conclusion: Rejected the null hypothesis.\n")
    
    def run_multi_group_test(self, visualize: bool = True) -> None:
        """
        Runs a multi-group comparison test based on the data's normality and homoscedasticity.
        
        Parameters:
        - visualize (bool): If True, visualizes the distribution of groups using a joyplot during the test.
        
        Behavior:
        - Checks if all data groups are normally distributed using `check_normality`.
        - If all groups are normal:
            - Checks homoscedasticity using `check_homoscedasticity`.
            - If variances are equal, runs a one-way ANOVA test (`f_oneway`).
            - If variances are not equal, runs the Alexander-Govern test.
        - If not all groups are normal, runs the Kruskal-Wallis H-test.
        """
        try:
            is_all_normal = self.check_normality()
            if is_all_normal:
                is_homoscedasticity = self.check_homoscedasticity()
                if is_homoscedasticity:
                    self.perform_test(f_oneway, visualize)
                else:
                    self.perform_test(alexandergovern, visualize)
            else:
                self.perform_test(kruskal, visualize)
        except Exception as e:
            print(f"Error during multi-group test: {e}")