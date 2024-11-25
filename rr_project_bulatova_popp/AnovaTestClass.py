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
            
        super().__init__(data, columns, alpha)

    
        
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
    
    def run_test(self, visualize: bool = True, check_variance: bool = False) -> None:
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
                if check_variance:
                    is_homoscedasticity = self.check_homoscedasticity()
                    if is_homoscedasticity:
                        self.perform_test(f_oneway, visualize)
                    else:
                        self.perform_test(alexandergovern, visualize)
                else: 
                    self.perform_test(f_oneway, visualize)               
            else:
                self.perform_test(kruskal, visualize)
        except Exception as e:
            print(f"Error during multi-group test: {e}")