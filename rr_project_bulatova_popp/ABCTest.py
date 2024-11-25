import scipy.stats as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class StatisticalTest(ABC):
    def __init__(self, data: pd.DataFrame, columns: list[str], alpha: float = 0.05) -> None:
       self.data = data[columns] 
       self.columns=columns
       self.alpha=alpha
   
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
    
    @abstractmethod
    def check_normality(self) -> bool:
        raise NotImplementedError()
    @abstractmethod
    def normality_visualization(self, vis_func) -> None:
        raise NotImplementedError()
    @abstractmethod
    def run_test(self) -> dict[str, float]|None:
        raise NotImplementedError()