"""
Statistical hypothesis testing framework for photonic AI research.

Provides robust statistical validation tools for experimental methodology,
including power analysis, effect size calculation, and multiple comparison corrections.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available, some statistical tests will be limited")

class TestType(Enum):
    """Statistical test types."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"

class EffectSizeType(Enum):
    """Effect size calculation methods."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFFS_DELTA = "cliffs_delta"
    ETA_SQUARED = "eta_squared"

@dataclass
class TestResult:
    """Statistical test result."""
    test_type: TestType
    statistic: float
    p_value: float
    effect_size: float
    effect_size_type: EffectSizeType
    confidence_interval: Tuple[float, float]
    power: float
    sample_sizes: List[int]
    interpretation: str

class HypothesisTest:
    """
    Advanced hypothesis testing framework.
    
    Provides comprehensive statistical testing capabilities with proper
    effect size calculation, power analysis, and multiple comparison correction.
    """
    
    def __init__(self, alpha: float = 0.05, power_threshold: float = 0.8):
        self.alpha = alpha
        self.power_threshold = power_threshold
        
    def t_test(self, 
               group1: np.ndarray, 
               group2: np.ndarray,
               paired: bool = False,
               alternative: str = 'two-sided') -> TestResult:
        """
        Perform t-test with effect size calculation.
        
        Args:
            group1: First group of measurements
            group2: Second group of measurements  
            paired: Whether to perform paired t-test
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            TestResult with comprehensive statistics
        """
        if not SCIPY_AVAILABLE:
            return self._fallback_t_test(group1, group2, paired)
            
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
        else:
            statistic, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
            
        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(group1, group2, paired)
        
        # Calculate confidence interval
        ci = self._calculate_ci_cohens_d(group1, group2, paired)
        
        # Calculate power
        power = self._calculate_power_t_test(group1, group2, effect_size)
        
        # Interpretation
        interpretation = self._interpret_t_test(p_value, effect_size, power)
        
        return TestResult(
            test_type=TestType.T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=EffectSizeType.COHENS_D,
            confidence_interval=ci,
            power=power,
            sample_sizes=[len(group1), len(group2)],
            interpretation=interpretation
        )
    
    def mann_whitney_u(self, 
                      group1: np.ndarray, 
                      group2: np.ndarray,
                      alternative: str = 'two-sided') -> TestResult:
        """
        Non-parametric Mann-Whitney U test.
        
        Args:
            group1: First group of measurements
            group2: Second group of measurements
            alternative: Alternative hypothesis
            
        Returns:
            TestResult with Cliff's delta effect size
        """
        if not SCIPY_AVAILABLE:
            return self._fallback_mann_whitney(group1, group2)
            
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )
        
        # Calculate Cliff's delta
        effect_size = self._calculate_cliffs_delta(group1, group2)
        
        # Bootstrap confidence interval for Cliff's delta
        ci = self._bootstrap_cliffs_delta_ci(group1, group2)
        
        # Approximate power calculation
        power = self._approximate_power_mann_whitney(group1, group2, effect_size)
        
        interpretation = self._interpret_mann_whitney(p_value, effect_size, power)
        
        return TestResult(
            test_type=TestType.MANN_WHITNEY,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=EffectSizeType.CLIFFS_DELTA,
            confidence_interval=ci,
            power=power,
            sample_sizes=[len(group1), len(group2)],
            interpretation=interpretation
        )
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> float:
        """Calculate Cohen's d effect size."""
        if paired:
            diff = group1 - group2
            return np.mean(diff) / np.std(diff, ddof=1)
        else:
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                                 (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _calculate_cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta effect size."""
        n1, n2 = len(group1), len(group2)
        dominance = 0
        
        for x in group1:
            for y in group2:
                if x > y:
                    dominance += 1
                elif x < y:
                    dominance -= 1
                    
        return dominance / (n1 * n2)
    
    def _calculate_ci_cohens_d(self, group1: np.ndarray, group2: np.ndarray, 
                              paired: bool = False, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for Cohen's d."""
        n1, n2 = len(group1), len(group2)
        d = self._calculate_cohens_d(group1, group2, paired)
        
        if paired:
            n = n1
            se = np.sqrt((1/n) + (d**2)/(2*n))
        else:
            se = np.sqrt((n1 + n2)/(n1 * n2) + d**2/(2*(n1 + n2)))
            
        alpha = 1 - confidence
        if SCIPY_AVAILABLE:
            t_crit = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
        else:
            # Approximate for large samples
            t_crit = 1.96
            
        margin = t_crit * se
        return (d - margin, d + margin)
    
    def _bootstrap_cliffs_delta_ci(self, group1: np.ndarray, group2: np.ndarray,
                                  n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for Cliff's delta."""
        deltas = []
        
        for _ in range(n_bootstrap):
            boot_group1 = np.random.choice(group1, len(group1), replace=True)
            boot_group2 = np.random.choice(group2, len(group2), replace=True)
            deltas.append(self._calculate_cliffs_delta(boot_group1, boot_group2))
            
        alpha = 1 - confidence
        lower = np.percentile(deltas, 100 * alpha/2)
        upper = np.percentile(deltas, 100 * (1 - alpha/2))
        
        return (lower, upper)
    
    def _calculate_power_t_test(self, group1: np.ndarray, group2: np.ndarray, effect_size: float) -> float:
        """Calculate statistical power for t-test."""
        n1, n2 = len(group1), len(group2)
        
        if not SCIPY_AVAILABLE:
            # Simplified power approximation
            n_total = n1 + n2
            return min(0.99, max(0.05, 0.5 + abs(effect_size) * np.sqrt(n_total) / 4))
        
        # More accurate power calculation would require additional statistical packages
        # For now, use approximation based on effect size and sample size
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        ncp = effect_size * np.sqrt(n_harmonic / 2)
        
        # Approximate power using normal distribution
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = ncp - z_alpha
        power = 1 - stats.norm.cdf(z_beta)
        
        return min(0.99, max(0.05, power))
    
    def _approximate_power_mann_whitney(self, group1: np.ndarray, group2: np.ndarray, effect_size: float) -> float:
        """Approximate power for Mann-Whitney U test."""
        n1, n2 = len(group1), len(group2)
        n_total = n1 + n2
        
        # Approximate power based on sample size and effect size
        power_approx = min(0.99, max(0.05, 0.4 + abs(effect_size) * np.sqrt(n_total) / 3))
        return power_approx
    
    def _interpret_t_test(self, p_value: float, effect_size: float, power: float) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if abs(effect_size) < 0.2:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.5:
            effect_desc = "small"
        elif abs(effect_size) < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
            
        power_desc = "adequate" if power >= self.power_threshold else "inadequate"
        
        return f"Result is {significance} (p={p_value:.4f}) with {effect_desc} effect size (d={effect_size:.3f}). Statistical power is {power_desc} ({power:.3f})."
    
    def _interpret_mann_whitney(self, p_value: float, effect_size: float, power: float) -> str:
        """Interpret Mann-Whitney U test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if abs(effect_size) < 0.147:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.33:
            effect_desc = "small"
        elif abs(effect_size) < 0.474:
            effect_desc = "medium"
        else:
            effect_desc = "large"
            
        power_desc = "adequate" if power >= self.power_threshold else "inadequate"
        
        return f"Result is {significance} (p={p_value:.4f}) with {effect_desc} effect size (δ={effect_size:.3f}). Statistical power is {power_desc} ({power:.3f})."
    
    def _fallback_t_test(self, group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> TestResult:
        """Fallback t-test implementation when SciPy is not available."""
        if paired:
            diff = group1 - group2
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            n = len(diff)
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            df = n - 1
        else:
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            
            t_stat = (mean1 - mean2) / se
            df = n1 + n2 - 2
        
        # Approximate p-value for large df
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        effect_size = self._calculate_cohens_d(group1, group2, paired)
        power = self._calculate_power_t_test(group1, group2, effect_size)
        ci = self._calculate_ci_cohens_d(group1, group2, paired)
        interpretation = self._interpret_t_test(p_value, effect_size, power)
        
        return TestResult(
            test_type=TestType.T_TEST,
            statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=EffectSizeType.COHENS_D,
            confidence_interval=ci,
            power=power,
            sample_sizes=[len(group1), len(group2)],
            interpretation=interpretation
        )
    
    def _fallback_mann_whitney(self, group1: np.ndarray, group2: np.ndarray) -> TestResult:
        """Fallback Mann-Whitney implementation when SciPy is not available."""
        # Simple rank-based test
        n1, n2 = len(group1), len(group2)
        combined = np.concatenate([group1, group2])
        ranks = stats.rankdata(combined) if SCIPY_AVAILABLE else self._rank_data(combined)
        
        r1 = np.sum(ranks[:n1])
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        
        u_stat = min(u1, u2)
        
        # Normal approximation for p-value
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u_stat - mean_u) / std_u
        p_value = 2 * (1 - self._normal_cdf(abs(z)))
        
        effect_size = self._calculate_cliffs_delta(group1, group2)
        power = self._approximate_power_mann_whitney(group1, group2, effect_size)
        ci = self._bootstrap_cliffs_delta_ci(group1, group2)
        interpretation = self._interpret_mann_whitney(p_value, effect_size, power)
        
        return TestResult(
            test_type=TestType.MANN_WHITNEY,
            statistic=u_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=EffectSizeType.CLIFFS_DELTA,
            confidence_interval=ci,
            power=power,
            sample_sizes=[n1, n2],
            interpretation=interpretation
        )
    
    def _rank_data(self, data: np.ndarray) -> np.ndarray:
        """Simple ranking function when SciPy is not available."""
        sorted_indices = np.argsort(data)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(1, len(data) + 1)
        return ranks
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

class StatisticalValidator:
    """
    Comprehensive statistical validation framework.
    
    Provides multiple comparison corrections, power analysis, and 
    experimental design validation for photonic AI research.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.hypothesis_test = HypothesisTest(alpha=alpha)
    
    def bonferroni_correction(self, p_values: List[float]) -> Tuple[List[float], float]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of uncorrected p-values
            
        Returns:
            Tuple of (corrected_p_values, adjusted_alpha)
        """
        n_tests = len(p_values)
        adjusted_alpha = self.alpha / n_tests
        corrected_p_values = [min(1.0, p * n_tests) for p in p_values]
        
        return corrected_p_values, adjusted_alpha
    
    def holm_sidak_correction(self, p_values: List[float]) -> Tuple[List[bool], float]:
        """
        Apply Holm-Šídák step-down correction.
        
        Args:
            p_values: List of uncorrected p-values
            
        Returns:
            Tuple of (significance_flags, adjusted_alpha)
        """
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        significance = [False] * n_tests
        
        for i, idx in enumerate(sorted_indices):
            adjusted_alpha = 1 - (1 - self.alpha)**(1/(n_tests - i))
            if p_values[idx] <= adjusted_alpha:
                significance[idx] = True
            else:
                break
                
        return significance, adjusted_alpha
    
    def validate_experimental_design(self, 
                                   sample_sizes: List[int],
                                   expected_effect_size: float,
                                   test_type: TestType = TestType.T_TEST) -> Dict[str, Union[float, bool, str]]:
        """
        Validate experimental design for adequate statistical power.
        
        Args:
            sample_sizes: Sample sizes for each group
            expected_effect_size: Expected effect size
            test_type: Type of statistical test to be performed
            
        Returns:
            Validation results with power analysis and recommendations
        """
        results = {
            'adequate_power': False,
            'estimated_power': 0.0,
            'minimum_sample_size': 0,
            'recommendations': []
        }
        
        # Calculate power based on test type
        if test_type == TestType.T_TEST:
            if len(sample_sizes) >= 2:
                n1, n2 = sample_sizes[0], sample_sizes[1]
                n_harmonic = 2 * n1 * n2 / (n1 + n2)
                
                # Power approximation for t-test
                if SCIPY_AVAILABLE:
                    ncp = expected_effect_size * np.sqrt(n_harmonic / 2)
                    z_alpha = stats.norm.ppf(1 - self.alpha/2)
                    z_beta = ncp - z_alpha
                    power = 1 - stats.norm.cdf(z_beta)
                else:
                    power = min(0.99, max(0.05, 0.5 + abs(expected_effect_size) * np.sqrt(n_harmonic) / 4))
                    
                results['estimated_power'] = power
                results['adequate_power'] = power >= 0.8
                
                # Calculate minimum sample size for 80% power
                if power < 0.8:
                    target_n = self._calculate_minimum_sample_size(expected_effect_size, 0.8, test_type)
                    results['minimum_sample_size'] = target_n
                    results['recommendations'].append(f"Increase sample size to at least {target_n} per group for 80% power")
        
        # Add general recommendations
        if results['estimated_power'] < 0.5:
            results['recommendations'].append("Statistical power is critically low - consider increasing effect size or sample size")
        elif results['estimated_power'] < 0.8:
            results['recommendations'].append("Statistical power is below recommended threshold of 80%")
            
        if abs(expected_effect_size) < 0.2:
            results['recommendations'].append("Expected effect size is very small - ensure measurement precision is adequate")
            
        return results
    
    def _calculate_minimum_sample_size(self, effect_size: float, target_power: float, test_type: TestType) -> int:
        """Calculate minimum sample size for target power."""
        if test_type == TestType.T_TEST:
            # Approximate calculation for t-test
            if SCIPY_AVAILABLE:
                z_alpha = stats.norm.ppf(1 - self.alpha/2)
                z_beta = stats.norm.ppf(target_power)
                n = 2 * ((z_alpha + z_beta) / effect_size)**2
            else:
                # Simplified approximation
                n = 16 / (effect_size**2)
                
            return int(np.ceil(n))
        
        # Default fallback
        return 50