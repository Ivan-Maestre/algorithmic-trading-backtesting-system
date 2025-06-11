"""
Data Validator Module

Provides data quality validation and integrity checks for OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from ..config import Config


class DataValidator:
    """
    Validates OHLCV data quality and integrity.
    
    This class provides comprehensive validation checks for financial data
    to ensure it meets quality standards for backtesting.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_ohlcv_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Perform comprehensive validation on OHLCV data.
        
        Args:
            df: OHLCV DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Starting comprehensive OHLCV data validation")
        
        results = {
            'structure_valid': self._validate_structure(df),
            'price_integrity': self._validate_price_integrity(df),
            'volume_valid': self._validate_volume(df),
            'completeness': self._validate_completeness(df),
            'timestamp_valid': self._validate_timestamps(df),
            'no_extreme_outliers': self._validate_outliers(df),
            'sufficient_data': self._validate_data_sufficiency(df)
        }
        
        self.validation_results = results
        
        # Overall validation result
        all_valid = all(results.values())
        results['overall_valid'] = all_valid
        
        self.logger.info(f"Validation completed. Overall valid: {all_valid}")
        return results
    
    def _validate_structure(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame structure and required columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        try:
            # Check if DataFrame is not empty
            if df.empty:
                self.logger.error("DataFrame is empty")
                return False
            
            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.logger.error(f"Column {col} is not numeric")
                    return False
            
            # Check index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error("Index is not DatetimeIndex")
                return False
            
            self.logger.info("Structure validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Structure validation failed: {e}")
            return False
    
    def _validate_price_integrity(self, df: pd.DataFrame) -> bool:
        """Validate OHLC price relationships."""
        try:
            # Check High >= max(Open, Close)
            high_valid = (df['high'] >= np.maximum(df['open'], df['close'])).all()
            if not high_valid:
                invalid_count = (~(df['high'] >= np.maximum(df['open'], df['close']))).sum()
                self.logger.warning(f"High price integrity violated in {invalid_count} records")
                return False
            
            # Check Low <= min(Open, Close)
            low_valid = (df['low'] <= np.minimum(df['open'], df['close'])).all()
            if not low_valid:
                invalid_count = (~(df['low'] <= np.minimum(df['open'], df['close']))).sum()
                self.logger.warning(f"Low price integrity violated in {invalid_count} records")
                return False
            
            # Check no negative prices
            negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
            if negative_prices > 0:
                self.logger.error(f"Found {negative_prices} records with negative or zero prices")
                return False
            
            self.logger.info("Price integrity validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Price integrity validation failed: {e}")
            return False
    
    def _validate_volume(self, df: pd.DataFrame) -> bool:
        """Validate volume data."""
        try:
            # Check volume is non-negative
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                self.logger.error(f"Found {negative_volume} records with negative volume")
                return False
            
            # Check for excessive zero volume periods
            zero_volume_pct = (df['volume'] == 0).mean()
            if zero_volume_pct > 0.1:  # More than 10% zero volume
                self.logger.warning(f"High percentage of zero volume periods: {zero_volume_pct:.2%}")
                # Don't fail validation, just warn
            
            self.logger.info("Volume validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Volume validation failed: {e}")
            return False
    
    def _validate_completeness(self, df: pd.DataFrame) -> bool:
        """Validate data completeness."""
        try:
            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.any():
                self.logger.warning(f"Missing data found: {missing_data.to_dict()}")
                # Allow some missing data, but warn
                total_missing_pct = missing_data.sum() / (len(df) * len(df.columns))
                if total_missing_pct > 0.05:  # More than 5% missing
                    self.logger.error(f"Excessive missing data: {total_missing_pct:.2%}")
                    return False
            
            self.logger.info("Completeness validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Completeness validation failed: {e}")
            return False
    
    def _validate_timestamps(self, df: pd.DataFrame) -> bool:
        """Validate timestamp consistency."""
        try:
            # Check for duplicate timestamps
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                self.logger.error(f"Found {duplicates} duplicate timestamps")
                return False
            
            # Check if data is sorted
            if not df.index.is_monotonic_increasing:
                self.logger.error("Timestamps are not sorted")
                return False
            
            # Check for reasonable time gaps
            if len(df) > 1:
                time_diffs = df.index.to_series().diff()[1:]
                
                # Expected interval based on timeframe
                timeframe = Config.TIMEFRAME
                if timeframe == '4h':
                    expected_interval = timedelta(hours=4)
                elif timeframe == '1h':
                    expected_interval = timedelta(hours=1)
                elif timeframe == '1d':
                    expected_interval = timedelta(days=1)
                else:
                    expected_interval = timedelta(hours=4)  # Default
                
                # Check for gaps larger than 2x expected interval
                large_gaps = (time_diffs > expected_interval * 2).sum()
                if large_gaps > len(df) * 0.01:  # More than 1% large gaps
                    self.logger.warning(f"Found {large_gaps} large time gaps")
            
            self.logger.info("Timestamp validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Timestamp validation failed: {e}")
            return False
    
    def _validate_outliers(self, df: pd.DataFrame) -> bool:
        """Validate for extreme outliers."""
        try:
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                # Calculate percentage changes
                pct_changes = df[col].pct_change().abs()
                
                # Check for extreme moves (>500% in one period)
                extreme_moves = (pct_changes > 5.0).sum()
                if extreme_moves > 0:
                    self.logger.warning(f"Found {extreme_moves} extreme price moves in {col}")
                    # Don't fail validation for crypto data, just warn
                
                # Check for unrealistic price levels
                if col == 'close':
                    # For BTC, check if price is within reasonable range
                    if df[col].min() < 1 or df[col].max() > 1000000:
                        self.logger.warning(f"Price levels seem unrealistic: {df[col].min()} - {df[col].max()}")
            
            self.logger.info("Outlier validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Outlier validation failed: {e}")
            return False
    
    def _validate_data_sufficiency(self, df: pd.DataFrame) -> bool:
        """Validate that there's sufficient data for backtesting."""
        try:
            # Check minimum number of records
            min_records = 1000  # Minimum for meaningful backtesting
            if len(df) < min_records:
                self.logger.error(f"Insufficient data: {len(df)} records (minimum: {min_records})")
                return False
            
            # Check date range coverage
            start_date = pd.to_datetime(Config.DATA_START_DATE)
            end_date = pd.to_datetime(Config.DATA_END_DATE)
            
            data_start = df.index.min()
            data_end = df.index.max()
            
            # Allow some tolerance in date coverage
            if data_start > start_date + timedelta(days=30):
                self.logger.warning(f"Data starts later than expected: {data_start} vs {start_date}")
            
            if data_end < end_date - timedelta(days=30):
                self.logger.warning(f"Data ends earlier than expected: {data_end} vs {end_date}")
            
            # Calculate coverage percentage
            expected_days = (end_date - start_date).days
            actual_days = (data_end - data_start).days
            coverage = actual_days / expected_days
            
            if coverage < 0.9:  # Less than 90% coverage
                self.logger.warning(f"Low data coverage: {coverage:.1%}")
            
            self.logger.info(f"Data sufficiency validation passed. Records: {len(df)}, Coverage: {coverage:.1%}")
            return True
            
        except Exception as e:
            self.logger.error(f"Data sufficiency validation failed: {e}")
            return False
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        self.logger.info("Generating data quality report")
        
        try:
            report = {
                'basic_stats': {
                    'total_records': len(df),
                    'date_range': {
                        'start': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                        'end': df.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                        'days': (df.index.max() - df.index.min()).days
                    },
                    'columns': list(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                
                'data_quality': {
                    'missing_values': df.isnull().sum().to_dict(),
                    'zero_volume_periods': (df['volume'] == 0).sum(),
                    'duplicate_timestamps': df.index.duplicated().sum(),
                },
                
                'price_statistics': {},
                'validation_results': self.validation_results
            }
            
            # Calculate price statistics
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    report['price_statistics'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'median': float(df[col].median())
                    }
            
            # Volume statistics
            if 'volume' in df.columns:
                report['volume_statistics'] = {
                    'total': float(df['volume'].sum()),
                    'mean': float(df['volume'].mean()),
                    'std': float(df['volume'].std()),
                    'zero_periods': int((df['volume'] == 0).sum())
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate data quality report: {e}")
            return {}
    
    def validate_for_strategy(self, df: pd.DataFrame, strategy_name: str) -> Dict[str, bool]:
        """
        Validate data specifically for a trading strategy.
        
        Args:
            df: DataFrame to validate
            strategy_name: Name of the strategy
            
        Returns:
            Strategy-specific validation results
        """
        self.logger.info(f"Validating data for {strategy_name} strategy")
        
        # Base validation
        results = self.validate_ohlcv_data(df)
        
        # Strategy-specific validation
        if strategy_name.lower() == 'ema_crossover':
            results.update(self._validate_ema_data(df))
        elif strategy_name.lower() == 'momentum':
            results.update(self._validate_momentum_data(df))
        elif strategy_name.lower() == 'q_learning':
            results.update(self._validate_rl_data(df))
        
        return results
    
    def _validate_ema_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate data for EMA crossover strategy."""
        ema_config = Config.get_strategy_config('ema_crossover')
        fast_period = ema_config['fast_period']
        slow_period = ema_config['slow_period']
        
        # Need enough data for slow EMA calculation
        min_records_needed = slow_period * 3  # 3x the slow period for good EMA calculation
        sufficient_data = len(df) >= min_records_needed
        
        if not sufficient_data:
            self.logger.error(f"Insufficient data for EMA strategy: {len(df)} records (need {min_records_needed})")
        
        return {'ema_sufficient_data': sufficient_data}
    
    def _validate_momentum_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate data for momentum strategy."""
        momentum_config = Config.get_strategy_config('momentum')
        lookback_period = momentum_config.get('lookback_period', 20)
        
        # Need enough data for momentum calculation
        min_records_needed = lookback_period * 2
        sufficient_data = len(df) >= min_records_needed
        
        if not sufficient_data:
            self.logger.error(f"Insufficient data for momentum strategy: {len(df)} records (need {min_records_needed})")
        
        return {'momentum_sufficient_data': sufficient_data}
    
    def _validate_rl_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate data for reinforcement learning strategy."""
        # RL needs substantial data for training
        min_records_needed = 5000  # Minimum for RL training
        sufficient_data = len(df) >= min_records_needed
        
        if not sufficient_data:
            self.logger.error(f"Insufficient data for RL strategy: {len(df)} records (need {min_records_needed})")
        
        # Check for sufficient price movement
        price_volatility = df['close'].pct_change().std()
        sufficient_volatility = price_volatility > 0.001  # At least 0.1% daily volatility
        
        if not sufficient_volatility:
            self.logger.warning(f"Low price volatility may affect RL training: {price_volatility:.4f}")
        
        return {
            'rl_sufficient_data': sufficient_data,
            'rl_sufficient_volatility': sufficient_volatility
        } 