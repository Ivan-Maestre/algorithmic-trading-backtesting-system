"""
Binance Data Fetcher Module

Provides functionality to fetch historical cryptocurrency data from Binance API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
import time
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from ..config import Config


class BinanceDataFetcher:
    """
    Fetches historical cryptocurrency data from Binance API.
    
    This class handles authentication, rate limiting, and data retrieval
    for the backtesting system.
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize the Binance data fetcher.
        
        Args:
            api_key: Binance API key (optional, uses config if not provided)
            secret_key: Binance secret key (optional, uses config if not provided)
        """
        self.api_key = api_key or Config.BINANCE_CONFIG["api_key"]
        self.secret_key = secret_key or Config.BINANCE_CONFIG["secret_key"]
        
        # Initialize Binance client
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.secret_key,
            testnet=Config.BINANCE_CONFIG["testnet"]
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache for market info
        self._exchange_info = None
        
    def _rate_limit(self):
        """Implement rate limiting to respect Binance API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_exchange_info(self) -> Dict:
        """Get exchange information and cache it."""
        if self._exchange_info is None:
            try:
                self._rate_limit()
                self._exchange_info = self.client.get_exchange_info()
                self.logger.info("Successfully retrieved exchange information")
            except Exception as e:
                self.logger.error(f"Failed to get exchange info: {e}")
                raise
        
        return self._exchange_info
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists on Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            True if symbol exists, False otherwise
        """
        try:
            exchange_info = self._get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']]
            return symbol in symbols
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[List]:
        """
        Fetch historical kline/candlestick data from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '4h')
            start_date: Start date string (e.g., '2018-01-01')
            end_date: End date string (optional)
            limit: Number of klines to retrieve per request (max 1000)
            
        Returns:
            List of kline data
        """
        try:
            self._rate_limit()
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date,
                limit=limit
            )
            
            self.logger.info(f"Retrieved {len(klines)} klines for {symbol}")
            return klines
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error: {e}")
            raise
        except BinanceRequestException as e:
            self.logger.error(f"Binance request error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching klines: {e}")
            raise
    
    def fetch_ohlcv_data(
        self,
        symbol: str,
        timeframe: str = "4h",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data and convert to pandas DataFrame.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for candlesticks
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Use config defaults if not provided
        start_date = start_date or Config.DATA_START_DATE
        end_date = end_date or Config.DATA_END_DATE
        
        # Convert symbol to Binance format
        binance_symbol = Config.get_binance_symbol(symbol)
        binance_timeframe = Config.get_binance_timeframe(timeframe)
        
        self.logger.info(f"Fetching {binance_symbol} data from {start_date} to {end_date}")
        
        # Validate symbol
        if not self.validate_symbol(binance_symbol):
            raise ValueError(f"Invalid symbol: {binance_symbol}")
        
        try:
            # Fetch all historical data
            all_klines = []
            current_start = start_date
            
            while True:
                klines = self.get_historical_klines(
                    symbol=binance_symbol,
                    interval=binance_timeframe,
                    start_date=current_start,
                    end_date=end_date,
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Check if we've reached the end
                last_timestamp = klines[-1][0]
                last_date = datetime.fromtimestamp(last_timestamp / 1000, timezone.utc)
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                
                if last_date >= end_datetime:
                    break
                
                # Update start date for next batch
                current_start = last_date.strftime("%Y-%m-%d")
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(all_klines)
            
            # Filter by date range
            df = self._filter_by_date_range(df, start_date, end_date)
            
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            raise
    
    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        Convert Binance klines data to pandas DataFrame.
        
        Args:
            klines: Raw klines data from Binance API
            
        Returns:
            DataFrame with proper column names and data types
        """
        if not klines:
            return pd.DataFrame()
        
        # Define column names
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price and volume columns to float
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'quote_asset_volume', 'taker_buy_base_asset_volume', 
                           'taker_buy_quote_asset_volume']
        
        for col in price_volume_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert number of trades to int
        df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce', downcast='integer')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Keep only essential columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'number_of_trades']]
        
        # Remove any duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: Input DataFrame with timestamp index
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Filtered DataFrame
        """
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include end date
        
        mask = (df.index >= start_datetime) & (df.index < end_datetime)
        return df.loc[mask]
    
    def test_connection(self) -> bool:
        """
        Test connection to Binance API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._rate_limit()
            server_time = self.client.get_server_time()
            self.logger.info(f"Binance connection successful. Server time: {server_time}")
            return True
        except Exception as e:
            self.logger.error(f"Binance connection failed: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        try:
            exchange_info = self._get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'] 
                      if s['status'] == 'TRADING']
            return sorted(symbols)
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a trading symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol information dictionary
        """
        try:
            exchange_info = self._get_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    return s
            return {}
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
    
    def save_data_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Path to saved file
        """
        filename = f"{symbol}_{timeframe}_{Config.DATA_START_DATE}_{Config.DATA_END_DATE}.csv"
        filepath = Config.RAW_DATA_DIR + "/" + filename
        
        df.to_csv(filepath)
        self.logger.info(f"Data saved to {filepath}")
        
        return filepath 