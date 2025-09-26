"""
Usage:
    from src.aqf.data_access import FirstRateDataClient
    
    client = FirstRateDataClient()
    
    # Load metadata
    metadata = client.get_all_metadata()
    
    # Load single day
    data = client.load_day('2024-08-01')
    
    # Load date range
    data = client.load_date_range('2024-08-01', '2024-08-05')
"""

import boto3
import pandas as pd
import s3fs
import pytz
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

class FirstRateDataClient:
    """Client for accessing FirstRateData from S3 bucket."""
    
    def __init__(self, profile_name: str = "firstratedata"):
        """
        Initialize the FirstRateData client.
        
        Args:
            profile_name: AWS profile name (default: firstratedata)
        """
        self.profile_name = profile_name
        self.endpoint_url = "https://nbg1.your-objectstorage.com"
        self.bucket_name = "aqf"
        self.base_path = "firstratedata"
        
        # Timezone for the data
        self.timezone = pytz.timezone('America/New_York')
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=profile_name)
        self.s3_client = self.session.client('s3', endpoint_url=self.endpoint_url)
        self.fs = s3fs.S3FileSystem(
            profile=profile_name,
            client_kwargs={'endpoint_url': self.endpoint_url}
        )
        
        print(f"FirstRateData client initialized")
    
    def get_available_dates(self) -> Dict[str, List]:
        """Get all available years, and sample months for latest year."""
        try:
            # Get years
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.base_path}/y=",
                Delimiter="/"
            )
            
            years = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    folder_name = prefix['Prefix'].split('/')[-2]
                    if folder_name.startswith('y='):
                        year = int(folder_name.replace('y=', ''))
                        years.append(year)
            
            years = sorted(years)
            
            # Get months for latest year
            latest_year = max(years) if years else None
            months = []
            if latest_year:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"{self.base_path}/y={latest_year}/m=",
                    Delimiter="/"
                )
                
                if 'CommonPrefixes' in response:
                    for prefix in response['CommonPrefixes']:
                        folder_name = prefix['Prefix'].split('/')[-2]
                        if folder_name.startswith('m='):
                            month = int(folder_name.replace('m=', ''))
                            months.append(month)
            
            return {
                'years': years,
                'latest_year': latest_year,
                'latest_year_months': sorted(months),
                'date_range': f"{min(years)}-{max(years)}" if years else "No data"
            }
            
        except Exception as e:
            print(f"Error getting available dates: {e}")
            return {}
    
    def get_metadata_stock(self) -> pd.DataFrame:
        """Load stock metadata (ticker, name, first_date, last_date)."""
        try:
            s3_path = f"s3://{self.bucket_name}/{self.base_path}/metadata_stock.parquet"
            df = pd.read_parquet(s3_path, filesystem=self.fs)
            
            # Convert dates to datetime
            if 'first_date' in df.columns:
                df['first_date'] = pd.to_datetime(df['first_date'])
            if 'last_date' in df.columns:
                df['last_date'] = pd.to_datetime(df['last_date'])
                
            print(f"Loaded stock metadata: {len(df)} tickers")
            return df
        except Exception as e:
            print(f"Error loading stock metadata: {e}")
            return pd.DataFrame()
    
    def get_company_profiles(self) -> pd.DataFrame:
        """Load company profiles (ticker, company_name, country, sector, etc.)."""
        try:
            s3_path = f"s3://{self.bucket_name}/{self.base_path}/stock_company_profiles.parquet"
            df = pd.read_parquet(s3_path, filesystem=self.fs)
            
            # Convert IPO date
            if 'ipo_date' in df.columns:
                df['ipo_date'] = pd.to_datetime(df['ipo_date'])
                
            print(f"Loaded company profiles: {len(df)} companies")
            return df
        except Exception as e:
            print(f"Error loading company profiles: {e}")
            return pd.DataFrame()
    
    def get_sp500_changes(self) -> pd.DataFrame:
        """Load S&P 500 index changes (entry/exit data)."""
        try:
            s3_path = f"s3://{self.bucket_name}/{self.base_path}/stock_index_changes.parquet"
            df = pd.read_parquet(s3_path, filesystem=self.fs)
            print(f"Loaded S&P 500 changes: {len(df)} changes")
            return df
        except Exception as e:
            print(f"Error loading S&P 500 changes: {e}")
            return pd.DataFrame()
    
    def get_all_metadata(self) -> Dict[str, pd.DataFrame]:
        """Load all metadata files."""
        print("Loading all metadata files...")
        return {
            'stock_metadata': self.get_metadata_stock(),
            'company_profiles': self.get_company_profiles(),
            'sp500_changes': self.get_sp500_changes()
        }
    
    def _parse_date(self, date_str: Union[str, date, datetime]) -> date:
        """Parse various date formats to date object."""
        if isinstance(date_str, str):
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        elif isinstance(date_str, datetime):
            return date_str.date()
        elif isinstance(date_str, date):
            return date_str
        else:
            raise ValueError(f"Invalid date format: {date_str}")
    
    def load_day(self, trading_date: Union[str, date, datetime]) -> Optional[pd.DataFrame]:
        """
        Load all stock data for a specific trading day.
        
        Args:
            trading_date: Date in format 'YYYY-MM-DD' or date/datetime object
            
        Returns:
            DataFrame with columns: timestamp, ticker, open, high, low, close, volume
        """
        try:
            date_obj = self._parse_date(trading_date)
            
            # Construct S3 path
            s3_path = (f"s3://{self.bucket_name}/{self.base_path}/"
                      f"y={date_obj.year}/m={date_obj.month:02d}/d={date_obj.day:02d}/"
                      f"stock_1min.parquet")
            
            # Check if file exists
            try:
                df = pd.read_parquet(s3_path, filesystem=self.fs)
            except Exception:
                print(f"No data found for {date_obj}")
                return None
            
            # Ensure timestamp is in correct timezone
            if 'timestamp' in df.columns:
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(self.timezone)
                else:
                    df['timestamp'] = df['timestamp'].dt.tz_convert(self.timezone)
            
            print(f"Loaded {len(df):,} records for {date_obj}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {trading_date}: {e}")
            return None
    
    def load_date_range(self, start_date: Union[str, date, datetime], 
                       end_date: Union[str, date, datetime],
                       max_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Load stock data for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_days: Maximum number of days to load (safety limit)
            
        Returns:
            Combined DataFrame for the date range
        """
        try:
            start_obj = self._parse_date(start_date)
            end_obj = self._parse_date(end_date)
            
            if (end_obj - start_obj).days > max_days:
                print(f"Date range too large ({(end_obj - start_obj).days} days). "
                      f"Maximum allowed: {max_days} days")
                return None
            
            print(f"Loading data from {start_obj} to {end_obj}")
            
            all_dfs = []
            current_date = start_obj
            
            while current_date <= end_obj:
                df = self.load_day(current_date)
                if df is not None:
                    all_dfs.append(df)
                current_date += timedelta(days=1)
            
            if not all_dfs:
                print("No data found for the specified date range")
                return None
            
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
            
            return combined_df
            
        except Exception as e:
            print(f"Error loading date range: {e}")
            return None
    
    def get_ticker_data(self, ticker: str, trading_date: Union[str, date, datetime]) -> Optional[pd.DataFrame]:
        """
        Get data for a specific ticker on a specific day.
        
        Args:
            ticker: Stock ticker symbol
            trading_date: Trading date
            
        Returns:
            DataFrame filtered for the specific ticker
        """
        df = self.load_day(trading_date)
        if df is not None:
            ticker_data = df[df['ticker'] == ticker].copy()
            if len(ticker_data) > 0:
                print(f"Found {len(ticker_data)} records for {ticker} on {trading_date}")
                return ticker_data
            else:
                print(f"No data found for {ticker} on {trading_date}")
        return None
    
    # ---------------------------
    # Important for NN training
    # ---------------------------
    
    def load_multi_ticker_data(self, tickers: List[str], 
                            start_date: Optional[Union[str, date, datetime]] = None,
                            end_date: Optional[Union[str, date, datetime]] = None,
                            days_back: Optional[int] = None, 
                            frequency: str = '5T',
                            add_features: bool = True,
                            fill_method: str = 'forward') -> pd.DataFrame:
        """
        Load data for multiple tickers over a date range with custom sampling frequency.
        Designed for neural network input preparation.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (if provided, overrides days_back)
            end_date: End date (if None, uses most recent available data)
            days_back: Number of trading days to go back from end_date (ignored if start_date provided)
            frequency: Pandas frequency string ('1T', '5T', '15T', '1H', etc.) - T for minutes
            add_features: Whether to add engineered features (returns, moving averages, etc.)
            fill_method: How to handle missing data ('forward', 'backward', 'drop', 'zero')
            
        Returns:
            Combined DataFrame with all tickers and timestamps
        """
        try:            
            # Determine date range
            if start_date is not None and end_date is not None:
                start_date = self._parse_date(start_date)
                end_date = self._parse_date(end_date)
            elif days_back is not None:
                # Find the most recent available date and go back
                if end_date is None:
                    dates_info = self.get_available_dates()
                    if not dates_info.get('years'):
                        print("No data available")
                        return pd.DataFrame()
                    
                    latest_year = dates_info['latest_year']
                    latest_months = dates_info['latest_year_months']
                    
                    if not latest_months:
                        print("No months available for latest year")
                        return pd.DataFrame()
                    
                    end_date = date(latest_year, max(latest_months), 28)
                else:
                    end_date = self._parse_date(end_date)
                
                # Calculate start date (going back by days_back trading days)
                calendar_days_back = int(days_back * 1.4)  # Account for weekends
                start_date = end_date - timedelta(days=calendar_days_back)
            else:
                print("Must provide either (start_date, end_date) or days_back parameter")
                return pd.DataFrame()
            
            # Load raw data for the date range
            max_days = (end_date - start_date).days + 5
            raw_data = self.load_date_range(start_date, end_date, max_days=max_days)
            
            if raw_data is None or raw_data.empty:
                print("No data loaded for date range")
                return pd.DataFrame()
            
            # Filter for requested tickers
            available_tickers = set(raw_data['ticker'].unique())
            missing_tickers = set(tickers) - available_tickers
            found_tickers = list(set(tickers) & available_tickers)
            
            if missing_tickers:
                print(f"Missing tickers (no data): {list(missing_tickers)}")
            if found_tickers:
                print(f"Found tickers: {found_tickers}")
            
            if not found_tickers:
                print("None of the requested tickers found")
                return pd.DataFrame()
            
            # Filter for found tickers only
            filtered_data = raw_data[raw_data['ticker'].isin(found_tickers)].copy()
            
            # Create a complete time index for trading days/hours only (excludes weekends)
            complete_time_index = self._create_trading_time_index(
                start_date, end_date, frequency, include_extended_hours=False
            )
            
            print(f"Creating trading-only time index with {len(complete_time_index)} timestamps (weekends excluded)")
            
            # Process each ticker and combine results
            all_ticker_data = []
            
            for ticker in found_tickers:
                print(f"   Processing {ticker}...")
                
                # Filter data for this ticker
                ticker_df = filtered_data[filtered_data['ticker'] == ticker].copy()
                
                if ticker_df.empty:
                    print(f"      Warning: No data for {ticker}")
                    continue
                    
                ticker_df = ticker_df.sort_values('timestamp').set_index('timestamp')
                
                # Resample to requested frequency
                resampled_df = ticker_df.resample(frequency).agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                
                # Reindex to complete time index to ensure all timestamps are present
                resampled_df = resampled_df.reindex(complete_time_index)
                
                # Handle missing data based on fill_method
                original_nans = resampled_df.isnull().sum().sum()
                
                if fill_method == 'forward':
                    # First try forward fill
                    resampled_df = resampled_df.fillna(method='ffill')
                    # If there are still NaNs at the beginning, use backward fill for those
                    remaining_nans = resampled_df.isnull().sum().sum()
                    if remaining_nans > 0:
                        resampled_df = resampled_df.fillna(method='bfill')
                        
                elif fill_method == 'backward':
                    # First try backward fill
                    resampled_df = resampled_df.fillna(method='bfill')
                    # If there are still NaNs at the end, use forward fill for those
                    remaining_nans = resampled_df.isnull().sum().sum()
                    if remaining_nans > 0:
                        resampled_df = resampled_df.fillna(method='ffill')
                        
                elif fill_method == 'zero':
                    resampled_df = resampled_df.fillna(0)
                    
                elif fill_method == 'drop':
                    resampled_df = resampled_df.dropna()
                
                final_nans = resampled_df.isnull().sum().sum()
                
                # Add ticker column
                resampled_df['ticker'] = ticker
                
                # Add useful features for neural networks if requested
                if add_features:
                    resampled_df['returns'] = resampled_df['close'].pct_change()
                    resampled_df['log_returns'] = np.log(resampled_df['close'] / resampled_df['close'].shift(1))
                    resampled_df['high_low_pct'] = (resampled_df['high'] - resampled_df['low']) / resampled_df['close']
                    resampled_df['volume_ma'] = resampled_df['volume'].rolling(window=20, min_periods=1).mean()
                    resampled_df['price_ma'] = resampled_df['close'].rolling(window=20, min_periods=1).mean()
                
                # Reset index to make timestamp a column
                resampled_df = resampled_df.reset_index()
                resampled_df.rename(columns={'index': 'timestamp'}, inplace=True)
                
                all_ticker_data.append(resampled_df)
                print(f"      {ticker}: {len(resampled_df)} samples (filled {original_nans - final_nans} NaNs)")
            
            # Combine all ticker data
            if all_ticker_data:
                combined_df = pd.concat(all_ticker_data, ignore_index=True)
                combined_df = combined_df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
                
                # Final summary
                total_nans = combined_df.isnull().sum().sum()
                print(f"\nFinal dataset: {len(combined_df):,} total records")
                print(f"   Tickers: {len(found_tickers)} ({', '.join(found_tickers)})")
                print(f"   Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
                print(f"   Remaining NaN values: {total_nans}")
                
                return combined_df
            else:
                print("No data processed")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error loading multi-ticker data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_aligned_dataset(self, multi_ticker_data: pd.DataFrame, 
                              value_column: str = 'close',
                              fill_method: str = 'forward') -> Optional[pd.DataFrame]:
        """
        Create a time-aligned dataset with all tickers as columns.
        
        Args:
            multi_ticker_data: DataFrame from load_multi_ticker_data()
            value_column: Column to use as the main value (e.g., 'close', 'returns', 'log_returns')
            fill_method: How to handle missing data ('forward', 'backward', 'drop', 'zero')
            
        Returns:
            DataFrame with timestamp index and tickers as columns
        """
        try:
            if multi_ticker_data.empty:
                print("Empty multi-ticker data provided")
                return None
            
            print(f"Creating aligned dataset from {value_column} column...")
            
            # Check if the required column exists
            if value_column not in multi_ticker_data.columns:
                print(f"Column '{value_column}' not found in data")
                print(f"Available columns: {list(multi_ticker_data.columns)}")
                return None
            
            # Pivot the data to have tickers as columns
            aligned_df = multi_ticker_data.pivot(
                index='timestamp', 
                columns='ticker', 
                values=value_column
            )
            
            # Handle missing data based on fill_method
            if fill_method == 'forward':
                aligned_df = aligned_df.fillna(method='ffill')
            elif fill_method == 'backward':
                aligned_df = aligned_df.fillna(method='bfill')
            elif fill_method == 'zero':
                aligned_df = aligned_df.fillna(0)
            elif fill_method == 'drop':
                aligned_df = aligned_df.dropna()
            
            # Check for missing values
            missing_counts = aligned_df.isnull().sum()
            if missing_counts.sum() > 0:
                print(f"   Missing values per ticker:")
                for ticker, count in missing_counts.items():
                    if count > 0:
                        pct = count / len(aligned_df) * 100
                        print(f"      {ticker}: {count} ({pct:.1f}%)")
            else:
                print(f"   No missing values")
            
            return aligned_df
            
        except Exception as e:
            print(f"Error creating aligned dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_trading_days_only(self, start_date: date, end_date: date) -> List[date]:
        """Get only trading days (exclude weekends and holidays)."""
        import pandas as pd
        
        # Create business day range (excludes weekends)
        business_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Convert to list of date objects
        trading_days = [day.date() for day in business_days]
        
        return trading_days
    
    def _create_trading_time_index(self, start_date: date, end_date: date, frequency: str, 
                                  include_extended_hours: bool = False):
        """Create time index only for trading hours on trading days."""
        
        # Get only trading days (excludes weekends)
        trading_days = self._get_trading_days_only(start_date, end_date)
        
        # Define market hours
        if include_extended_hours:
            # Pre-market: 4:00 AM, Regular: 9:30 AM - 4:00 PM, After: 8:00 PM
            session_start = pd.Timedelta(hours=4, minutes=0)
            session_end = pd.Timedelta(hours=20, minutes=0)
        else:
            # Regular market hours only: 9:30 AM to 4:00 PM Eastern
            session_start = pd.Timedelta(hours=9, minutes=30)
            session_end = pd.Timedelta(hours=16, minutes=0)
        
        all_timestamps = []
        
        for trading_day in trading_days:
            # Create timestamps for this trading day
            day_start = pd.Timestamp(trading_day).tz_localize(self.timezone) + session_start
            day_end = pd.Timestamp(trading_day).tz_localize(self.timezone) + session_end
            
            # Create intraday timestamps at specified frequency
            day_timestamps = pd.date_range(start=day_start, end=day_end, freq=frequency)
            all_timestamps.extend(day_timestamps)
        
        return pd.DatetimeIndex(all_timestamps)

# ---------------------------------------
# Convenience functions for quick access
# ---------------------------------------

def quick_load_day(trading_date: str) -> Optional[pd.DataFrame]:
    """Quick function to load a single day's data."""
    client = FirstRateDataClient()
    return client.load_day(trading_date)

def quick_load_range(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Quick function to load a date range."""
    client = FirstRateDataClient()
    return client.load_date_range(start_date, end_date)

def get_metadata() -> Dict[str, pd.DataFrame]:
    """Quick function to get all metadata."""
    client = FirstRateDataClient()
    return client.get_all_metadata()


if __name__ == "__main__":
    # Demo usage
    print("FirstRateData Client Demo")
    print("=" * 40)
    
    # Initialize client
    client = FirstRateDataClient()
    
    # Show available dates
    dates = client.get_available_dates()
    print(f"\nAvailable data: {dates['date_range']}")
    print(f"Latest year: {dates['latest_year']}")
    print(f"Available months in {dates['latest_year']}: {dates['latest_year_months']}")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = client.get_all_metadata()
    
    # Load sample day
    print("\nLoading sample data (2024-08-01)...")
    sample_data = client.load_day('2024-08-01')
    
    if sample_data is not None:
        client.print_summary(sample_data)
