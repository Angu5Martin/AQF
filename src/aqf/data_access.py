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
        
        print(f"✅ FirstRateData client initialized")
        print(f"   Profile: {profile_name}")
        print(f"   Bucket: {self.bucket_name}")
        print(f"   Timezone: America/New_York")
    
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
                
            print(f"✅ Loaded stock metadata: {len(df)} tickers")
            return df
        except Exception as e:
            print(f"❌ Error loading stock metadata: {e}")
            return pd.DataFrame()
    
    def get_company_profiles(self) -> pd.DataFrame:
        """Load company profiles (ticker, company_name, country, sector, etc.)."""
        try:
            s3_path = f"s3://{self.bucket_name}/{self.base_path}/stock_company_profiles.parquet"
            df = pd.read_parquet(s3_path, filesystem=self.fs)
            
            # Convert IPO date
            if 'ipo_date' in df.columns:
                df['ipo_date'] = pd.to_datetime(df['ipo_date'])
                
            print(f"✅ Loaded company profiles: {len(df)} companies")
            return df
        except Exception as e:
            print(f"❌ Error loading company profiles: {e}")
            return pd.DataFrame()
    
    def get_sp500_changes(self) -> pd.DataFrame:
        """Load S&P 500 index changes (entry/exit data)."""
        try:
            s3_path = f"s3://{self.bucket_name}/{self.base_path}/stock_index_changes.parquet"
            df = pd.read_parquet(s3_path, filesystem=self.fs)
            print(f"✅ Loaded S&P 500 changes: {len(df)} changes")
            return df
        except Exception as e:
            print(f"❌ Error loading S&P 500 changes: {e}")
            return pd.DataFrame()
    
    def get_all_metadata(self) -> Dict[str, pd.DataFrame]:
        """Load all metadata files."""
        print("📋 Loading all metadata files...")
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
                print(f"❌ No data found for {date_obj}")
                return None
            
            # Ensure timestamp is in correct timezone
            if 'timestamp' in df.columns:
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(self.timezone)
                else:
                    df['timestamp'] = df['timestamp'].dt.tz_convert(self.timezone)
            
            print(f"✅ Loaded {len(df):,} records for {date_obj}")
            print(f"   Unique tickers: {df['ticker'].nunique():,}")
            print(f"   Time range: {df['timestamp'].min().strftime('%H:%M')} - {df['timestamp'].max().strftime('%H:%M')}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data for {trading_date}: {e}")
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
                print(f"⚠️  Date range too large ({(end_obj - start_obj).days} days). "
                      f"Maximum allowed: {max_days} days")
                return None
            
            print(f"📈 Loading data from {start_obj} to {end_obj}")
            
            all_dfs = []
            current_date = start_obj
            
            while current_date <= end_obj:
                df = self.load_day(current_date)
                if df is not None:
                    all_dfs.append(df)
                current_date += timedelta(days=1)
            
            if not all_dfs:
                print("❌ No data found for the specified date range")
                return None
            
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
            
            print(f"✅ Combined dataset: {len(combined_df):,} records")
            print(f"   Trading days: {len(all_dfs)}")
            print(f"   Unique tickers: {combined_df['ticker'].nunique():,}")
            
            return combined_df
            
        except Exception as e:
            print(f"❌ Error loading date range: {e}")
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
                print(f"✅ Found {len(ticker_data)} records for {ticker} on {trading_date}")
                return ticker_data
            else:
                print(f"❌ No data found for {ticker} on {trading_date}")
        return None
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for a dataset."""
        if df is None or len(df) == 0:
            return {}
        
        stats = {
            'total_records': len(df),
            'unique_tickers': df['ticker'].nunique(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'trading_hours': f"{df['timestamp'].dt.time.min()} - {df['timestamp'].dt.time.max()}"
            },
            'volume_stats': {
                'total_volume': df['volume'].sum(),
                'avg_volume': df['volume'].mean(),
                'median_volume': df['volume'].median()
            },
            'price_stats': {
                'avg_price': df['close'].mean(),
                'price_range': f"${df['close'].min():.2f} - ${df['close'].max():.2f}",
                'most_expensive': df.loc[df['close'].idxmax(), 'ticker'],
                'least_expensive': df.loc[df['close'].idxmin(), 'ticker']
            },
            'top_tickers_by_volume': df.groupby('ticker')['volume'].sum().sort_values(ascending=False).head(10)
        }
        
        return stats
    
    def print_summary(self, df: pd.DataFrame):
        """Print a formatted summary of the dataset."""
        if df is None or len(df) == 0:
            print("❌ No data to summarize")
            return
        
        stats = self.get_summary_stats(df)
        
        print("📊 DATASET SUMMARY")
        print("=" * 50)
        print(f"Records: {stats['total_records']:,}")
        print(f"Unique tickers: {stats['unique_tickers']:,}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"Trading hours: {stats['date_range']['trading_hours']}")
        print()
        print(f"Total volume: {stats['volume_stats']['total_volume']:,.0f}")
        print(f"Average volume per record: {stats['volume_stats']['avg_volume']:.2f}")
        print()
        print(f"Average price: ${stats['price_stats']['avg_price']:.2f}")
        print(f"Price range: {stats['price_stats']['price_range']}")
        print(f"Most expensive: {stats['price_stats']['most_expensive']}")
        print(f"Least expensive: {stats['price_stats']['least_expensive']}")
        print()
        print("Top 10 tickers by total volume:")
        for ticker, volume in stats['top_tickers_by_volume'].items():
            print(f"  {ticker}: {volume:,.0f}")


# Convenience functions for quick access
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
    print("🚀 FirstRateData Client Demo")
    print("=" * 40)
    
    # Initialize client
    client = FirstRateDataClient()
    
    # Show available dates
    dates = client.get_available_dates()
    print(f"\nAvailable data: {dates['date_range']}")
    print(f"Latest year: {dates['latest_year']}")
    print(f"Available months in {dates['latest_year']}: {dates['latest_year_months']}")
    
    # Load metadata
    print("\n📋 Loading metadata...")
    metadata = client.get_all_metadata()
    
    # Load sample day
    print("\n📈 Loading sample data (2024-08-01)...")
    sample_data = client.load_day('2024-08-01')
    
    if sample_data is not None:
        client.print_summary(sample_data)
