import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalesPerformanceVisualization:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(random_state=42)
        
    def generate_sample_data(self, n_salespeople=100):
        """Generate realistic sales data for demonstration"""
        np.random.seed(42)
        
        # Generate sales person data
        names = [f"Sales_Person_{i+1}" for i in range(n_salespeople)]
        experience = np.random.randint(1, 15, n_salespeople)
        age = np.random.randint(22, 60, n_salespeople)
        region = np.random.choice(['North', 'South', 'East', 'West'], n_salespeople)
        
        # Generate sales metrics with some realistic correlations
        calls_per_day = np.random.poisson(25, n_salespeople) + experience
        meetings_per_week = np.random.poisson(8, n_salespeople) + (experience * 0.5).astype(int)
        conversion_rate = np.random.beta(2, 8, n_salespeople) + (experience * 0.01)
        conversion_rate = np.clip(conversion_rate, 0, 1)
        
        # Sales revenue with correlation to experience and conversion rate
        base_sales = 50000 + (experience * 3000) + (conversion_rate * 100000)
        sales_revenue = base_sales + np.random.normal(0, 15000, n_salespeople)
        sales_revenue = np.clip(sales_revenue, 20000, 200000)
        
        # Customer satisfaction
        customer_satisfaction = 3.0 + (conversion_rate * 2) + np.random.normal(0, 0.5, n_salespeople)
        customer_satisfaction = np.clip(customer_satisfaction, 1, 5)
        
        self.data = pd.DataFrame({
            'name': names,
            'experience_years': experience,
            'age': age,
            'region': region,
            'calls_per_day': calls_per_day,
            'meetings_per_week': meetings_per_week,
            'conversion_rate': conversion_rate,
            'sales_revenue': sales_revenue,
            'customer_satisfaction': customer_satisfaction
        })
        
        print(f"Generated sample data for {n_salespeople} sales people")
        return self.data
    
    def load_data(self, filepath=None):
        """Load data from file or generate sample data"""
        if filepath:
            self.data = pd.read_csv(filepath)
        else:
            self.generate_sample_data()
        return self.data
    
    def explore_data(self):
        """Basic data exploration"""
        print("=== SALES PERSON DATA OVERVIEW ===")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nData types:")
        print(self.data.dtypes)
        print("\nBasic statistics:")
        print(self.data.describe())
        print("\nMissing values:")
        print(self.data.isnull().sum())
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Sales Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sales Revenue Distribution
        axes[0,0].hist(self.data['sales_revenue'], bins=20, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Sales Revenue Distribution')
        axes[0,0].set_xlabel('Sales Revenue ($)')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Experience vs Sales Revenue
        axes[0,1].scatter(self.data['experience_years'], self.data['sales_revenue'], alpha=0.6)
        axes[0,1].set_title('Experience vs Sales Revenue')
        axes[0,1].set_xlabel('Years of Experience')
        axes[0,1].set_ylabel('Sales Revenue ($)')
        
        # 3. Conversion Rate vs Sales Revenue
        axes[0,2].scatter(self.data['conversion_rate'], self.data['sales_revenue'], alpha=0.6, color='green')
        axes[0,2].set_title('Conversion Rate vs Sales Revenue')
        axes[0,2].set_xlabel('Conversion Rate')
        axes[0,2].set_ylabel('Sales Revenue ($)')
        
        # 4. Sales by Region
        region_sales = self.data.groupby('region')['sales_revenue'].mean()
        axes[1,0].bar(region_sales.index, region_sales.values, color=['red', 'blue', 'green', 'orange'])
        axes[1,0].set_title('Average Sales by Region')
        axes[1,0].set_xlabel('Region')
        axes[1,0].set_ylabel('Average Sales Revenue ($)')
        
        # 5. Customer Satisfaction Distribution
        axes[1,1].hist(self.data['customer_satisfaction'], bins=15, alpha=0.7, color='purple')
        axes[1,1].set_title('Customer Satisfaction Distribution')
        axes[1,1].set_xlabel('Customer Satisfaction Score')
        axes[1,1].set_ylabel('Frequency')
        
        # 6. Calls per Day vs Meetings per Week
        axes[1,2].scatter(self.data['calls_per_day'], self.data['meetings_per_week'], alpha=0.6, color='orange')
        axes[1,2].set_title('Daily Calls vs Weekly Meetings')
        axes[1,2].set_xlabel('Calls per Day')
        axes[1,2].set_ylabel('Meetings per Week')
        
        # 7. Age Distribution by Region
        for i, region in enumerate(self.data['region'].unique()):
            region_data = self.data[self.data['region'] == region]['age']
            axes[2,0].hist(region_data, alpha=0.5, label=region, bins=10)
        axes[2,0].set_title('Age Distribution by Region')
        axes[2,0].set_xlabel('Age')
        axes[2,0].set_ylabel('Frequency')
        axes[2,0].legend()
        
        # 8. Performance Heatmap
        corr_columns = ['experience_years', 'calls_per_day', 'meetings_per_week', 
                       'conversion_rate', 'sales_revenue', 'customer_satisfaction']
        corr_matrix = self.data[corr_columns].corr()
        im = axes[2,1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[2,1].set_xticks(range(len(corr_columns)))
        axes[2,1].set_yticks(range(len(corr_columns)))
        axes[2,1].set_xticklabels([col.replace('_', '\n') for col in corr_columns], rotation=45)
        axes[2,1].set_yticklabels([col.replace('_', '\n') for col in corr_columns])
        axes[2,1].set_title('Performance Correlation Matrix')
        plt.colorbar(im, ax=axes[2,1])
        
        # 9. Top Performers
        top_performers = self.data.nlargest(10, 'sales_revenue')
        axes[2,2].barh(range(len(top_performers)), top_performers['sales_revenue'])
        axes[2,2].set_yticks(range(len(top_performers)))
        axes[2,2].set_yticklabels(top_performers['name'], fontsize=8)
        axes[2,2].set_title('Top 10 Performers by Revenue')
        axes[2,2].set_xlabel('Sales Revenue ($)')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_ml_data(self):
        """Prepare data for machine learning"""
        # Select features for prediction
        feature_columns = ['experience_years', 'age', 'calls_per_day', 
                          'meetings_per_week', 'conversion_rate', 'customer_satisfaction']
        
        # Create dummy variables for categorical data
        region_dummies = pd.get_dummies(self.data['region'], prefix='region')
        
        # Combine features
        X = pd.concat([self.data[feature_columns], region_dummies], axis=1)
        y = self.data['sales_revenue']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_model(self):
        """Train a machine learning model to predict sales revenue"""
        X_train, X_test, y_train, y_test, feature_names = self.prepare_ml_data()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("=== MODEL PERFORMANCE ===")
        print(f"Mean Squared Error: ${mse:,.2f}")
        print(f"R² Score: {r2:.3f}")
        print(f"Root Mean Squared Error: ${np.sqrt(mse):,.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== FEATURE IMPORTANCE ===")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title('Top 10 Most Important Features for Sales Prediction')
        plt.xlabel('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return y_test, y_pred
    
    def generate_insights(self):
        """Generate business insights from the data"""
        print("=== BUSINESS INSIGHTS ===")
        
        # Top performers analysis
        top_10_percent = int(len(self.data) * 0.1)
        top_performers = self.data.nlargest(top_10_percent, 'sales_revenue')
        
        print(f"\nTop 10% Performers Analysis:")
        print(f"Average Experience: {top_performers['experience_years'].mean():.1f} years")
        print(f"Average Conversion Rate: {top_performers['conversion_rate'].mean():.3f}")
        print(f"Average Customer Satisfaction: {top_performers['customer_satisfaction'].mean():.2f}")
        print(f"Most Common Region: {top_performers['region'].mode().iloc[0]}")
        
        # Regional analysis
        regional_performance = self.data.groupby('region').agg({
            'sales_revenue': ['mean', 'std'],
            'conversion_rate': 'mean',
            'customer_satisfaction': 'mean'
        }).round(2)
        
        print(f"\nRegional Performance:")
        print(regional_performance)
        
        # Correlation insights
        correlations = self.data[['experience_years', 'conversion_rate', 'sales_revenue', 
                                 'customer_satisfaction']].corr()['sales_revenue'].sort_values(ascending=False)
        print(f"\nFactors most correlated with Sales Revenue:")
        for factor, corr in correlations.items():
            if factor != 'sales_revenue':
                print(f"{factor}: {corr:.3f}")

def main():
    """Main execution function"""
    print("🚀 Sales Performance Visualization Project")
    print("=" * 50)
    
    # Initialize the visualization class
    analyzer = SalesPerformanceVisualization()
    
    # Load or generate data
    data = analyzer.load_data()  # This will generate sample data
    
    # Explore the data
    analyzer.explore_data()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    analyzer.create_visualizations()
    
    # Train machine learning model
    print("\n🤖 Training prediction model...")
    analyzer.train_model()
    
    # Generate business insights
    analyzer.generate_insights()
    
    print("\n✅ Analysis complete!")
    print("Check the generated plots and insights above.")

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
    print("Please run it directly to see the sales performance visualizations and insights.")