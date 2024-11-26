import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Optional imports with fallback
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not installed. Some visualizations may be limited.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("SciPy not installed. Some statistical operations may be limited.")

class Inefficient_DataProcessor:
    def __init__(self, num_records=10000):
        """
        Intentionally create an inefficient data processing class
        with multiple redundant and computationally expensive operations
        """
        self.generate_messy_data(num_records)
    
    def generate_messy_data(self, num_records):
        """
        Generate a large, intentionally unoptimized dataset
        """
        np.random.seed(42)
        
        # Deliberately create a huge, inefficient DataFrame
        self.df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(num_records)],
            'age': np.random.randint(18, 80, num_records),
            'income': np.random.normal(50000, 15000, num_records),
            'spending_score': np.random.uniform(0, 100, num_records),
            'purchase_category': [random.choice(['electronics', 'fashion', 'food', 'travel', 'services']) for _ in range(num_records)]
        })
    
    def unnecessarily_complex_filtering(self):
        """
        Perform multiple redundant filtering operations
        """
        start_time = time.time()
        
        # Ridiculously inefficient filtering with multiple redundant steps
        result = self.df.copy()
        for _ in range(5):  # Repeat filtering multiple times
            result = result[result['age'] > 30]
            result = result[result['income'] > result['income'].mean()]
            result = result[result['spending_score'] > 50]
        
        end_time = time.time()
        print(f"Complex Filtering Time: {end_time - start_time} seconds")
        return result
    
    def create_nested_aggregations(self):
        """
        Perform nested, computationally expensive aggregations
        """
        start_time = time.time()
        
        # Multiple nested and redundant aggregation operations
        result = (
            self.df.groupby('purchase_category', group_keys=False)
            .apply(lambda x: x.groupby('age')['income'].agg(['mean', 'sum', 'max']))
            .apply(lambda x: x.apply(lambda y: y * random.uniform(0.9, 1.1)))
        )
        
        end_time = time.time()
        print(f"Nested Aggregation Time: {end_time - start_time} seconds")
        return result
    
    def visualize_inefficiently(self):
        """
        Create multiple, overlapping visualizations
        """
        plt.figure(figsize=(15, 10))
        
        # Multiple inefficient visualization techniques
        plt.subplot(2, 2, 1)
        if HAS_SEABORN:
            sns.scatterplot(data=self.df, x='age', y='income', hue='purchase_category', alpha=0.5)
        else:
            self.df.plot(kind='scatter', x='age', y='income', alpha=0.5, figsize=(10, 5))
        plt.title('Scattered and Messy Income vs Age')
        
        plt.subplot(2, 2, 2)
        if HAS_SCIPY:
            # Manual KDE plot as a fallback
            for category in self.df['purchase_category'].unique():
                subset = self.df[self.df['purchase_category'] == category]['spending_score']
                kde = stats.gaussian_kde(subset)
                x_range = np.linspace(subset.min(), subset.max(), 100)
                plt.plot(x_range, kde(x_range), label=category)
            plt.title('Kernel Density of Spending Scores')
            plt.legend()
        else:
            self.df.boxplot(column='spending_score', by='purchase_category')
            plt.title('Spending Score Distribution')
        
        plt.subplot(2, 2, 3)
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='purchase_category', y='income')
        else:
            self.df.boxplot(column='income', by='purchase_category')
        plt.title('Income Distribution by Category')
        
        plt.subplot(2, 2, 4)
        category_counts = self.df['purchase_category'].value_counts()
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
        plt.title('Category Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def memory_intensive_operation(self):
        """
        Perform a deliberately memory-intensive and slow operation
        """
        start_time = time.time()
        
        # Create multiple copies and perform unnecessary operations
        temp_df = self.df.copy()
        for _ in range(10):
            temp_df = pd.concat([temp_df, temp_df], ignore_index=True)
            temp_df['random_column'] = np.random.random(len(temp_df))
        
        end_time = time.time()
        print(f"Memory Intensive Operation Time: {end_time - start_time} seconds")
        return temp_df

def main():
    processor = Inefficient_DataProcessor()
    
    # Demonstrate various inefficient operations
    filtered_data = processor.unnecessarily_complex_filtering()
    nested_aggs = processor.create_nested_aggregations()
    processor.visualize_inefficiently()
    memory_intensive_result = processor.memory_intensive_operation()

if __name__ == "__main__":
    main()