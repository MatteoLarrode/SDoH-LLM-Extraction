import pandas as pd
from typing import List, Dict

class MultiModelAnalyzer:
    """Analyze results across multiple models/prompts"""
    
    @staticmethod
    def combine_results(csv_files: List[str]) -> pd.DataFrame:
        """Combine multiple result CSVs into one DataFrame"""
        dfs = []
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    
    @staticmethod
    def compare_models(combined_df: pd.DataFrame) -> pd.DataFrame:
        """Compare model performance"""
        comparison = combined_df.groupby('model').agg({
            'sentence_id': 'count',  # total sentences
            'has_sdoh': ['sum', 'mean'],  # count and rate of SDoH detection
            'num_factors': 'mean',  # average factors per sentence
        }).round(3)
        
        # Flatten column names
        comparison.columns = ['total_sentences', 'sentences_with_sdoh', 'sdoh_detection_rate', 'avg_factors_per_sentence']
        
        return comparison.reset_index()
    
    @staticmethod
    def sentence_level_comparison(combined_df: pd.DataFrame, sentence_id: str) -> pd.DataFrame:
        """Compare how different models handled the same sentence"""
        sentence_data = combined_df[combined_df['sentence_id'] == sentence_id]
        
        if sentence_data.empty:
            print(f"Sentence ID {sentence_id} not found")
            return pd.DataFrame()
        
        # Show sentence text and results by model
        comparison = sentence_data[['sentence_id', 'sentence', 'model', 'sdoh_factors', 'has_sdoh', 'num_factors']].copy()
        
        return comparison
    
    @staticmethod
    def factor_analysis(combined_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze SDoH factors by model"""
        results = {}
        
        # Split factors and count frequency by model
        factor_data = []
        for _, row in combined_df[combined_df['has_sdoh']].iterrows():
            factors = row['sdoh_factors'].split(', ')
            for factor in factors:
                if factor.strip() and factor != "NoSDoH":
                    factor_data.append({
                        'model': row['model'],
                        'factor': factor.strip()
                    })
        
        if factor_data:
            factor_df = pd.DataFrame(factor_data)
            
            # Factor frequency by model
            results['factor_frequency'] = factor_df.groupby(['model', 'factor']).size().unstack(fill_value=0)
            
            # Most common factors overall
            results['overall_frequency'] = factor_df['factor'].value_counts().head(10)
        
        return results