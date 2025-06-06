import pandas as pd
import hashlib
from typing import List, Dict, Any
from pathlib import Path

class BatchProcessor:
    """Simplified batch processor that outputs clean DataFrames for analysis"""
    
    def __init__(self, output_dir: str = "batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_note_id(self, note_text: str, index: int) -> str:
        """Create deterministic note ID"""
        note_hash = hashlib.md5(note_text.encode('utf-8')).hexdigest()[:8]
        return f"note_{index:04d}_{note_hash}"
    
    def create_sentence_id(self, note_id: str, sentence_num: int) -> str:
        """Create sentence ID within note"""
        return f"{note_id}_s{sentence_num:02d}"
    
    def process_to_dataframe(self, 
                           df: pd.DataFrame, 
                           extractor, 
                           note_column: str,
                           model_name: str = None,
                           start_index: int = 0,
                           batch_size: int = None) -> pd.DataFrame:
        """
        Process notes directly to a clean DataFrame
        
        Args:
            df: DataFrame containing notes
            extractor: SDoHExtractor instance
            note_column: Column name with note text
            model_name: Optional model identifier
            start_index: Starting index
            batch_size: Number of notes to process (None = all)
        
        Returns:
            Clean DataFrame with columns: note_id, sentence_id, sentence, sdoh_factors, model
        """
        # Select subset if batch_size specified
        if batch_size is not None:
            end_index = min(start_index + batch_size, len(df))
            batch_df = df.iloc[start_index:end_index]
        else:
            batch_df = df.iloc[start_index:]
        
        # Get model name
        if model_name is None:
            model_name = getattr(extractor.tokenizer, 'name_or_path', 'unknown').split('/')[-1]
        
        # Add prompt info to model name for better identification
        full_model_id = f"{model_name}_{extractor.prompt_type}_L{extractor.level}"
        
        results = []
        
        for idx, row in batch_df.iterrows():
            note_text = row[note_column]
            if pd.isna(note_text) or not note_text.strip():
                continue
            
            note_id = self.create_note_id(note_text, idx)
            print(f"Processing note {idx} -> {note_id}")
            
            # Extract using your existing method
            extraction_result = extractor.extract_from_note(note_text)
            
            # Convert to flat structure
            for sentence_data in extraction_result["sentences"]:
                sentence_id = self.create_sentence_id(note_id, sentence_data["sentence_number"])
                
                # Get factors as comma-separated string (like your to_dataframe method)
                factors = sentence_data["sdoh_factors"]
                sdoh_factors = ", ".join(factors) if factors else "NoSDoH"
                
                results.append({
                    "note_id": note_id,
                    "sentence_id": sentence_id,
                    "sentence": sentence_data["sentence"],
                    "sdoh_factors": sdoh_factors,
                    "has_sdoh": factors != ["NoSDoH"] and bool(factors),
                    "num_factors": len(factors) if factors != ["NoSDoH"] else 0,
                    "model": full_model_id
                })
        
        return pd.DataFrame(results)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save DataFrame to CSV"""
        if filename is None:
            filename = f"batch_results_{df['model'].iloc[0].replace('/', '_')}.csv"
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        return str(filepath)


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

# Example usage workflow:
"""
# 1. Process multiple models
processor = BatchProcessor()

# Process with different models/prompts
df1 = processor.process_to_dataframe(notes_df, extractor1, 'note_text', 'llama-7b')
df2 = processor.process_to_dataframe(notes_df, extractor2, 'note_text', 'phi-3')

# Save individual results
processor.save_dataframe(df1, 'llama_results.csv')
processor.save_dataframe(df2, 'phi_results.csv')

# 2. Compare models
analyzer = MultiModelAnalyzer()
combined = analyzer.combine_results(['llama_results.csv', 'phi_results.csv'])

# Model comparison
comparison = analyzer.compare_models(combined)
print(comparison)

# Sentence-level comparison
sentence_comparison = analyzer.sentence_level_comparison(combined, 'note_0001_abc123_s01')
print(sentence_comparison)

# Factor analysis
factor_analysis = analyzer.factor_analysis(combined)
print(factor_analysis['overall_frequency'])
"""