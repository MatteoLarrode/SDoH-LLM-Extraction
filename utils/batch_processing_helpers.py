import pandas as pd
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

class BatchProcessor:
    """Handles batch processing of referral notes for SDoH extraction"""
    
    def __init__(self, output_dir: str = "batch_results"):
        """
        Initialize batch processor
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_note_id(self, note_text: str, index: int) -> str:
        """
        Create a unique, deterministic ID for a note
        
        Args:
            note_text: The text content of the note
            index: Index of the note in the dataset
            
        Returns:
            Unique note ID
        """
        # Create hash of note content for consistency across runs
        note_hash = hashlib.md5(note_text.encode('utf-8')).hexdigest()[:8]
        return f"note_{index:04d}_{note_hash}"
    
    def create_sentence_id(self, note_id: str, sentence_num: int, sentence_text: str) -> str:
        """
        Create a unique ID for a sentence within a note
        
        Args:
            note_id: ID of the parent note
            sentence_num: Sentence number within the note
            sentence_text: The sentence text
            
        Returns:
            Unique sentence ID
        """
        sentence_hash = hashlib.md5(sentence_text.encode('utf-8')).hexdigest()[:6]
        return f"{note_id}_s{sentence_num:02d}_{sentence_hash}"
    
    def process_batch(self, 
                     df: pd.DataFrame, 
                     extractor, 
                     note_column: str,
                     batch_size: int = 10,
                     start_index: int = 0) -> Dict[str, Any]:
        """
        Process a batch of notes
        
        Args:
            df: DataFrame containing the notes
            extractor: SDoHExtractor instance
            note_column: Column name containing the notes
            batch_size: Number of notes to process
            start_index: Starting index in the DataFrame
            
        Returns:
            Batch processing results
        """
        # Select batch
        end_index = min(start_index + batch_size, len(df))
        batch_df = df.iloc[start_index:end_index].copy()
        
        # Initialize results
        batch_results = {
            "metadata": {
                "batch_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "model_name": getattr(extractor.tokenizer, 'name_or_path', 'unknown'),
                "prompt_type": extractor.prompt_type,
                "level": extractor.level,
                "debug_enabled": extractor.debug,
                "batch_size": len(batch_df),
                "start_index": start_index,
                "end_index": end_index - 1,
                "total_processing_time": 0
            },
            "notes": [],
            "summary_stats": {}
        }
        
        start_time = time.time()
        
        # Process each note
        for idx, row in batch_df.iterrows():
            note_text = row[note_column]
            if pd.isna(note_text) or not note_text.strip():
                continue
                
            note_id = self.create_note_id(note_text, idx)
            
            print(f"Processing note {idx} (ID: {note_id})...")
            
            # Extract SDoH from note
            extraction_result = extractor.extract_from_note(note_text)
            
            # Enhanced note result with IDs
            note_result = {
                "note_id": note_id,
                "original_index": idx,
                "note_text": note_text,
                "extraction_result": extraction_result,
                "sentences_with_ids": []
            }
            
            # Add sentence IDs
            for sentence_data in extraction_result["sentences"]:
                sentence_id = self.create_sentence_id(
                    note_id, 
                    sentence_data["sentence_number"], 
                    sentence_data["sentence"]
                )
                
                sentence_with_id = sentence_data.copy()
                sentence_with_id["sentence_id"] = sentence_id
                sentence_with_id["note_id"] = note_id
                
                note_result["sentences_with_ids"].append(sentence_with_id)
            
            batch_results["notes"].append(note_result)
        
        # Calculate processing time
        end_time = time.time()
        batch_results["metadata"]["total_processing_time"] = end_time - start_time
        
        # Generate summary statistics
        batch_results["summary_stats"] = self._generate_batch_stats(batch_results["notes"])
        
        return batch_results
    
    def _generate_batch_stats(self, notes: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the batch"""
        total_notes = len(notes)
        total_sentences = sum(len(note["sentences_with_ids"]) for note in notes)
        
        # Count SDoH factors
        all_factors = []
        sentences_with_sdoh = 0
        
        for note in notes:
            for sentence in note["sentences_with_ids"]:
                factors = sentence["sdoh_factors"]
                if factors != ["NoSDoH"]:
                    all_factors.extend(factors)
                    sentences_with_sdoh += 1
        
        # Factor frequency
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        return {
            "total_notes": total_notes,
            "total_sentences": total_sentences,
            "sentences_with_sdoh": sentences_with_sdoh,
            "sentences_without_sdoh": total_sentences - sentences_with_sdoh,
            "unique_factors_found": list(set(all_factors)),
            "factor_frequencies": factor_counts,
            "total_factor_mentions": len(all_factors)
        }
    
    def save_results(self, batch_results: Dict[str, Any], filename_prefix: str = None) -> str:
        """
        Save batch results to JSON file
        
        Args:
            batch_results: Results from process_batch
            filename_prefix: Optional prefix for filename
            
        Returns:
            Path to saved file
        """
        metadata = batch_results["metadata"]
        
        if filename_prefix is None:
            filename_prefix = "batch_results"
        
        filename = (
            f"{filename_prefix}_"
            f"{metadata['model_name'].split('/')[-1]}_"
            f"{metadata['prompt_type']}_"
            f"level{metadata['level']}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filepath}")
        return str(filepath)
    
    def create_comparison_dataset(self, results_files: List[str]) -> pd.DataFrame:
        """
        Create a comparison dataset from multiple result files
        
        Args:
            results_files: List of paths to result JSON files
            
        Returns:
            DataFrame for comparison analysis
        """
        comparison_data = []
        
        for file_path in results_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            metadata = results["metadata"]
            
            for note in results["notes"]:
                for sentence in note["sentences_with_ids"]:
                    comparison_data.append({
                        "sentence_id": sentence["sentence_id"],
                        "note_id": sentence["note_id"],
                        "sentence_text": sentence["sentence"],
                        "model_name": metadata["model_name"],
                        "prompt_type": metadata["prompt_type"],
                        "level": metadata["level"],
                        "sdoh_factors": sentence["sdoh_factors"],
                        "factors_count": len(sentence["sdoh_factors"]) if sentence["sdoh_factors"] != ["NoSDoH"] else 0,
                        "has_sdoh": sentence["sdoh_factors"] != ["NoSDoH"],
                        "batch_id": metadata["batch_id"],
                        "timestamp": metadata["timestamp"]
                    })
        
        return pd.DataFrame(comparison_data)


class ResultsAnalyzer:
    """Analyze and compare batch processing results"""
    
    @staticmethod
    def compare_models(comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare performance across different models/prompts
        
        Args:
            comparison_df: DataFrame from create_comparison_dataset
            
        Returns:
            Comparison analysis
        """
        analysis = {}
        
        # Group by model and prompt type
        for (model, prompt_type, level), group in comparison_df.groupby(['model_name', 'prompt_type', 'level']):
            key = f"{model}_{prompt_type}_level{level}"
            
            analysis[key] = {
                "total_sentences": len(group),
                "sentences_with_sdoh": group['has_sdoh'].sum(),
                "sdoh_detection_rate": group['has_sdoh'].mean(),
                "avg_factors_per_sentence": group['factors_count'].mean(),
                "unique_factors": len(set([f for factors in group['sdoh_factors'] for f in factors if f != "NoSDoH"])),
                "most_common_factors": group[group['has_sdoh']]['sdoh_factors'].explode().value_counts().head(10).to_dict()
            }
        
        return analysis
    
    @staticmethod
    def sentence_level_comparison(comparison_df: pd.DataFrame, sentence_id: str) -> Dict[str, Any]:
        """
        Compare how different models/prompts handled the same sentence
        
        Args:
            comparison_df: DataFrame from create_comparison_dataset
            sentence_id: ID of sentence to compare
            
        Returns:
            Sentence-level comparison
        """
        sentence_data = comparison_df[comparison_df['sentence_id'] == sentence_id]
        
        if sentence_data.empty:
            return {"error": f"Sentence ID {sentence_id} not found"}
        
        comparison = {
            "sentence_id": sentence_id,
            "sentence_text": sentence_data.iloc[0]['sentence_text'],
            "results_by_model": {}
        }
        
        for _, row in sentence_data.iterrows():
            key = f"{row['model_name']}_{row['prompt_type']}_level{row['level']}"
            comparison["results_by_model"][key] = {
                "sdoh_factors": row['sdoh_factors'],
                "has_sdoh": row['has_sdoh'],
                "factors_count": row['factors_count']
            }
        
        return comparison