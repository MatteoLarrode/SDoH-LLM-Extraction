# ===========================================================
# === Helper functions for batch processing of SDoH notes ===
# ===========================================================

import pandas as pd
import hashlib
import time
import psutil
import gc
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch

class TimingMemoryTracker:
    """Utility class for tracking timing and memory usage"""
    
    def __init__(self):
        self.start_time = None
        self.note_times = []
        self.memory_usage = []
        self.process = psutil.Process()
    
    def start_tracking(self):
        """Start timing and memory tracking"""
        self.start_time = time.time()
        self.note_times = []
        self.memory_usage = []
        self._record_memory()
    
    def record_note_processing(self, note_start_time: float):
        """Record timing for a single note"""
        note_duration = time.time() - note_start_time
        self.note_times.append(note_duration)
        self._record_memory()
    
    def _record_memory(self):
        """Record current memory usage"""
        memory_info = self.process.memory_info()
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        self.memory_usage.append({
            'ram_mb': memory_info.rss / 1024**2,
            'gpu_mb': gpu_memory,
            'timestamp': time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timing and memory summary"""
        if not self.note_times:
            return {}
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Memory stats
        ram_usage = [m['ram_mb'] for m in self.memory_usage]
        gpu_usage = [m['gpu_mb'] for m in self.memory_usage]
        
        return {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'notes_processed': len(self.note_times),
            'avg_time_per_note': sum(self.note_times) / len(self.note_times),
            'min_time_per_note': min(self.note_times),
            'max_time_per_note': max(self.note_times),
            'notes_per_second': len(self.note_times) / total_time if total_time > 0 else 0,
            'ram_usage_mb': {
                'min': min(ram_usage) if ram_usage else 0,
                'max': max(ram_usage) if ram_usage else 0,
                'final': ram_usage[-1] if ram_usage else 0
            },
            'gpu_usage_mb': {
                'min': min(gpu_usage) if gpu_usage else 0,
                'max': max(gpu_usage) if gpu_usage else 0,
                'final': gpu_usage[-1] if gpu_usage else 0
            }
        }


class BatchSDoHProcessor:
    """Simplified batch processor that outputs clean DataFrames for analysis"""
    
    def __init__(self, output_dir: str = "results/batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.tracker = TimingMemoryTracker()
    
    def create_note_id(self, case_reference: str, note_text: str, index: int) -> str:
        """Create deterministic note ID using Case Reference"""
        if pd.notna(case_reference) and str(case_reference).strip():
            # Use case reference if available
            clean_ref = str(case_reference).strip().replace(' ', '_').replace('/', '_')
            return f"case_{clean_ref}"
        else:
            # Fallback to hash method
            note_hash = hashlib.md5(note_text.encode('utf-8')).hexdigest()[:8]
            return f"note_{index:04d}_{note_hash}"
    
    def create_sentence_id(self, note_id: str, sentence_num: int) -> str:
        """Create sentence ID within note"""
        return f"{note_id}_s{sentence_num:02d}"
    
    def process_to_dataframe(self, 
                           df: pd.DataFrame, 
                           extractor, 
                           note_column: str,
                           case_ref_column: str = "Case Reference",
                           model_name: str = None,
                           start_index: int = 0,
                           batch_size: int = None) -> pd.DataFrame:
        """
        Process notes directly to a clean DataFrame with timing and memory tracking
        
        Args:
            df: DataFrame containing notes
            extractor: SDoHExtractor instance
            note_column: Column name with note text
            case_ref_column: Column name with case reference
            model_name: Optional model identifier
            start_index: Starting index
            batch_size: Number of notes to process (None = all)
        
        Returns:
            Clean DataFrame with columns: note_id, sentence_id, sentence, sdoh_factors, model
        """
        # Start tracking
        self.tracker.start_tracking()
        
        # Select subset if batch_size specified
        if batch_size is not None:
            end_index = min(start_index + batch_size, len(df))
            batch_df = df.iloc[start_index:end_index]
        else:
            batch_df = df.iloc[start_index:]
        
        # Get model name, prompt type, and level from extractor
        if model_name is None:
            model_name = getattr(extractor.tokenizer, 'name_or_path', 'unknown').split('/')[-1]
        prompt_type = extractor.prompt_type
        level = extractor.level
        
        results = []
        
        print(f"Processing {len(batch_df)} notes with timing and memory tracking...")
        
        for idx, row in batch_df.iterrows():
            note_start_time = time.time()
            
            note_text = row[note_column]
            if pd.isna(note_text) or not note_text.strip():
                continue
            
            # Use case reference for note ID
            case_ref = row.get(case_ref_column, None)
            print(f"Processing note {idx} -> {case_ref}")
            
            # Extract using your existing method
            extraction_result = extractor.extract_from_note(note_text)
            
            # Convert to flat structure
            for sentence_data in extraction_result["sentences"]:
                sentence_id = self.create_sentence_id(case_ref, sentence_data["sentence_number"])
                
                # Get factors as comma-separated string
                factors = sentence_data["sdoh_factors"]
                sdoh_factors = ", ".join(factors) if factors else "NoSDoH"
                
                results.append({
                    "case_reference": case_ref,
                    "sentence_id": sentence_id,
                    "sentence": sentence_data["sentence"],
                    "sdoh_factors": sdoh_factors,
                    "has_sdoh": factors != ["NoSDoH"] and bool(factors),
                    "num_factors": len(factors) if factors != ["NoSDoH"] else 0,
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "level": level,
                })
            
            # Record timing for this note
            self.tracker.record_note_processing(note_start_time)
        
        # Print timing summary
        timing_summary = self.tracker.get_summary()
        self.print_timing_summary(timing_summary)
        
        return pd.DataFrame(results)
    
    def print_timing_summary(self, summary: Dict[str, Any]):
        """Print timing and memory summary"""
        if not summary:
            return
        
        print("\n" + "="*50)
        print("TIMING AND MEMORY SUMMARY")
        print("="*50)
        print(f"Total processing time: {summary['total_time_minutes']:.2f} minutes")
        print(f"Notes processed: {summary['notes_processed']}")
        print(f"Average time per note: {summary['avg_time_per_note']:.2f} seconds")
        print(f"Processing rate: {summary['notes_per_second']:.2f} notes/second")
        print(f"Fastest note: {summary['min_time_per_note']:.2f} seconds")
        print(f"Slowest note: {summary['max_time_per_note']:.2f} seconds")
        print(f"RAM usage: {summary['ram_usage_mb']['final']:.0f} MB (peak: {summary['ram_usage_mb']['max']:.0f} MB)")
        if summary['gpu_usage_mb']['max'] > 0:
            print(f"GPU memory: {summary['gpu_usage_mb']['final']:.0f} MB (peak: {summary['gpu_usage_mb']['max']:.0f} MB)")
        print("="*50)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save DataFrame to CSV"""
        if filename is None:
            filename = f"batch_results_{df['model'].iloc[0].replace('/', '_')}.csv"
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        return str(filepath)

# Create optimised processor 
# class OptimizedBatchProcessor:
    #"""High-performance batch processor for large-scale SDoH extraction"""
    
