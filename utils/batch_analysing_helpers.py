import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')
    
class ModelConsistencyAnalyser:
    """Analyse consistency and agreement between different models/prompts for SDoH extraction"""
    
    def __init__(self, combined_df: pd.DataFrame):
        """
        Initialise analyser with combined results
        
        Args:
            combined_df: DataFrame with columns including sentence_id, model, sdoh_factors, has_sdoh
        """
        self.df = combined_df.copy()
        self.setup_data()
    
    def setup_data(self):
        """Prepare data for analysis"""
        # Create model-prompt combinations for clearer analysis
        if 'prompt_type' in self.df.columns:
            self.df['model_prompt'] = self.df['model'] + '_' + self.df['prompt_type']
        else:
            # Extract from model column if it contains prompt info
            self.df['model_prompt'] = self.df['model']
        
        # Get unique models and prompts
        self.models = self.df['model_prompt'].unique()
        self.sentences = self.df['sentence_id'].unique()
        
        print(f"Loaded data: {len(self.sentences)} sentences, {len(self.models)} model-prompt combinations")
    
    def calculate_pairwise_agreement(self) -> pd.DataFrame:
        """Calculate pairwise agreement between all model combinations"""
        agreement_matrix = []
        
        for model1 in self.models:
            for model2 in self.models:
                if model1 != model2:
                    agreement = self._calculate_agreement_between_models(model1, model2)
                    agreement_matrix.append({
                        'model1': model1,
                        'model2': model2,
                        'sentence_agreement': agreement['sentence_agreement'],
                        'factor_agreement': agreement['factor_agreement'],
                        'cohen_kappa': agreement['cohen_kappa'],
                        'common_sentences': agreement['common_sentences']
                    })
        
        return pd.DataFrame(agreement_matrix)
    
    def _calculate_agreement_between_models(self, model1: str, model2: str) -> Dict:
        """Calculate agreement metrics between two models"""
        # Get predictions for both models
        m1_data = self.df[self.df['model_prompt'] == model1].set_index('sentence_id')
        m2_data = self.df[self.df['model_prompt'] == model2].set_index('sentence_id')
        
        # Find common sentences (convert to list for pandas indexing)
        common_sentences = list(set(m1_data.index) & set(m2_data.index))
        
        if len(common_sentences) == 0:
            return {'sentence_agreement': 0, 'factor_agreement': 0, 'cohen_kappa': 0, 'common_sentences': 0}
        
        # Binary agreement (SDoH vs No SDoH)
        m1_binary = m1_data.loc[common_sentences, 'has_sdoh']
        m2_binary = m2_data.loc[common_sentences, 'has_sdoh']
        sentence_agreement = (m1_binary == m2_binary).mean()
        
        # Cohen's Kappa for binary classification
        try:
            kappa = cohen_kappa_score(m1_binary, m2_binary)
        except:
            kappa = 0
        
        # Factor-level agreement (for sentences where both found SDoH)
        m1_sdoh_sentences = set(m1_data[m1_data['has_sdoh']].index)
        m2_sdoh_sentences = set(m2_data[m2_data['has_sdoh']].index)
        both_sdoh = list(set(common_sentences) & m1_sdoh_sentences & m2_sdoh_sentences)
        
        if len(both_sdoh) > 0:
            factor_agreements = []
            for sent_id in both_sdoh:
                factors1 = set(m1_data.loc[sent_id, 'sdoh_factors'].split(', '))
                factors2 = set(m2_data.loc[sent_id, 'sdoh_factors'].split(', '))
                
                # Remove NoSDoH if present
                factors1.discard('NoSDoH')
                factors2.discard('NoSDoH')
                
                # Jaccard similarity
                if len(factors1) == 0 and len(factors2) == 0:
                    factor_agreements.append(1.0)  # Both found no specific factors
                else:
                    intersection = len(factors1 & factors2)
                    union = len(factors1 | factors2)
                    factor_agreements.append(intersection / union if union > 0 else 0)
            
            factor_agreement = np.mean(factor_agreements)
        else:
            factor_agreement = 0
        
        return {
            'sentence_agreement': sentence_agreement,
            'factor_agreement': factor_agreement,
            'cohen_kappa': kappa,
            'common_sentences': len(common_sentences)
        }
    
    def model_consistency_ranking(self) -> pd.DataFrame:
        """Rank models by their consistency with others"""
        agreement_df = self.calculate_pairwise_agreement()
        
        # Calculate average agreement for each model
        model_consistency = []
        
        for model in self.models:
            # Get agreements where this model is involved
            as_model1 = agreement_df[agreement_df['model1'] == model]
            as_model2 = agreement_df[agreement_df['model2'] == model]
            
            all_agreements = pd.concat([as_model1, as_model2])
            
            if len(all_agreements) > 0:
                avg_sentence_agreement = all_agreements['sentence_agreement'].mean()
                avg_factor_agreement = all_agreements['factor_agreement'].mean()
                avg_kappa = all_agreements['cohen_kappa'].mean()
                
                # Overall consistency score (weighted average)
                consistency_score = (0.4 * avg_sentence_agreement + 
                                   0.3 * avg_factor_agreement + 
                                   0.3 * avg_kappa)
                
                model_consistency.append({
                    'model': model,
                    'avg_sentence_agreement': avg_sentence_agreement,
                    'avg_factor_agreement': avg_factor_agreement,
                    'avg_cohen_kappa': avg_kappa,
                    'consistency_score': consistency_score,
                    'num_comparisons': len(all_agreements)
                })
        
        consistency_df = pd.DataFrame(model_consistency)
        return consistency_df.sort_values('consistency_score', ascending=False)
    
    def detection_rate_analysis(self) -> pd.DataFrame:
        """Analyse SDoH detection rates by model"""
        detection_stats = self.df.groupby('model_prompt').agg({
            'has_sdoh': ['count', 'sum', 'mean'],
            'num_factors': ['mean', 'std'],
            'sentence_id': 'nunique'
        }).round(3)
        
        # Flatten column names
        detection_stats.columns = [
            'total_sentences', 'sentences_with_sdoh', 'detection_rate',
            'avg_factors_per_sentence', 'std_factors_per_sentence', 'unique_sentences'
        ]
        
        return detection_stats.reset_index().sort_values('detection_rate', ascending=False)
    
    def factor_consistency_analysis(self) -> Dict[str, pd.DataFrame]:
        """Analyse consistency in specific factor detection"""
        # Get all unique factors
        all_factors = set()
        for factors_str in self.df[self.df['has_sdoh']]['sdoh_factors']:
            factors = [f.strip() for f in factors_str.split(',') if f.strip() != 'NoSDoH']
            all_factors.update(factors)
        
        factor_consistency = {}
        
        for factor in all_factors:
            # For each factor, see how consistently models detect it
            factor_detections = []
            
            for model in self.models:
                model_data = self.df[self.df['model_prompt'] == model]
                
                # Count sentences where this factor was detected
                factor_count = 0
                total_sentences = len(model_data)
                
                for _, row in model_data.iterrows():
                    if row['has_sdoh']:
                        factors = [f.strip() for f in row['sdoh_factors'].split(',')]
                        if factor in factors:
                            factor_count += 1
                
                factor_detections.append({
                    'model': model,
                    'factor': factor,
                    'detections': factor_count,
                    'total_sentences': total_sentences,
                    'detection_rate': factor_count / total_sentences if total_sentences > 0 else 0
                })
            
            factor_consistency[factor] = pd.DataFrame(factor_detections)
        
        return factor_consistency
    
    def disagreement_analysis(self) -> pd.DataFrame:
        """Find sentences with highest disagreement between models"""
        disagreement_scores = []
        
        for sentence_id in self.sentences:
            sentence_data = self.df[self.df['sentence_id'] == sentence_id]
            
            if len(sentence_data) < 2:  # Need at least 2 models
                continue
            
            # Calculate disagreement metrics
            sdoh_predictions = sentence_data['has_sdoh'].values
            
            # Variance in binary predictions (0 = all agree, 0.25 = maximum disagreement)
            disagreement_score = np.var(sdoh_predictions.astype(int))
            
            # Number of different factor sets
            unique_factors = sentence_data['sdoh_factors'].nunique()
            
            disagreement_scores.append({
                'sentence_id': sentence_id,
                'sentence': sentence_data.iloc[0]['sentence'],
                'disagreement_score': disagreement_score,
                'unique_factor_sets': unique_factors,
                'num_models': len(sentence_data),
                'models_detecting_sdoh': sdoh_predictions.sum(),
                'all_predictions': ', '.join([f"{row['model_prompt']}: {row['sdoh_factors']}" 
                                            for _, row in sentence_data.iterrows()])
            })
        
        disagreement_df = pd.DataFrame(disagreement_scores)
        return disagreement_df.sort_values('disagreement_score', ascending=False)
    
    def visualise_consistency(self, save_plots: bool = True):
        """Create comprehensive visualisation of model consistency"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Consistency Ranking
        plt.subplot(2, 3, 1)
        consistency_ranking = self.model_consistency_ranking()
        
        # Shorten model names for display
        consistency_ranking['short_model'] = consistency_ranking['model'].apply(
            lambda x: x.split('_')[0] + '_' + x.split('_')[-1] if '_' in x else x
        )
        
        plt.barh(consistency_ranking['short_model'], consistency_ranking['consistency_score'])
        plt.xlabel('Consistency Score')
        plt.title('Model Consistency Ranking\n(Higher = More Consistent)')
        plt.gca().invert_yaxis()
        
        # 2. Detection Rate Comparison
        plt.subplot(2, 3, 2)
        detection_rates = self.detection_rate_analysis()
        detection_rates['short_model'] = detection_rates['model_prompt'].apply(
            lambda x: x.split('_')[0] + '_' + x.split('_')[-1] if '_' in x else x
        )
        
        plt.bar(range(len(detection_rates)), detection_rates['detection_rate'])
        plt.xticks(range(len(detection_rates)), detection_rates['short_model'], rotation=45)
        plt.ylabel('SDoH Detection Rate')
        plt.title('SDoH Detection Rates by Model')
        
        # 3. Agreement Heatmap
        plt.subplot(2, 3, 3)
        agreement_df = self.calculate_pairwise_agreement()
        
        # Create agreement matrix
        agreement_matrix = np.zeros((len(self.models), len(self.models)))
        model_to_idx = {model: i for i, model in enumerate(self.models)}
        
        for _, row in agreement_df.iterrows():
            i = model_to_idx[row['model1']]
            j = model_to_idx[row['model2']]
            agreement_matrix[i, j] = row['sentence_agreement']
        
        # Fill diagonal with 1s
        np.fill_diagonal(agreement_matrix, 1.0)
        
        short_models = [m.split('_')[0] + '_' + m.split('_')[-1] if '_' in m else m for m in self.models]
        
        sns.heatmap(agreement_matrix, 
                   xticklabels=short_models, 
                   yticklabels=short_models,
                   annot=True, fmt='.2f', cmap='RdYlBu_r',
                   vmin=0, vmax=1)
        plt.title('Pairwise Agreement Matrix\n(Binary SDoH Detection)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 4. Factor Agreement Distribution
        plt.subplot(2, 3, 4)
        factor_agreements = agreement_df['factor_agreement'].dropna()
        plt.hist(factor_agreements, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Factor Agreement Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Factor-Level\nAgreement Scores')
        plt.axvline(factor_agreements.mean(), color='red', linestyle='--', 
                   label=f'Mean: {factor_agreements.mean():.2f}')
        plt.legend()
        
        # 5. Disagreement Analysis
        plt.subplot(2, 3, 5)
        disagreement_df = self.disagreement_analysis()
        top_disagreements = disagreement_df.head(10)
        
        plt.barh(range(len(top_disagreements)), top_disagreements['disagreement_score'])
        plt.yticks(range(len(top_disagreements)), 
                  [f"Sent {i+1}" for i in range(len(top_disagreements))])
        plt.xlabel('Disagreement Score')
        plt.title('Top 10 Most Disagreed\nSentences')
        plt.gca().invert_yaxis()
        
        # 6. Cohen's Kappa Distribution
        plt.subplot(2, 3, 6)
        kappa_scores = agreement_df['cohen_kappa'].dropna()
        plt.hist(kappa_scores, bins=20, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel("Cohen's Kappa")
        plt.ylabel('Frequency')
        plt.title("Distribution of Cohen's Kappa\nScores")
        plt.axvline(kappa_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {kappa_scores.mean():.2f}')
        
        # Add interpretation lines
        plt.axvline(0.2, color='orange', linestyle=':', alpha=0.7, label='Fair')
        plt.axvline(0.4, color='yellow', linestyle=':', alpha=0.7, label='Moderate')
        plt.axvline(0.6, color='lightgreen', linestyle=':', alpha=0.7, label='Substantial')
        plt.legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_consistency_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualisation saved as 'model_consistency_analysis.png'")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive consistency report"""
        report = []
        report.append("="*60)
        report.append("MODEL CONSISTENCY ANALYSIS REPORT")
        report.append("="*60)
        
        # Basic stats
        report.append(f"\nDATASET OVERVIEW:")
        report.append(f"• Total sentences analysed: {len(self.sentences)}")
        report.append(f"• Model-prompt combinations: {len(self.models)}")
        report.append(f"• Models tested: {self.models}")
        
        # Consistency ranking
        consistency_ranking = self.model_consistency_ranking()
        report.append(f"\nMODEL CONSISTENCY RANKING:")
        report.append("(Consistency Score: 0=Low, 1=High)")
        
        for i, row in consistency_ranking.iterrows():
            report.append(f"{i+1:2d}. {row['model']:<40} Score: {row['consistency_score']:.3f}")
        
        # Detection rates
        detection_rates = self.detection_rate_analysis()
        report.append(f"\nSDOH DETECTION RATES:")
        
        for _, row in detection_rates.iterrows():
            report.append(f"• {row['model_prompt']:<40} {row['detection_rate']:.1%} "
                         f"({row['sentences_with_sdoh']}/{row['total_sentences']} sentences)")
        
        # Agreement summary
        agreement_df = self.calculate_pairwise_agreement()
        if len(agreement_df) > 0:
            avg_sentence_agreement = agreement_df['sentence_agreement'].mean()
            avg_factor_agreement = agreement_df['factor_agreement'].mean()
            avg_kappa = agreement_df['cohen_kappa'].mean()
            
            report.append(f"\nOVERALL AGREEMENT METRICS:")
            report.append(f"• Average sentence agreement: {avg_sentence_agreement:.1%}")
            report.append(f"• Average factor agreement: {avg_factor_agreement:.1%}")
            report.append(f"• Average Cohen's Kappa: {avg_kappa:.3f}")
            
            # Kappa interpretation
            if avg_kappa < 0.2:
                kappa_interp = "Poor"
            elif avg_kappa < 0.4:
                kappa_interp = "Fair"
            elif avg_kappa < 0.6:
                kappa_interp = "Moderate"
            elif avg_kappa < 0.8:
                kappa_interp = "Substantial"
            else:
                kappa_interp = "Almost Perfect"
            
            report.append(f"• Kappa interpretation: {kappa_interp}")
        
        # Top disagreements
        disagreement_df = self.disagreement_analysis()
        report.append(f"\nTOP 5 MOST DISAGREED SENTENCES:")
        
        for i, row in disagreement_df.head(5).iterrows():
            report.append(f"\n{i+1}. Disagreement Score: {row['disagreement_score']:.3f}")
            report.append(f"   Sentence: {row['sentence'][:100]}...")
            report.append(f"   Predictions: {row['all_predictions']}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


# Usage example and utility functions
def load_and_analyse_consistency(csv_directory: str) -> ModelConsistencyAnalyser:
    """
    Load all CSV files from directory and create consistency analyser
    
    Args:
        csv_directory: Path to directory containing CSV files
    
    Returns:
        ModelConsistencyAnalyser instance
    """
    import glob
    
    csv_files = glob.glob(f"{csv_directory}/*.csv")
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and combine all files
    dfs = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} rows")
    
    return ModelConsistencyAnalyser(combined_df)

# Example usage:
"""
# Load and analyse
analyser = load_and_analyse_consistency("results/comparison_batch")

# Generate visualisations
analyser.visualise_consistency(save_plots=True)

# Get consistency ranking
ranking = analyser.model_consistency_ranking()
print(ranking)

# Generate full report
report = analyser.generate_report()
print(report)

# Save report to file
with open('consistency_report.txt', 'w') as f:
    f.write(report)
"""