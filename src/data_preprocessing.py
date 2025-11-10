"""
Data preprocessing module for SQL generation with LoRA fine-tuning.
Handles loading, cleaning, and formatting the sql-create-context dataset.
"""

import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict, List, Optional
import re


class SQLDataPreprocessor:
    """Preprocesses SQL generation dataset for instruction fine-tuning."""
    
    def __init__(self, dataset_name: str = "b-mc2/sql-create-context"):
        """
        Initialize the preprocessor.
        
        Args:
            dataset_name: HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_data(self) -> DatasetDict:
        """Load the dataset from HuggingFace."""
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        print(f"Dataset loaded. Keys: {self.dataset.keys()}")
        
        # Print dataset statistics
        for split in self.dataset.keys():
            print(f"{split}: {len(self.dataset[split])} examples")
        
        return self.dataset
    
    def clean_sql(self, sql: str) -> str:
        """
        Clean and normalize SQL queries.
        
        Args:
            sql: Raw SQL query string
            
        Returns:
            Cleaned SQL query
        """
        if not sql:
            return ""
        
        # Remove extra whitespace
        sql = " ".join(sql.split())
        
        # Ensure consistent spacing around operators
        sql = re.sub(r'\s*=\s*', ' = ', sql)
        sql = re.sub(r'\s*<\s*', ' < ', sql)
        sql = re.sub(r'\s*>\s*', ' > ', sql)
        
        # Normalize SQL keywords (optional - comment out if you want to preserve case)
        # sql = sql.upper()
        
        return sql.strip()
    
    def format_instruction(self, example: Dict) -> Dict:
        """
        Format a single example for instruction fine-tuning.
        Uses Alpaca-style instruction format.
        
        Args:
            example: Dictionary with 'question', 'context', 'answer' keys
            
        Returns:
            Dictionary with formatted 'text' field
        """
        context = example.get('context', '').strip()
        question = example.get('question', '').strip()
        sql = self.clean_sql(example.get('answer', ''))
        
        # Alpaca-style instruction format
        instruction_text = (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n"
            f"Generate a SQL query to answer the following question given the database schema.\n\n"
            f"### Input:\n"
            f"Database Schema:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"### Response:\n"
            f"{sql}"
        )
        
        return {
            'text': instruction_text,
            'question': question,
            'context': context,
            'sql': sql
        }
    
    def format_instruction_simple(self, example: Dict) -> Dict:
        """
        Simpler instruction format (alternative).
        
        Args:
            example: Dictionary with 'question', 'context', 'answer' keys
            
        Returns:
            Dictionary with formatted 'text' field
        """
        context = example.get('context', '').strip()
        question = example.get('question', '').strip()
        sql = self.clean_sql(example.get('answer', ''))
        
        instruction_text = (
            f"### Database Schema:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### SQL Query:\n{sql}"
        )
        
        return {
            'text': instruction_text,
            'question': question,
            'context': context,
            'sql': sql
        }
    
    def split_dataset(self, 
                     train_size: float = 0.8, 
                     val_size: float = 0.1,
                     seed: int = 42) -> DatasetDict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train_size: Proportion for training
            val_size: Proportion for validation
            seed: Random seed for reproducibility
            
        Returns:
            DatasetDict with train/val/test splits
        """
        if self.dataset is None:
            self.load_data()
        
        # Get the train split (this dataset only has 'train')
        full_dataset = self.dataset['train']
        
        # Calculate test size
        test_size = 1.0 - train_size - val_size
        
        # First split: separate test set
        train_val = full_dataset.train_test_split(
            test_size=test_size, 
            seed=seed
        )
        
        # Second split: separate train and validation
        val_proportion = val_size / (train_size + val_size)
        train_test = train_val['train'].train_test_split(
            test_size=val_proportion,
            seed=seed
        )
        
        # Create final dataset dict
        splits = DatasetDict({
            'train': train_test['train'],
            'validation': train_test['test'],
            'test': train_val['test']
        })
        
        print(f"\nDataset splits:")
        for split, data in splits.items():
            print(f"  {split}: {len(data)} examples")
        
        return splits
    
    def preprocess_dataset(self, 
                          format_type: str = 'alpaca',
                          train_size: float = 0.8,
                          val_size: float = 0.1,
                          seed: int = 42) -> DatasetDict:
        """
        Complete preprocessing pipeline.
        
        Args:
            format_type: 'alpaca' or 'simple' instruction format
            train_size: Proportion for training
            val_size: Proportion for validation
            seed: Random seed
            
        Returns:
            Preprocessed DatasetDict ready for training
        """
        # Load and split
        splits = self.split_dataset(train_size, val_size, seed)
        
        # Choose formatting function
        format_fn = (self.format_instruction if format_type == 'alpaca' 
                    else self.format_instruction_simple)
        
        # Apply formatting to all splits
        processed = DatasetDict({
            split: data.map(
                format_fn,
                remove_columns=data.column_names,
                desc=f"Formatting {split} set"
            )
            for split, data in splits.items()
        })
        
        return processed
    
    def save_processed_data(self, 
                           processed_dataset: DatasetDict, 
                           output_dir: str = "./data/processed"):
        """
        Save processed dataset to disk.
        
        Args:
            processed_dataset: Preprocessed dataset
            output_dir: Directory to save data
        """
        print(f"\nSaving processed dataset to {output_dir}")
        processed_dataset.save_to_disk(output_dir)
        print("Dataset saved successfully!")
        
    def export_samples_to_json(self, 
                               processed_dataset: DatasetDict,
                               output_file: str = "./data/samples.json",
                               n_samples: int = 5):
        """
        Export sample examples to JSON for inspection.
        
        Args:
            processed_dataset: Preprocessed dataset
            output_file: Output JSON file path
            n_samples: Number of samples to export
        """
        samples = {
            split: [processed_dataset[split][i] for i in range(min(n_samples, len(processed_dataset[split])))]
            for split in processed_dataset.keys()
        }
        
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"\nExported {n_samples} samples per split to {output_file}")


def main():
    """Example usage of the preprocessor."""
    
    # Initialize preprocessor
    preprocessor = SQLDataPreprocessor()
    
    # Load and preprocess data
    processed_dataset = preprocessor.preprocess_dataset(
        format_type='alpaca',  # or 'simple'
        train_size=0.8,
        val_size=0.1,
        seed=42
    )
    
    # Save processed data
    preprocessor.save_processed_data(processed_dataset)
    
    # Export samples for inspection
    preprocessor.export_samples_to_json(processed_dataset, n_samples=3)
    
    # Print example
    print("\n" + "="*80)
    print("EXAMPLE FORMATTED INSTRUCTION:")
    print("="*80)
    print(processed_dataset['train'][0]['text'])
    print("="*80)


if __name__ == "__main__":
    main()