import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import json
import base64
from io import BytesIO
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import requests

class DataAnalyzer:
    def __init__(self, csv_path: str):
        # Get token from environment variable
        self.token = os.environ.get("AI_PROXY")
        if not self.token:
            raise ValueError("AI_PROXY environment variable not set")
            
        # Safely load CSV with error handling
        try:
            self.df = pd.read_csv(csv_path)
            if len(self.df.columns) == 0:
                raise ValueError("CSV file has no columns")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
            
        self.filename = os.path.basename(csv_path)
        self.plots: List[str] = []

    def query_llm(self, messages: List[Dict[str, str]], functions: List[Dict] = None) -> Dict:
        """Query the LLM through AI Proxy with error handling."""
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages
            }
            if functions:
                payload["functions"] = functions
                
            response = requests.post(
                "https://api.aiproxy.io/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # Add timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error querying LLM: {str(e)}")

    def get_data_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive data summary with error handling."""
        try:
            summary = {
                "filename": self.filename,
                "rows": len(self.df),
                "columns": list(self.df.columns),
                "dtypes": self.df.dtypes.astype(str).to_dict(),
                "null_counts": self.df.isnull().sum().to_dict(),
                "numeric_summaries": {},
                "categorical_summaries": {}
            }
            
            # Handle numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary["numeric_summaries"] = self.df[numeric_cols].describe().to_dict()
            
            # Handle categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                summary["categorical_summaries"][col] = {
                    "unique_values": self.df[col].nunique(),
                    "top_values": self.df[col].value_counts().head(5).to_dict()
                }
            
            # Get sample values safely
            summary["sample_values"] = {
                col: self.df[col].dropna().sample(min(3, len(self.df))).tolist()
                for col in self.df.columns
            }
            
            return summary
        except Exception as e:
            raise RuntimeError(f"Error generating data summary: {str(e)}")

    # ... [rest of the methods remain the same] ...

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_file>")
        sys.exit(1)
        
    try:
        analyzer = DataAnalyzer(sys.argv[1])
        analyzer.analyze_and_visualize()
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()
