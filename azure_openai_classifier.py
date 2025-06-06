import pandas as pd
import os
from datetime import datetime
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import time
import re
from typing import Dict, List, Tuple, Optional

# Load environment variables
load_dotenv()

# Set up logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"azure_openai_classification_logs_{current_time}.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Azure OpenAI configuration
client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
)

# Log configuration (without sensitive data)
logger.info("Azure OpenAI Configuration:")
logger.info(f"API Version: {os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')}")
logger.info(f"Endpoint: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
logger.info(f"Deployment Name: {os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')}")

# Numerical pattern detection
NUMERICAL_PATTERNS = [
    # Basic numerical patterns
    r'\b\d+\b',                    # Basic digits
    r'\b\d+\.\d+\b',              # Decimal numbers
    r'\b\d+\s*(?:kg|g|mg|ml|l|bpm|mmHg|%|years?|months?|days?|hours?|minutes?|seconds?)\b',  # Units
    r'\$\d+(?:\.\d{2})?',         # Money
    r'\b\d+(?:\.\d+)?%\b',        # Percentages
    r'\b\d{1,2}:\d{2}(?::\d{2})?\b',  # Time
    
    # Medical-specific patterns
    r'\b(?:blood\s+)?(?:sugar|glucose)\s+level\b',
    r'\b(?:blood\s+)?(?:pressure|bp)\s+reading\b',
    r'\b(?:heart\s+)?rate\b',
    r'\b(?:body\s+)?(?:weight|mass|bmi)\b',
    r'\b(?:cholesterol|hdl|ldl)\s+level\b',
    r'\b(?:triglyceride|trig)\s+level\b',
    r'\b(?:hemoglobin|hba1c)\s+level\b',
    r'\b(?:insulin|glucose)\s+dose\b',
    r'\b(?:blood\s+)?(?:test|testing)\s+result\b',
    r'\b(?:lab|laboratory)\s+value\b',
    r'\b(?:medical|clinical)\s+measurement\b',
    
    # Range and threshold patterns
    r'\b(?:normal|reference|target)\s+range\b',
    r'\b(?:above|below|under|over)\s+\d+\b',
    r'\b(?:less|more|greater|fewer)\s+than\s+\d+\b',
    r'\b(?:every|each|per)\s+\d+\b',
    r'\b(?:once|twice|thrice)\s+(?:daily|weekly|monthly|yearly)\b',
    r'\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b',  # Ranges
    
    # Frequency and duration patterns
    r'\b(?:every|each|per)\s+\d+\s*(?:day|week|month|year|hour|minute|second)\b',
    r'\b(?:for|over|during)\s+\d+\s*(?:day|week|month|year|hour|minute|second)\b',
    r'\b(?:last|past|previous)\s+\d+\s*(?:day|week|month|year|hour|minute|second)\b',
    
    # Measurement patterns
    r'\b(?:measure|measurement|monitor|monitoring|check|checking|test|testing)\b',
    r'\b(?:frequency|interval|duration|period|time|times)\b',
    r'\b(?:amount|quantity|number|count|total|sum)\b',
    
    # Statistical patterns
    r'\b(?:average|mean|median|mode|range|standard deviation)\b',
    r'\b(?:percentage|percent|rate|ratio|proportion|fraction)\b',
    r'\b(?:probability|likelihood|chance|risk|odds)\b'
]

# Medical terms to exclude from numerical detection
MEDICAL_TERMS = [
    r'Type\s+[12]\s+Diabetes',
    r'Type\s+[12]\s+DM',
    r'Type\s+[12]\s+diabetes',
    r'Type\s+[12]\s+diabetic',
    r'Type\s+[12]\s+condition',
    r'Type\s+[12]\s+patient',
    r'Type\s+[12]\s+management',
    r'Type\s+[12]\s+treatment',
    r'Type\s+[12]\s+diagnosis',
    r'Type\s+[12]\s+prevention',
    r'Type\s+[12]\s+complications',
    r'Type\s+[12]\s+risk',
    r'Type\s+[12]\s+factors',
    r'Type\s+[12]\s+symptoms',
    r'Type\s+[12]\s+signs',
    r'Type\s+[12]\s+causes',
    r'Type\s+[12]\s+effects',
    r'Type\s+[12]\s+impact',
    r'Type\s+[12]\s+outcomes',
    r'Type\s+[12]\s+prognosis'
]

def contains_numerical_pattern(text: str) -> bool:
    """
    Check if the text contains any numerical patterns, excluding medical terms.
    """
    if pd.isna(text):
        return False
        
    # First check for medical terms to exclude
    for term in MEDICAL_TERMS:
        if re.search(term, text, re.IGNORECASE):
            return False
    
    # Then check for numerical patterns
    for pattern in NUMERICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False

# Enhanced classification prompt
CLASSIFICATION_PROMPT = """You are a medical assistant trained to classify health-related question-answer (Q/A) pairs.

Each pair must be classified into one of the following categories:

1. Numerical – if the **answer includes ANY of these numerical details**:
   - Specific values (e.g., 98.6°F, 120/80 mmHg)
   - Ranges or thresholds (e.g., 70-99 mg/dL is normal)
   - Durations (e.g., for 7 days, over 3 months)
   - Percentages/statistics (e.g., 20% of patients, 1 in 5 cases)
   - Dosages (e.g., 500mg twice a day)
   - Test results or scores (e.g., HbA1c of 6.5%)
   - Any measurement-based or quantity-focused response
   - Frequency indicators (e.g., twice daily, every 6 hours)
   - Time periods (e.g., for 2 weeks, over 3 months)
   - Quantities (e.g., 3 tablets, 2 cups)
   - Statistical measures (e.g., average, mean, median)
   - Risk factors (e.g., 1 in 4 chance, 25% risk)

2. Textual – if the **answer is purely descriptive, explanatory, or conceptual**, such as:
   - Symptoms or causes (e.g., fatigue, dizziness)
   - General health guidance or lifestyle suggestions
   - Explanations without referencing numbers
   - Definitions, concepts, or preventive advice
   - Treatment approaches without specific dosages
   - General recommendations without quantities

IMPORTANT CLASSIFICATION RULES:
1. If the answer contains ANY numerical information, classify as Numerical
2. Only classify as Textual if the answer contains NO numerical information
3. Consider both explicit numbers and implicit numerical references
4. Pay special attention to medical measurements, ranges, and thresholds
5. Include frequency and duration indicators in numerical classification

---

**Classify the following Q/A pair strictly in this format**:

Question: <insert question>
Answer: <insert answer>
Classification: <Numerical or Textual>
Reason: <1–2 line reason explaining the classification>

---

Here is the Q/A to classify:

Question: {question}
Answer: {answer}"""

def classify_qa_pair(question: str, answer: str, max_retries: int = 3) -> dict:
    """
    Classify a question-answer pair using Azure OpenAI API with retry logic.
    """
    # First check for numerical patterns
    has_numerical = contains_numerical_pattern(answer)
    
    for attempt in range(max_retries):
        try:
            # Prepare the prompt
            prompt = CLASSIFICATION_PROMPT.format(question=question, answer=answer)
            
            # Call Azure OpenAI API
            response = client.chat.completions.create(
                model=os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME'),
                messages=[
                    {"role": "system", "content": "You are a medical classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Use 0 temperature for consistent results
                max_tokens=150
            )
            
            # Extract the classification from the response
            classification_text = response.choices[0].message.content
            
            # Parse the classification
            lines = classification_text.strip().split('\n')
            classification = None
            reason = None
            
            for line in lines:
                if line.startswith('Classification:'):
                    classification = line.replace('Classification:', '').strip()
                elif line.startswith('Reason:'):
                    reason = line.replace('Reason:', '').strip()
            
            if not classification or not reason:
                raise ValueError("Could not parse classification or reason from response")
            
            # Validate classification against numerical pattern detection
            if has_numerical and classification == 'Textual':
                logger.warning(f"Numerical pattern detected but classified as Textual. Question: {question}")
                classification = 'Numerical'
                reason = f"Numerical pattern detected: {reason}"
            
            return {
                'Question': question,
                'Answer': answer,
                'Classification': classification,
                'Reason': reason,
                'Has_Numerical_Pattern': has_numerical
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Error classifying Q/A pair after {max_retries} attempts: {str(e)}")
                return {
                    'Question': question,
                    'Answer': answer,
                    'Classification': 'Error',
                    'Reason': f'Error during classification: {str(e)}',
                    'Has_Numerical_Pattern': has_numerical
                }
            time.sleep(2 ** attempt)  # Exponential backoff

def process_dataset(file_path: str, output_dir: str = "Classification_using_API") -> None:
    """
    Process a dataset of questions and answers using Azure OpenAI API.
    
    Args:
        file_path: Path to the input Excel file
        output_dir: Directory to save the classified results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the input file
        df = pd.read_excel(file_path)
        
        # Log column names for debugging
        logger.info(f"Columns in file: {df.columns.tolist()}")
        
        # Initialize results list
        results = []
        
        # Process each question-answer pair
        total_questions = len(df)
        for index, row in df.iterrows():
            question = row['Question']
            answer = row['Answer']
            
            logger.info(f"Processing question {index + 1}/{total_questions}")
            
            # Classify the Q/A pair
            result = classify_qa_pair(question, answer)
            results.append(result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        total_classified = len(results_df)
        numerical_count = (results_df['Classification'] == 'Numerical').sum()
        textual_count = (results_df['Classification'] == 'Textual').sum()
        error_count = (results_df['Classification'] == 'Error').sum()
        pattern_detected = results_df['Has_Numerical_Pattern'].sum()
        
        # Save results
        output_file = os.path.join(output_dir, f'API_classification_{current_time}.xlsx')
        results_df.to_excel(output_file, index=False)
        
        # Create summary report
        summary_file = os.path.join(output_dir, f'API_classification_summary_{current_time}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Azure OpenAI Classification Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Questions Processed: {total_classified}\n")
            f.write(f"Numerical Classifications: {numerical_count} ({numerical_count/total_classified*100:.2f}%)\n")
            f.write(f"Textual Classifications: {textual_count} ({textual_count/total_classified*100:.2f}%)\n")
            f.write(f"Errors: {error_count} ({error_count/total_classified*100:.2f}%)\n")
            f.write(f"Numerical Patterns Detected: {pattern_detected} ({pattern_detected/total_classified*100:.2f}%)\n\n")
            
            # Add sample classifications
            f.write("Sample Classifications:\n")
            f.write("-" * 40 + "\n\n")
            
            # Sample numerical classifications
            numerical_samples = results_df[results_df['Classification'] == 'Numerical'].head(3)
            f.write("Numerical Classifications:\n")
            for _, row in numerical_samples.iterrows():
                f.write(f"Q: {row['Question']}\n")
                f.write(f"A: {row['Answer']}\n")
                f.write(f"Classification: {row['Classification']}\n")
                f.write(f"Reason: {row['Reason']}\n")
                f.write(f"Pattern Detected: {row['Has_Numerical_Pattern']}\n\n")
            
            # Sample textual classifications
            textual_samples = results_df[results_df['Classification'] == 'Textual'].head(3)
            f.write("\nTextual Classifications:\n")
            for _, row in textual_samples.iterrows():
                f.write(f"Q: {row['Question']}\n")
                f.write(f"A: {row['Answer']}\n")
                f.write(f"Classification: {row['Classification']}\n")
                f.write(f"Reason: {row['Reason']}\n")
                f.write(f"Pattern Detected: {row['Has_Numerical_Pattern']}\n\n")
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Total questions processed: {total_classified}")
        logger.info(f"Numerical classifications: {numerical_count} ({numerical_count/total_classified*100:.2f}%)")
        logger.info(f"Textual classifications: {textual_count} ({textual_count/total_classified*100:.2f}%)")
        logger.info(f"Errors: {error_count} ({error_count/total_classified*100:.2f}%)")
        logger.info(f"Numerical patterns detected: {pattern_detected} ({pattern_detected/total_classified*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

def main():
    """Main function to run the Azure OpenAI classifier."""
    try:
        # Define the input file to process
        xlsx_file = 'Top-56-Questions-Anwsers.xlsx'
        
        logger.info(f"\nProcessing file: {xlsx_file}")
        process_dataset(xlsx_file)
            
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 