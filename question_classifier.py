import pandas as pd
import re
import logging
from typing import Dict, List, Tuple
import os
from datetime import datetime

# Medical Assistant Prompt
MEDICAL_ASSISTANT_PROMPT = """You are a medical assistant trained to classify health-related question-answer (Q/A) pairs.

Each pair must be classified into one of the following categories:

1. Numerical – if the **answer includes numeric details**, such as:
   - Specific values (e.g., 98.6°F, 120/80 mmHg)
   - Ranges or thresholds (e.g., 70-99 mg/dL is normal)
   - Durations (e.g., for 7 days, over 3 months)
   - Percentages/statistics (e.g., 20% of patients, 1 in 5 cases)
   - Dosages (e.g., 500mg twice a day)
   - Test results or scores (e.g., HbA1c of 6.5%)
   - Any measurement-based or quantity-focused response.

2. Textual – if the **answer is descriptive, explanatory, or conceptual**, such as:
   - Symptoms or causes (e.g., fatigue, dizziness)
   - General health guidance or lifestyle suggestions
   - Explanations without referencing numbers
   - Definitions, concepts, or preventive advice

---

**Classify the following Q/A pair strictly in this format**:

Question: <insert question>  
Answer: <insert answer>  
Classification: <Numerical or Textual>  
Reason: <1–2 line reason explaining the classification>

---

Here is the Q/A to classify:

Question: {{QUESTION}}  
Answer: {{ANSWER}}
"""

# Set up logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"question_classifier_logs_{current_time}.txt"

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

def is_numerical(text: str) -> bool:
    """
    Returns True if the given text contains any numerical pattern, such as:
    - Digits ("123")
    - Decimal ("45.6")
    - Units ("50 kg", "70 bpm")
    - Money, time, percentages, etc.
    
    Excludes medical terms like "Type 1 Diabetes" or "Type 2 Diabetes"
    """
    if pd.isna(text):
        return False
        
    # First, check for medical terms to exclude
    medical_terms = [
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
        r'Type\s+[12]\s+prognosis',
        r'Type\s+[12]\s+difference',
        r'Type\s+[12]\s+comparison',
        r'Type\s+[12]\s+versus',
        r'Type\s+[12]\s+vs',
        r'Type\s+[12]\s+and',
        r'Type\s+[12]\s+or',
        r'Type\s+[12]\s+versus',
        r'Type\s+[12]\s+versus',
        r'Type\s+[12]\s+versus'
    ]
    
    # Check for medical terms first
    for term in medical_terms:
        if re.search(term, text, re.IGNORECASE):
            return False
    
    # Patterns to check for numerical content
    patterns = [
        # Basic numerical patterns with word boundaries
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
    
    # Check if any pattern matches
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False

def is_numerical_question(text: str) -> bool:
    """
    Determine if a question is asking for a numerical answer based on context/phrases.
    """
    if pd.isna(text):
        return False
        
    text = str(text).lower()
    
    # Exclude certain medical questions that might trigger false positives
    exclude_phrases = [
        'how can i manage',
        'what is the difference',
        'what medications are',
        'what are the symptoms',
        'what are the causes',
        'what is the treatment',
        'what is the diagnosis',
        'what is the prevention',
        'what are the complications',
        'what are the risk factors',
        'what are the signs',
        'what are the effects',
        'what is the impact',
        'what are the outcomes',
        'what is the prognosis'
    ]
    
    for phrase in exclude_phrases:
        if phrase in text:
            return False
    
    # Common patterns/phrases for numerical questions
    numerical_phrases = [
        # Measurement-related
        'how many', 'how much', 'what is the value', 'what is the normal', 'normal range',
        'amount', 'level', 'rate', 'percentage', 'concentration', 'count', 'number of',
        'duration', 'age', 'weight', 'height', 'score', 'pressure', 'mg/dl', 'mmol/l',
        'bpm', 'mmhg', 'frequency', 'interval', 'time', 'years', 'months', 'days', 'minutes', 'seconds',
        'maximum', 'minimum', 'upper limit', 'lower limit', 'reference range', 'threshold', 'cutoff',
        'dose', 'dosage', 'volume', 'length', 'distance', 'size', 'capacity', 'proportion', 'ratio',
        'median', 'mean', 'average', 'percent', 'probability', 'risk', 'incidence', 'prevalence',
        'score', 'index', 'rate', 'value', 'quantity', 'interval', 'intervals', 'score', 'scoring',
        
        # Medical measurement-specific
        'blood sugar level', 'glucose level', 'blood pressure', 'heart rate',
        'body weight', 'bmi', 'cholesterol level', 'triglyceride level',
        'hemoglobin level', 'hba1c', 'insulin dose', 'glucose dose',
        'blood test result', 'lab value', 'medical measurement',
        
        # Frequency and monitoring
        'how often', 'how long', 'how far', 'how high', 'how low',
        'how many times', 'how many days', 'how many weeks',
        'how many months', 'how many years', 'how many hours',
        'how many minutes', 'how many seconds',
        
        # Units and measurements
        'how many milligrams', 'how many grams', 'how many kilograms',
        'how many milliliters', 'how many liters', 'how many units',
        'how many doses', 'how many pills', 'how many tablets',
        'how many capsules', 'how many injections',
        
        # Time-based
        'how many times per day', 'how many times per week',
        'how many times per month', 'how many times per year',
        'how long should', 'how often should', 'how frequently should',
        
        # Monitoring and testing
        'measure', 'measurement', 'monitor', 'monitoring',
        'check', 'checking', 'test', 'testing', 'frequency'
    ]
    
    # Check for numerical phrases
    for phrase in numerical_phrases:
        if phrase in text:
            return True
            
    # Check if the question starts with "calculate"
    if text.startswith('calculate'):
        return True
        
    return False

def classify_question_answer_pair(question: str, answer: str) -> Dict[str, str]:
    """
    Classifies a question-answer pair as either 'Numerical' or 'Textual'
    based on both the question content and answer content.
    
    Returns a dictionary with classifications for both question and answer.
    """
    if pd.isna(answer):
        return {
            'Question_Type': 'Textual',
            'Answer_Type': 'Textual',
            'Classification': 'Textual'
        }
    
    question_type = "Numerical" if is_numerical_question(question) else "Textual"
    answer_type = "Numerical" if is_numerical(answer) else "Textual"
    
    # If either question or answer is numerical, classify as numerical
    final_classification = "Numerical" if (question_type == "Numerical" or answer_type == "Numerical") else "Textual"
    
    return {
        'Question_Type': question_type,
        'Answer_Type': answer_type,
        'Classification': final_classification
    }

def process_dataset(file_path: str, output_dir: str = "classified_questions") -> None:
    """
    Process a dataset of questions and answers, classifying each pair.
    
    Args:
        file_path: Path to the input Excel/CSV file
        output_dir: Directory to save the classified results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the input file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Log column names for debugging
        logger.info(f"Columns in file: {df.columns.tolist()}")
        
        # Extract relevant columns - handle both formats
        if '1st Question' in df.columns:
            # New format
            relevant_data = pd.DataFrame({
                'Question': df['1st Question'],
                'Options': df['1st Choices'],
                'Answer': df['1st Answer']
            })
        else:
            # Original format (Top-56-Questions-Anwsers.xlsx)
            relevant_data = pd.DataFrame({
                'Question': df.iloc[:, 0],  # First column
                'Answer': df.iloc[:, 1]     # Second column
            })
        
        # Classify each question-answer pair
        classifications = relevant_data.apply(
            lambda row: classify_question_answer_pair(row['Question'], row['Answer']),
            axis=1
        )
        
        # Add classification columns
        relevant_data['Question_Type'] = classifications.apply(lambda x: x['Question_Type'])
        relevant_data['Answer_Type'] = classifications.apply(lambda x: x['Answer_Type'])
        relevant_data['Classification'] = classifications.apply(lambda x: x['Classification'])
        
        # Calculate statistics
        total_questions = len(relevant_data)
        numerical_questions = (relevant_data['Question_Type'] == 'Numerical').sum()
        numerical_answers = (relevant_data['Answer_Type'] == 'Numerical').sum()
        numerical_total = (relevant_data['Classification'] == 'Numerical').sum()
        textual_total = total_questions - numerical_total
        
        # Get filename without extension for output naming
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save results
        output_file = os.path.join(output_dir, f'{base_filename}_classified_{current_time}.xlsx')
        relevant_data.to_excel(output_file, index=False)
        
        # Create summary report
        summary_file = os.path.join(output_dir, f'{base_filename}_summary_{current_time}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Question Classification Summary for {base_filename}\n")
            f.write("=" * (len(base_filename) + 40) + "\n\n")
            f.write(f"Total Questions: {total_questions}\n")
            f.write(f"Questions with Numerical Phrases: {numerical_questions} ({numerical_questions/total_questions*100:.2f}%)\n")
            f.write(f"Questions with Numerical Answers: {numerical_answers} ({numerical_answers/total_questions*100:.2f}%)\n")
            f.write(f"Final Numerical Classification: {numerical_total} ({numerical_total/total_questions*100:.2f}%)\n")
            f.write(f"Final Textual Classification: {textual_total} ({textual_total/total_questions*100:.2f}%)\n\n")
            
            # Add sample questions for each category
            f.write("Sample Numerical Questions:\n")
            numerical_samples = relevant_data[relevant_data['Classification'] == 'Numerical'].head(5)
            for _, row in numerical_samples.iterrows():
                f.write(f"Q: {row['Question']}\n")
                f.write(f"A: {row['Answer']}\n")
                f.write(f"Question Type: {row['Question_Type']}\n")
                f.write(f"Answer Type: {row['Answer_Type']}\n\n")
            
            f.write("\nSample Textual Questions:\n")
            textual_samples = relevant_data[relevant_data['Classification'] == 'Textual'].head(5)
            for _, row in textual_samples.iterrows():
                f.write(f"Q: {row['Question']}\n")
                f.write(f"A: {row['Answer']}\n")
                f.write(f"Question Type: {row['Question_Type']}\n")
                f.write(f"Answer Type: {row['Answer_Type']}\n\n")
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Total questions processed: {total_questions}")
        logger.info(f"Questions with numerical phrases: {numerical_questions} ({numerical_questions/total_questions*100:.2f}%)")
        logger.info(f"Questions with numerical answers: {numerical_answers} ({numerical_answers/total_questions*100:.2f}%)")
        logger.info(f"Final numerical classification: {numerical_total} ({numerical_total/total_questions*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

def main():
    """Main function to run the question classifier."""
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