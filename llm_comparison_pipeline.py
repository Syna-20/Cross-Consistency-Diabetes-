import pandas as pd
from google import genai
import os
from dotenv import load_dotenv
import time
from typing import Dict, List, Tuple
import logging
import tiktoken
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini client
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Gemini API key not found in .env file")
client = genai.Client(api_key=api_key)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def extract_answer_letter(answer_text: str) -> str:
    """Extract the letter from answer text like 'Answer: B) ...'"""
    if pd.isna(answer_text):
        return ""
    # Use regex to find the pattern "Answer: X)" where X is A, B, C, or D
    match = re.search(r'Answer:\s*([ABCD])\)', answer_text)
    if match:
        return match.group(1)
    return ""

def read_csv_data(file_path: str) -> pd.DataFrame:
    """Read the CSV file and return a DataFrame."""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Extract relevant columns
        relevant_data = pd.DataFrame({
            'Question': df['1st Question'],
            'Options': df['1st Choices'],
            'Correct_Answer': df['1st Answer'].apply(extract_answer_letter)
        })
        
        logger.info(f"Successfully read CSV file with {len(relevant_data)} rows")
        
        # Analyze data structure
        logger.info("\nData Structure Analysis:")
        logger.info(f"Columns: {', '.join(relevant_data.columns)}")
        logger.info(f"Total questions: {len(relevant_data)}")
        
        # Sample analysis
        sample_question = relevant_data.iloc[0]
        logger.info("\nSample Question Analysis:")
        logger.info(f"Question: {sample_question['Question']}")
        logger.info(f"Options: {sample_question['Options']}")
        logger.info(f"Correct Answer: {sample_question['Correct_Answer']}")
        
        return relevant_data
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise

def create_chain_of_thought_prompt(question: str, options: str) -> str:
    """Create a chain of thought prompt for the question."""
    return f"""I have a multiple-choice question (MCQ) related to Type 2 Diabetes. Please analyze the question step by step and provide the correct answer with an explanation.

Read the question carefully.
Understand the key concepts involved.
Evaluate each answer choice systematically.
Eliminate incorrect options with reasoning.
Select the best answer and justify why it is correct.

Here's the question:
{question}

Options:
{options}

Please provide the correct answer along with a step-by-step explanation. Make sure to clearly state your final answer in the format 'Final Answer: X' where X is A, B, C, or D."""

def get_gemini_response(prompt: str) -> Tuple[str, str, int]:
    """Get response from Gemini model and count tokens."""
    try:
        # Count input tokens
        input_tokens = count_tokens(prompt)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Count output tokens
        output_tokens = count_tokens(response.text)
        
        # Extract the final answer (look for "Final Answer: X" pattern first)
        full_response = response.text.strip()
        final_answer_match = re.search(r'Final Answer:\s*([ABCD])', full_response)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
        else:
            # Fallback: take the last letter A, B, C, or D in the response
            final_answer = ''.join(filter(lambda x: x in 'ABCD', full_response))[-1] if any(x in 'ABCD' for x in full_response) else "ERROR"
        
        return final_answer, full_response, input_tokens + output_tokens
    except Exception as e:
        logger.error(f"Error getting Gemini response: {str(e)}")
        return "ERROR", "Error occurred", 0

def process_question(row: pd.Series) -> Dict:
    """Process a single question and get response from Gemini."""
    question = row['Question']
    options = row['Options']
    correct_answer = row['Correct_Answer']
    
    # Create chain of thought prompt
    prompt = create_chain_of_thought_prompt(question, options)
    
    # Get response from Gemini with token count
    gemini_answer, gemini_full_response, gemini_tokens = get_gemini_response(prompt)
    
    # Log the answers for debugging
    logger.info(f"\nQuestion: {question}")
    logger.info(f"Correct Answer: {correct_answer}")
    logger.info(f"Gemini Answer: {gemini_answer}")
    logger.info(f"Match: {gemini_answer == correct_answer}")
    
    return {
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Gemini_Answer': gemini_answer,
        'Gemini_Full_Response': gemini_full_response,
        'Gemini_Match': gemini_answer == correct_answer,
        'Gemini_Tokens': gemini_tokens
    }

def main():
    """Main function to run the pipeline."""
    try:
        # Read CSV data
        df = read_csv_data('Hard - Sheet1.csv')
        
        # Process each question
        results = []
        total_gemini_tokens = 0
        
        for index, row in df.iterrows():
            logger.info(f"Processing question {index + 1}/{len(df)}")
            result = process_question(row)
            results.append(result)
            total_gemini_tokens += result['Gemini_Tokens']
            time.sleep(2)  # Rate limiting between questions
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        gemini_accuracy = results_df['Gemini_Match'].mean() * 100
        
        # Save results to Excel
        results_df.to_excel('gemini_results_hard.xlsx', index=False)
        
        logger.info(f"\nPipeline completed successfully!")
        logger.info(f"Gemini accuracy: {gemini_accuracy:.2f}%")
        logger.info(f"\nToken Usage:")
        logger.info(f"Total Gemini tokens: {total_gemini_tokens}")
        logger.info(f"Average tokens per question: {total_gemini_tokens/len(df):.2f}")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 