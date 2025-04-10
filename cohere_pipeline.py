import pandas as pd
import os
import cohere
import time
from dotenv import load_dotenv
import logging
import re
import tiktoken
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Cohere client - you can get a free API key from https://dashboard.cohere.com/api-keys
# Adding a placeholder API key that will be updated in the code
COHERE_API_KEY = "3uYbaAJGV1KlICfYyZDSO7GxSbSj9a0THR8vhFaY"
client = cohere.Client(COHERE_API_KEY)

# Set the model
model_name = "command"  # This is Cohere's most capable model

# Initialize tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.error(f"Error initializing tokenizer: {str(e)}")
    tokenizer = None

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    if tokenizer is None:
        # Fallback if tokenizer initialization failed
        return len(text) // 4  # Rough estimate
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
    return f"""I have a multiple-choice question (MCQ). Please analyze the question step by step and provide the correct answer with an explanation.

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

def get_cohere_response(prompt: str) -> Tuple[str, str, int]:
    """Get response from Cohere model and count tokens."""
    try:
        # Count input tokens
        input_tokens = count_tokens(prompt)
        
        # Send request to Cohere
        response = client.chat(
            message=prompt,
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
            preamble="You are an expert at answering multiple choice questions with detailed explanations."
        )
        
        # Get the response text
        full_response = response.text.strip()
        
        # Count output tokens
        output_tokens = count_tokens(full_response)
        total_tokens = input_tokens + output_tokens
        
        # Extract the final answer (look for "Final Answer: X" pattern first)
        final_answer_match = re.search(r'Final Answer:\s*([ABCD])', full_response)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
        else:
            # Fallback: take the last letter A, B, C, or D in the response
            letters = [c for c in full_response if c in 'ABCD']
            final_answer = letters[-1] if letters else "ERROR"
        
        logger.info(f"Successfully got response from Cohere. Tokens used: {total_tokens}")
        
        return final_answer, full_response, total_tokens
    except Exception as e:
        logger.error(f"Error getting Cohere response: {str(e)}")
        return "ERROR", f"Error occurred: {str(e)}", 0

def process_question(row: pd.Series) -> Dict:
    """Process a single question with the LLM."""
    question = row['Question']
    options = row['Options']
    correct_answer = row['Correct_Answer']
    
    # Create chain of thought prompt
    prompt = create_chain_of_thought_prompt(question, options)
    
    # Get response from Cohere
    llm_answer, full_response, tokens_used = get_cohere_response(prompt)
    
    # Log the answer for debugging
    logger.info(f"\nQuestion: {question}")
    logger.info(f"Correct Answer: {correct_answer}")
    logger.info(f"Cohere Answer: {llm_answer}")
    logger.info(f"Match: {llm_answer == correct_answer}")
    
    return {
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Cohere_Answer': llm_answer,
        'Full_Response': full_response,
        'Match': llm_answer == correct_answer,
        'Tokens_Used': tokens_used
    }

def main():
    """Main function to run the pipeline."""
    try:
        # Ask for Cohere API key if not already set
        global COHERE_API_KEY
        global client
        
        if COHERE_API_KEY == "please-update-with-your-key":
            api_key = input("Please enter your Cohere API key (get one from https://dashboard.cohere.com/api-keys): ")
            if api_key:
                COHERE_API_KEY = api_key
                client = cohere.Client(COHERE_API_KEY)
                logger.info("Cohere API key updated.")
            else:
                logger.error("No API key provided. Exiting.")
                return
        
        # Read CSV data
        df = read_csv_data('Hard - Sheet1.csv')
        
        # Process each question
        results = []
        total_tokens = 0
        
        for index, row in df.iterrows():
            logger.info(f"Processing question {index + 1}/{len(df)}")
            result = process_question(row)
            results.append(result)
            total_tokens += result['Tokens_Used']
            
            # Rate limiting - add a delay between requests
            time.sleep(1)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate accuracy
        accuracy = results_df['Match'].mean() * 100
        
        # Save results to Excel
        results_df.to_excel('cohere_results_hard.xlsx', index=False)
        
        logger.info(f"\nPipeline completed successfully!")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Total tokens used: {total_tokens}")
        logger.info(f"Average tokens per question: {total_tokens/len(df):.2f}")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 