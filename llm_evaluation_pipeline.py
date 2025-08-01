import pandas as pd
from google import genai
import os
from dotenv import load_dotenv
import time
from typing import Dict, List, Tuple
import logging
import tiktoken
import re
import datetime
from together import Together
from pathlib import Path

# Set up logging
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"mistral_logs_{current_time}.txt"

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini client
api_key = os.getenv('TOGETHER_API_KEY')
if not api_key:
    raise ValueError("API key not found in .env file")
client = Together(api_key=api_key)

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
    """Extract the letter from answer text like 'Answer: B)' or just 'A'"""
    if pd.isna(answer_text):
        return ""
    
    # First, check if it's just a single letter (A, B, C, or D)
    if answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        return answer_text
    
    # Use regex to find the pattern "Answer: X)" where X is A, B, C, or D
    match = re.search(r'Answer:\s*([ABCDEFGHI])\)', answer_text)
    if match:
        return match.group(1)
    
    # If still not found, try to find any A, B, C, or D in the text
    match = re.search(r'([ABCDEFGHI])', answer_text)
    if match:
        return match.group(1)
    
    return ""

def read_csv_data(file_path: str) -> pd.DataFrame:
    """Read the CSV or Excel file and return a DataFrame."""
    try:
        # Check file extension and read accordingly
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path)
        elif str(file_path).endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Use the correct columns for this file
        relevant_data = pd.DataFrame({
            'Question': df['Generated Question'],
            'Options': df['Choices'],
            'Correct_Answer': df['Answer']
        })
        
        logger.info(f"Successfully read file with {len(relevant_data)} rows")
        logger.info(f"Columns: {', '.join(relevant_data.columns)}")
        logger.info(f"Sample Question: {relevant_data.iloc[0]['Question']}")
        logger.info(f"Sample Options: {relevant_data.iloc[0]['Options']}")
        logger.info(f"Sample Correct Answer: {relevant_data.iloc[0]['Correct_Answer']}")
        
        return relevant_data
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

def create_chain_of_thought_prompt(question: str, options: str) -> str:
    """Create a prompt for the question using the new concise format."""
    return f'''Answer the question using one of the given choices.
{question}
{options}

Please begin your response with the exact phrase: "The correct answer is ___." 
Replace the blank with one of the following options: A, B, C, D, E, F, G, H, I, or J.
This line should contain only the final answer. 
Then, in a new paragraph, provide a brief explanation justifying why this answer is correct. 
Do not include any introductory phrases, restatements of the question, or additional formatting.'''

def get_model_response(prompt: str, num_runs: int = 3) -> Tuple[str, str, int, List[str], bool]:
    """Get response from LLM with self-consistency checking through multiple runs."""
    try:
        # Ensure num_runs is at least 3
        num_runs = max(num_runs, 3)
        
        # Count input tokens
        input_tokens = count_tokens(prompt)
        
        # Run multiple times to check consistency
        answers = []
        full_responses = []
        total_tokens_list = []
        
        for run in range(num_runs):
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Get the response text
            full_response = response.choices[0].message.content.strip()
            full_responses.append(full_response)
            
            # Count output tokens
            output_tokens = count_tokens(full_response)
            run_total_tokens = input_tokens + output_tokens
            total_tokens_list.append(run_total_tokens)
            
            # Extract the final answer - similar to Azure OpenAI pipeline
            final_answer_match = re.search(r'([ABCDEFGHI])', full_response)
            if final_answer_match:
                final_answer = final_answer_match.group(1)
            else:
                # Fallback: take the last letter A, B, C, or D in the response
                letters = [c for c in full_response if c in 'ABCDEGHI']
                final_answer = letters[0] if letters else "ERROR"
            
            answers.append(final_answer)
            logger.info(f"Run {run+1}: Answer = {final_answer}")
            
            # Add delay between runs
            if run < num_runs - 1:
                time.sleep(1)
                
        # Check consistency
        consistent = all(a == answers[0] for a in answers)
        logger.info(f"Answers across {num_runs} runs: {answers}")
        logger.info(f"Self-consistency: {consistent}")
        
        # Use majority vote for final answer
        from collections import Counter
        most_common_answer = Counter(answers).most_common(1)[0][0]
        
        # Calculate average tokens
        avg_tokens = sum(total_tokens_list) / num_runs
        
        # Combine all responses for the full record
        combined_response = f"SELF-CONSISTENCY RESULTS:\nRuns: {num_runs}\nAnswers: {answers}\nConsistent: {consistent}\nMajority Answer: {most_common_answer}\n\n" + "\n\n===RESPONSE #{1}===\n\n" + full_responses[0]
        
        return most_common_answer, combined_response, avg_tokens, answers, consistent
    except Exception as e:
        logger.error(f"Error getting model response: {str(e)}")
        return "ERROR", f"Error occurred: {str(e)}", 0, ["ERROR"], False

def process_question(row: pd.Series, num_runs: int = 3) -> Dict:
    """Process a single question and get response from model with self-consistency checking."""
    question = row['Question']
    options = row['Options']
    correct_answer = row['Correct_Answer']
    
    # Create chain of thought prompt - using the same name as Azure pipeline
    prompt = create_chain_of_thought_prompt(question, options)
    
    # Get response from LLM with token count and self-consistency checking
    model_answer, model_full_response, model_tokens, all_answers, is_consistent = get_model_response(prompt, num_runs=num_runs)
    
    # Log the answers for debugging
    logger.info(f"\nQuestion: {question}")
    logger.info(f"Correct Answer: {correct_answer}")
    logger.info(f"Model Answer: {model_answer}")
    logger.info(f"Match: {model_answer == correct_answer}")
    
    return {
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Model_Answer': model_answer,
        'Full_Response': model_full_response,
        'Match': model_answer == correct_answer,
        'Tokens_Used': model_tokens,
        'Self_Consistent': 'Yes' if is_consistent else 'No',
        'All_Answers': str(all_answers),
        'Is_Consistent': is_consistent
    }

# Ensure results are saved in the 'TRICKY' folder
RESULTS_DIR = 'Results_Final'
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_difficulty_level(file_path: str, results_dir: str = RESULTS_DIR, num_runs: int = 3) -> Dict:
    """Process all questions for a specific difficulty level."""
    try:
        # Extract difficulty level from filename
        filename = os.path.basename(file_path)
        difficulty = filename.split('.')[0]  # Just take the name before the extension
        logger.info(f"\nProcessing {difficulty} difficulty level from file: {file_path}")
        
        # Read CSV data
        df = read_csv_data(file_path)
        
        # Process each question
        results = []
        total_tokens = 0
        
        for index, row in df.iterrows():
            logger.info(f"Processing {difficulty} question {index + 1}/{len(df)}")
            result = process_question(row, num_runs=num_runs)
            results.append(result)
            total_tokens += result['Tokens_Used']
            
            # Rate limiting - increase to reduce API rate limit errors
            time.sleep(5)  # 5 seconds delay to avoid hitting rate limits
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate accuracy
        accuracy = results_df['Match'].mean() * 100
        
        # Calculate self-consistency percentage
        self_consistency = results_df['Is_Consistent'].mean() * 100
        
        # Save results to Excel in the results directory
        output_file = os.path.join(results_dir, f'deepseek_results_{difficulty.lower()}_with_consistency.xlsx')
        results_df.to_excel(output_file, index=False)
        
        # Save detailed consistency report to text file
        consistency_report = os.path.join(results_dir, f'consistency_report_{difficulty.lower()}.txt')
        with open(consistency_report, 'w', encoding='utf-8') as f:
            f.write(f"Consistency Report for {difficulty} difficulty level\n")
            f.write(f"Total questions: {len(results_df)}\n")
            f.write(f"Overall consistency rate: {self_consistency:.2f}%\n\n")
            
            for i, row in results_df.iterrows():
                f.write(f"Question {i+1}: {row['Question']}\n")
                f.write(f"Correct Answer: {row['Correct_Answer']}\n")
                f.write(f"Model Answers across runs: {row['All_Answers']}\n")
                f.write(f"Final Answer: {row['Model_Answer']}\n")
                f.write(f"Consistent: {row['Is_Consistent']}\n")
                f.write(f"Match with correct answer: {row['Match']}\n\n")
                
        logger.info(f"\n{difficulty} level completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Consistency report saved to: {consistency_report}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Self-Consistency: {self_consistency:.2f}%")
        logger.info(f"Total tokens used: {total_tokens}")
        logger.info(f"Average tokens per question: {total_tokens/len(df):.2f}")
        
        return {
            'difficulty': difficulty,
            'accuracy': accuracy,
            'self_consistency': self_consistency,
            'total_tokens': total_tokens,
            'avg_tokens_per_question': total_tokens/len(df)
        }
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        raise

DATA_FILE = Path.cwd() / "SWEET.xlsx"

def main():
    """Main function to run the pipeline for all difficulty levels."""
    try:
        # Create SYNA_Q directory if it doesn't exist
        results_dir = "SYNA_Q"
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Created/Using results directory: {results_dir}")
        
        # Log script parameters
        logger.info(f"Script started at: {current_time}")
        logger.info(f"Log file: {log_file}")
        
        # Set number of runs for self-consistency checking (ensure at least 3)
        num_runs = max(3, 3)  # Explicitly set to 3 for consistency
        logger.info(f"Running with {num_runs} iterations for self-consistency checking")
        
        # Process the difficulty level
        summary = process_difficulty_level(DATA_FILE, results_dir, num_runs)
        
        # Create and save a summary report
        summary_df = pd.DataFrame([summary])
        summary_file = os.path.join(results_dir, 'deepseek_summary.xlsx')
        summary_df.to_excel(summary_file, index=False)
        
        # Create overall consistency report
        overall_consistency_file = os.path.join(results_dir, 'deepseek_consistency_report_llama.txt')
        with open(overall_consistency_file, 'w', encoding='utf-8') as f:
            f.write(f"OVERALL CONSISTENCY REPORT\n")
            f.write(f"=========================\n")
            f.write(f"Date/Time: {current_time}\n")
            f.write(f"Model: llama 3.3\n")
            f.write(f"Number of consistency runs per question: {num_runs}\n\n")
            
            f.write(f"Summary for {summary['difficulty']} level:\n")
            f.write(f"  Accuracy: {summary['accuracy']:.2f}%\n")
            f.write(f"  Self-Consistency: {summary['self_consistency']:.2f}%\n")
            f.write(f"  Total tokens: {summary['total_tokens']}\n")
            f.write(f"  Avg tokens per question: {summary['avg_tokens_per_question']:.2f}\n\n")
        
        logger.info(f"\nEntire pipeline completed successfully!")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Overall consistency report saved to: {overall_consistency_file}")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 