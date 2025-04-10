import pandas as pd
import os
import google.generativeai as genai
import cohere
import time
from dotenv import load_dotenv
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import tiktoken
import numpy as np
from sklearn.metrics import confusion_matrix
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize API clients
gemini_api_key = os.getenv("GEMINI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Configure Cohere API
co = cohere.Client(cohere_api_key)

# Configure Azure OpenAI API
azure_client = AzureOpenAI(
    api_key=azure_openai_api_key,
    api_version="2025-03-01-preview",
    azure_endpoint=azure_openai_endpoint
)

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    try:
        tokens = len(tokenizer.encode(text))
        return tokens
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

def read_csv_data(file_path: str) -> pd.DataFrame:
    """Read CSV data with questions and answers."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read CSV file with {len(df)} rows")
        
        # Log data structure
        logger.info("\nData Structure Analysis:")
        logger.info(f"Columns: {', '.join(df.columns)}")
        logger.info(f"Total questions: {len(df)}")
        
        # Extract relevant columns
        relevant_data = pd.DataFrame({
            'Question': df['1st Question'],
            'Options': df['1st Choices'],
            'Correct_Answer': df['1st Answer'].apply(lambda x: x[0] if isinstance(x, str) and len(x) > 0 else x)
        })
        
        # Log sample
        logger.info("\nSample Question Analysis:")
        sample_row = relevant_data.iloc[0]
        logger.info(f"Question: {sample_row['Question']}")
        logger.info(f"Options: {sample_row['Options']}")
        logger.info(f"Correct Answer: {sample_row['Correct_Answer']}")
        
        return relevant_data
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        raise

def create_chain_of_thought_prompt(question: str, options: str) -> str:
    """Create a chain-of-thought prompt for the question."""
    prompt = f"""Question: {question}

Options:
{options}

Think through this step-by-step:
1. Analyze the question carefully.
2. Consider each option one by one.
3. Evaluate each option based on your knowledge.
4. Eliminate options that are incorrect.
5. Choose the most accurate answer.

Final Answer: [A, B, C, or D]"""
    return prompt

def extract_answer_letter(text: str) -> str:
    """Extract the answer letter (A, B, C, D) from the text."""
    # First try to find a pattern like "Final Answer: X"
    final_answer_match = re.search(r'Final Answer:\s*([ABCD])', text)
    if final_answer_match:
        return final_answer_match.group(1)
    
    # If not found, look for "Answer: X"
    answer_match = re.search(r'Answer:\s*([ABCD])', text)
    if answer_match:
        return answer_match.group(1)
    
    # If still not found, look for any standalone A, B, C, or D near the end
    lines = text.split('\n')
    for line in reversed(lines[:10]):  # Check last 10 lines
        letter_match = re.search(r'\b([ABCD])\b', line)
        if letter_match:
            return letter_match.group(1)
    
    # As a last resort, find all instances of A, B, C, or D and use the last one
    letters = re.findall(r'\b([ABCD])\b', text)
    if letters:
        return letters[-1]
    
    # If no matching pattern is found, return ERROR
    return "ERROR"

def get_gemini_response(prompt: str) -> Tuple[str, str, int]:
    """Get response from Gemini model and count tokens."""
    try:
        # Count input tokens
        input_tokens = count_tokens(prompt)
        
        # Generate response with Gemini
        response = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 1000}
        )
        
        # Get the response text
        full_response = response.text
        
        # Count output tokens
        output_tokens = count_tokens(full_response)
        total_tokens = input_tokens + output_tokens
        
        # Extract the final answer
        final_answer = extract_answer_letter(full_response)
        
        return final_answer, full_response, total_tokens
    except Exception as e:
        logger.error(f"Error getting Gemini response: {str(e)}")
        return "ERROR", f"Error occurred: {str(e)}", 0

def get_cohere_response(prompt: str) -> Tuple[str, str, int]:
    """Get response from Cohere model and count tokens."""
    try:
        # Count input tokens
        input_tokens = count_tokens(prompt)
        
        # Get response from Cohere
        response = co.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        
        # Get the response text
        full_response = response.generations[0].text
        
        # Count output tokens
        output_tokens = count_tokens(full_response)
        total_tokens = input_tokens + output_tokens
        
        # Extract the final answer
        final_answer = extract_answer_letter(full_response)
        
        return final_answer, full_response, total_tokens
    except Exception as e:
        logger.error(f"Error getting Cohere response: {str(e)}")
        return "ERROR", f"Error occurred: {str(e)}", 0

def get_azure_openai_response(prompt: str) -> Tuple[str, str, int]:
    """Get response from Azure OpenAI model and count tokens."""
    try:
        # Count input tokens
        input_tokens = count_tokens(prompt)
        
        # Create message structure for Azure OpenAI
        messages = [
            {"role": "system", "content": "You are an expert at answering multiple choice questions with detailed explanations."},
            {"role": "user", "content": prompt}
        ]
        
        # Get response from Azure OpenAI
        response = azure_client.chat.completions.create(
            model=azure_openai_deployment,
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Get the response text
        full_response = response.choices[0].message.content
        
        # Count output tokens
        output_tokens = count_tokens(full_response)
        total_tokens = input_tokens + output_tokens
        
        # Extract the final answer
        final_answer = extract_answer_letter(full_response)
        
        return final_answer, full_response, total_tokens
    except Exception as e:
        logger.error(f"Error getting Azure OpenAI response: {str(e)}")
        return "ERROR", f"Error occurred: {str(e)}", 0

def process_question(row: pd.Series) -> Dict:
    """Process a single question with multiple LLM providers."""
    question = row['Question']
    options = row['Options']
    correct_answer = row['Correct_Answer']
    
    # Create chain of thought prompt
    prompt = create_chain_of_thought_prompt(question, options)
    
    # Get response from Gemini
    gemini_answer, gemini_full_response, gemini_tokens = get_gemini_response(prompt)
    
    # Placeholder for Cohere
    cohere_answer = "ERROR"
    cohere_full_response = "API token invalid"
    cohere_tokens = 0
    cohere_match = False
    
    # Get response from Azure OpenAI
    azure_answer, azure_full_response, azure_tokens = get_azure_openai_response(prompt)
    
    # Log the answers for debugging
    logger.info(f"\nQuestion: {question}")
    logger.info(f"Correct Answer: {correct_answer}")
    logger.info(f"Gemini Answer: {gemini_answer}")
    logger.info(f"Azure OpenAI Answer: {azure_answer}")
    logger.info(f"Gemini Match: {gemini_answer == correct_answer}")
    logger.info(f"Azure OpenAI Match: {azure_answer == correct_answer}")
    
    return {
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Gemini_Answer': gemini_answer,
        'Gemini_Full_Response': gemini_full_response,
        'Gemini_Match': gemini_answer == correct_answer,
        'Gemini_Tokens': gemini_tokens,
        'Cohere_Answer': cohere_answer,
        'Cohere_Full_Response': cohere_full_response,
        'Cohere_Match': cohere_match,
        'Cohere_Tokens': cohere_tokens,
        'Azure_OpenAI_Answer': azure_answer,
        'Azure_OpenAI_Full_Response': azure_full_response,
        'Azure_OpenAI_Match': azure_answer == correct_answer,
        'Azure_OpenAI_Tokens': azure_tokens
    }

def create_visualizations(results_df: pd.DataFrame):
    """Create visualizations for the comparison results."""
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 15))
    
    # 1. Accuracy Comparison
    plt.subplot(2, 2, 1)
    accuracy_data = {
        'Gemini': results_df['Gemini_Match'].mean() * 100,
        'Cohere': results_df['Cohere_Match'].mean() * 100,
        'Azure OpenAI': results_df['Azure_OpenAI_Match'].mean() * 100
    }
    plt.bar(accuracy_data.keys(), accuracy_data.values(), color=['blue', 'red', 'green'])
    plt.title('Accuracy Comparison (%)', fontsize=14)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for i, v in enumerate(accuracy_data.values()):
        plt.text(i, v + 2, f"{v:.2f}%", ha='center')
    
    # 2. Token Usage Comparison
    plt.subplot(2, 2, 2)
    token_data = {
        'Gemini': results_df['Gemini_Tokens'].mean(),
        'Cohere': results_df['Cohere_Tokens'].mean(),
        'Azure OpenAI': results_df['Azure_OpenAI_Tokens'].mean()
    }
    plt.bar(token_data.keys(), token_data.values(), color=['blue', 'red', 'green'])
    plt.title('Average Tokens Per Question', fontsize=14)
    plt.ylabel('Tokens')
    for i, v in enumerate(token_data.values()):
        plt.text(i, v + 20, f"{v:.2f}", ha='center')
    
    # 3. Question-by-Question Accuracy
    plt.subplot(2, 1, 2)
    x = np.arange(len(results_df))
    width = 0.25
    plt.bar(x - width, results_df['Gemini_Match'].apply(int), width, label='Gemini', color='blue', alpha=0.7)
    plt.bar(x, results_df['Cohere_Match'].apply(int), width, label='Cohere', color='red', alpha=0.7)
    plt.bar(x + width, results_df['Azure_OpenAI_Match'].apply(int), width, label='Azure OpenAI', color='green', alpha=0.7)
    plt.xlabel('Question Number')
    plt.ylabel('Correct (1) / Incorrect (0)')
    plt.title('Question-by-Question Accuracy Comparison', fontsize=14)
    plt.xticks(x, [i+1 for i in range(len(results_df))], rotation=90 if len(results_df) > 20 else 0)
    plt.legend()
    
    # 4. Confusion Matrices
    plt.figure(figsize=(15, 5))
    models = ['Gemini', 'Cohere', 'Azure_OpenAI']
    display_names = ['Gemini', 'Cohere', 'Azure OpenAI']
    for i, (model, name) in enumerate(zip(models, display_names), 1):
        plt.subplot(1, 3, i)
        try:
            cm = confusion_matrix(results_df['Correct_Answer'], results_df[f'{model}_Answer'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        except Exception as e:
            logger.error(f"Error creating confusion matrix for {model}: {str(e)}")
            plt.text(0.5, 0.5, f"Error: Could not create confusion matrix", 
                    horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Save the visualizations
    plt.savefig('llm_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the comparison pipeline."""
    try:
        # Read the CSV file
        df = read_csv_data('Hard - Sheet1.csv')
        
        # Process each question
        results = []
        for index, row in df.iterrows():
            logger.info(f"\nProcessing question {index + 1}/{len(df)}")
            result = process_question(row)
            results.append(result)
            time.sleep(1)  # Add delay to avoid rate limits
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate and log overall statistics
        logger.info("\nOverall Statistics:")
        for model, display_name in [('Gemini', 'Gemini'), ('Cohere', 'Cohere'), ('Azure_OpenAI', 'Azure OpenAI')]:
            accuracy = results_df[f'{model}_Match'].mean() * 100
            avg_tokens = results_df[f'{model}_Tokens'].mean()
            logger.info(f"{display_name}:")
            logger.info(f"  Accuracy: {accuracy:.2f}%")
            logger.info(f"  Average Tokens: {avg_tokens:.2f}")
        
        # Create visualizations
        create_visualizations(results_df)
        
        # Save results to CSV
        results_df.to_csv('llm_comparison_results.csv', index=False)
        logger.info("\nResults saved to llm_comparison_results.csv")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 