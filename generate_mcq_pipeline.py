import os
from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
import sys

# Load environment variables from .env file
load_dotenv()

# Set environment variables directly with corrected values
os.environ["AZURE_OPENAI_API_KEY"] = "P5IXChjIfa0qAoEKQ6Iv3HMXfenSwkyu2jhfJHl4ZPsn0OrYYYuuJQQJ99BBACYeBjFXJ3w3AAABACOGuSlj"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gpt-wm-yl.openai.azure.com/"

# Set the deployment name based on test results
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "o1"  # This deployment works with multiple API versions

# Get environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Validate environment variables
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_DEPLOYMENT_NAME:
    print("Error: Missing required Azure OpenAI environment variables.")
    print("Please check the environment variables in the script.")
    sys.exit(1)

# Create output directory
OUTPUT_DIR = "Numerics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")

# Use the correct input file
INPUT_FILE = "Top-56-Questions-Anwsers.xlsx"

# Initialize Azure OpenAI client with the working API version
try:
    print(f"Connecting to Azure OpenAI service at {AZURE_OPENAI_ENDPOINT}")
    print(f"Using deployment: {AZURE_DEPLOYMENT_NAME}")
    
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview",  # Updated to required version for o1 deployment
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {str(e)}")
    sys.exit(1)

PROMPT_TEMPLATE = '''You are an expert AI medical educator trained to craft USMLE Step 2+ level multiple-choice questions. I will provide you with a long-form Q&A about **Type 2 Diabetes Mellitus (T2DM)**.

Your task is to convert it into a **single, extremely difficult MCQ** that evaluates multi-step clinical reasoning, real-world complexity, and subtle decision-making — not factual recall.

---

🎯 GOAL

Your question should challenge top-tier test-takers or large language models (LLMs) like GPT-4. It must integrate labs, comorbidities, patient preferences, and ADA guideline knowledge — while avoiding all superficial pattern-matching.

---

🧱 STRUCTURE REQUIREMENTS

- **Exactly 10 answer options:**
  - ✅ 1 correct answer (subtle, guideline-supported, trial-aligned)
  - ❌ 8 distractors (plausible, contextually flawed, or partially correct)
  - ❓ 1 "Insufficient information to determine" option (always Option J — sounds cautious but is incorrect)

---

📐 QUESTION STEM VARIATION (NO REPETITION!)

Do **NOT** start every question with something like "A 64-year-old man presents with..."

Vary setting, voice, and framing across different questions.

Here are examples of **acceptable stem formats**:

- **Narrative Style (First-Person/Provider-Centered):**  
  "You are consulted by a cardiologist about a patient with rising A1c and borderline renal function. She wants to avoid injectable therapy. What's the most appropriate intervention?"

- **Time-Based Framing:**  
  "Over the past 12 months, a patient with type 2 diabetes has progressed from an A1c of 6.9% to 8.3%. eGFR declined from 72 to 59. He is adamant about avoiding weight gain. What intervention is best supported by trial data?"

- **Indirect Symptom/Complication-Based Framing:**  
  "A patient with T2DM is referred due to new-onset peripheral edema after starting a recent medication. Which change is most appropriate based on comorbidity risks?"

- **Decision-Based Framing:**  
  "Which of the following therapeutic adjustments is LEAST appropriate in a patient with stable diabetes, recent CHF exacerbation, and declining eGFR?"

- **Pre-Procedure or Incidental Discovery:**  
  "During pre-operative clearance, a 59-year-old man is found to have an A1c of 8.1% despite monotherapy. His GFR is 58. He refuses insulin or any injectable medication. What is the most appropriate next step?"

- **Caregiver/Family Perspective:**  
  "The daughter of a 67-year-old woman reports that her mother has become forgetful and often misses her oral diabetes medications. She also notes significant weight loss over the past 3 months. What should be prioritized in her management?"

- **Telehealth / Access Issue Context:**  
  "A 61-year-old man follows up via telehealth. He has insurance denials for GLP-1 receptor agonists and refuses insulin. His A1c is rising, and eGFR has dropped from 65 to 59. What is the best option considering cost constraints?"

✅ Mix:
- Age (40s–80s)
- Gender
- Care setting (clinic, hospital, ED, telehealth)
- Narrative style (case review, consult note, family report)

---

🧠 COGNITIVE TRAPS — INCLUDE ALL OF THESE:

1. **Second-best trap:** Almost correct, but fails due to subtle contraindication or trial nuance  
2. **Verbal authority bias:** Incorrect answer with confident clinical tone (e.g., "Initiate insulin per GLAZE-3 protocol")  
3. **Hallucination trap:** Fictional but realistic-sounding trial, drug, or protocol (e.g., "DEXTER-II," "SUSTAIN-PURE")  
4. **Threshold proximity trap:** Use labs near critical cutoffs to mislead (e.g., A1c = 6.5%, eGFR = 59)  
5. **Comorbidity trap:** Correct answer becomes wrong in light of coexisting disease (e.g., CHF + TZD)  
6. **Patient preference trap:** Suggest an intervention explicitly refused in the stem  
7. **Option J:** "Insufficient information to determine the best course of action." Must be **plausible-sounding but wrong**

---

📚 KNOWLEDGE REQUIREMENTS

- Ground answers in **ADA/EASD guidelines** and real-world trials (e.g., EMPA-REG, ACCORD, UKPDS)
- Assume your reader is medically literate — don't explain basics like "eGFR" or "GLP-1 RA"
- Cite trial logic where relevant, but do not over-explain

---

🧪 OPTIONAL CHALLENGE ENHANCERS (Use at least one):

- A time trend in labs (e.g., A1c ↑ from 6.8% to 8.1%, eGFR ↓ from 70 to 59)
- Prior medication failure, side effect, or intolerance
- Cost/access limitations (e.g., insurance denial of GLP-1 RA)
- Route-of-administration refusal (e.g., declines injectables)
- Cultural or family factors influencing care decisions
- Polypharmacy or drug interactions

---

📤 OUTPUT FORMAT

- **Question:** [Write the MCQ stem here — vary your phrasing. Make it realistic and clinically rich.]
- **Options:**  
  A) ...  
  B) ...  
  C) ...  
  D) ...  
  E) ...  
  F) ...  
  G) ...  
  H) ...  
  I) ...  
  J) Insufficient information to determine  
- **Answer:** [Correct letter only]

**The correct answer is [A/B/C/...].**

[Brief explanation justifying the correct answer.]
'''

def call_openai_gpt(question, answer):
    prompt = PROMPT_TEMPLATE.format(question=question.strip(), answer=answer.strip())
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content
        print("---- GPT RESPONSE ----")
        print(result)
        print("----------------------")
        return result
    except Exception as e:
        print("Error:", e)
        return None

def parse_response_to_components(response):
    try:
        # Print the response for debugging
        print("\nParsing response:")
        
        if not response:
            print("Empty response received")
            return None, None, None
            
        lines = response.strip().split('\n')
        
        # Debug information
        print(f"Number of lines in response: {len(lines)}")
        
        # Extract the question - try multiple patterns
        q_text = None
        
        # Find any line containing "Question" followed by a colon, regardless of formatting
        question_patterns = [
            "- Question:", 
            "**Question:**", 
            "Question:", 
            "Question :", 
            "**Question**:",
            "#### Question:",
            "#### Generated Question:"
        ]

        # Go through all the lines in the response
        for i, line in enumerate(lines):
            # Check if this line contains a question pattern
            if any(pattern in line for pattern in question_patterns):
                # Extract the question text
                for pattern in question_patterns:
                    if pattern in line:
                        q_text = line.split(pattern)[1].strip()
                        break
                        
                # If no pattern matched exactly, but the line contains "Question"
                if not q_text and "question" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        q_text = parts[1].strip()
                        
                # If we found a question, break the loop
                if q_text:
                    print(f"Found question: {q_text}")
                    break

        # If we still don't have a question, look for any line with a question mark
        if not q_text:
            question_candidates = [line for line in lines if "?" in line and len(line) > 20]
            if question_candidates:
                q_text = question_candidates[0].strip()
                print(f"Found question using backup method: {q_text}")

        # If we still don't have a question, give up
        if not q_text:
            print("Failed to extract a question")
            return None, None, None
            
        # Clean the question text - remove Markdown formatting
        if q_text:
            # Remove quotes
            q_text = q_text.replace('"', '')
            q_text = q_text.replace('"', '')
            q_text = q_text.replace('"', '')
            # Remove other common Markdown formatting
            q_text = q_text.replace("*", "")
            q_text = q_text.replace("_", "")
            q_text = q_text.replace("####", "")
            # Remove beginning/ending quotes if present
            q_text = q_text.strip('"\'')
            print(f"Cleaned question: {q_text}")
            
        # Extract options - search for all lines containing options    
        option_lines = []
        
        # Look for various option formats throughout the entire response
        for line in lines:
            line = line.strip()
            
            # Various patterns for option lines - now including A through J
            if any(line.startswith(f"{l})") for l in "ABCDEFGHIJ"):
                option_lines.append(line)
            elif any(line.startswith(f"{l}) ") for l in "ABCDEFGHIJ"):
                option_lines.append(line)
            elif any(line.startswith(f"{l}. ") for l in "ABCDEFGHIJ"):
                option_lines.append(line)
            elif any(line.startswith(f"**{l}**") for l in "ABCDEFGHIJ"):
                option_lines.append(line)
            elif any(line.startswith(f"  {l})") for l in "ABCDEFGHIJ"):
                option_lines.append(line.strip())
            elif any(line.startswith(f"  {l}) ") for l in "ABCDEFGHIJ"):
                option_lines.append(line.strip())
                
        # Remove any duplicate options and ensure we have exactly 10 options
        unique_options = []
        option_letters_seen = set()
        
        for option in option_lines:
            option_letter = option[0].upper()
            if option_letter in "ABCDEFGHIJ" and option_letter not in option_letters_seen:
                # Clean the option - remove Markdown formatting
                cleaned_option = option.replace("**", "").replace("*", "").replace("_", "")
                unique_options.append(cleaned_option)
                option_letters_seen.add(option_letter)
                
        # Verify we found 10 unique options
        if len(unique_options) != 10:
            print(f"Warning: Found {len(unique_options)} unique options instead of 10: {unique_options}")
            # If we found more than 10, just take the first 10
            if len(unique_options) > 10:
                unique_options = unique_options[:10]
        
        print(f"Found options: {unique_options}")
        
        # Extract the answer
        correct_letter = None
        
        # First look for explicit answer lines
        answer_patterns = [
            "- Answer:", 
            "**Answer:**", 
            "Answer:", 
            "Correct Answer:",
            "**Answer**:",
            "#### Answer:"
        ]
        
        for pattern in answer_patterns:
            for line in lines:
                if pattern in line:
                    # Extract the answer letter
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        answer_part = parts[1].strip()
                        # Look for the first occurrence of A through J
                        for letter in "ABCDEFGHIJ":
                            if letter in answer_part:
                                correct_letter = letter
                                print(f"Found answer from pattern '{pattern}': {correct_letter}")
                                break
                        if correct_letter:
                            break
            if correct_letter:
                break
                
        # If we still don't have an answer, look for a line containing "answer" and a letter
        if not correct_letter:
            for line in lines:
                if "answer" in line.lower() or "correct" in line.lower():
                    for letter in "ABCDEFGHIJ":
                        patterns = [
                            f": {letter}",
                            f":{letter}",
                            f" {letter})",
                            f"({letter})"
                        ]
                        for pattern in patterns:
                            if pattern in line:
                                correct_letter = letter
                                print(f"Found answer from general pattern: {correct_letter}")
                                break
                        if correct_letter:
                            break
                if correct_letter:
                    break
        
        # If we still don't have an answer, look at the context
        if not correct_letter and len(lines) > 5:
            last_lines = lines[-5:]  # Look at the last 5 lines
            for line in last_lines:
                for letter in "ABCDEFGHIJ":
                    if letter in line and ("answer" in line.lower() or "correct" in line.lower()):
                        correct_letter = letter
                        print(f"Found answer from context in last lines: {correct_letter}")
                        break
                if correct_letter:
                    break
        
        # If all else fails, look for any line that just has the letter by itself or with minimal context
        if not correct_letter:
            for line in lines:
                line = line.strip()
                if line in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] or line in ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J."]:
                    correct_letter = line[0]
                    print(f"Found answer from standalone letter: {correct_letter}")
                    break
        
        # If we still don't have an answer, use the first option as a last resort
        if not correct_letter and unique_options:
            correct_letter = unique_options[0][0]
            print(f"Using default answer: {correct_letter}")
            
        # Return the extracted components
        if q_text and unique_options and correct_letter:
            return q_text, unique_options, correct_letter
        else:
            print("Failed to extract all required components")
            print(f"Question: {q_text}")
            print(f"Options: {unique_options}")
            print(f"Answer: {correct_letter}")
            return None, None, None
            
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        print(f"Raw response: {response}")
        return None, None, None

def build_mcq_df(df, max_retries=3, retry_delay=2):
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating MCQs"):
        question, answer = row["Question"], row["Answer"]
        # Retry mechanism
        for attempt in range(1, max_retries + 1):
            response = call_openai_gpt(question, answer)
            if response is None:
                print(f"Attempt {attempt} failed: No response.")
                time.sleep(retry_delay)
                continue
            new_q, choices, correct = parse_response_to_components(response)
            if new_q and choices and correct:
                records.append({
                    "ID": idx + 1,
                    "Original Question": question,
                    "Original Answer": answer,
                    "Generated Question": new_q,
                    "Choices": "\n".join(choices),
                    "Answer": correct
                })
                break  # Success: exit retry loop
            else:
                print(f"Attempt {attempt} failed: Could not parse response.")
                time.sleep(retry_delay)
        else:
            # All retries failed
            print(f"Failed to generate question for row {idx + 1}: '{question}'")
        time.sleep(1)  # Rate limit buffer
    return pd.DataFrame(records)

def main():
    try:
        # Read the input Excel file
        df = pd.read_excel(INPUT_FILE)
        print(f"Successfully read {len(df)} questions from {INPUT_FILE}")
        print("Columns in DataFrame:", df.columns.tolist())
        # Check for required columns
        required_columns = ['Question', 'Answer']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}. Columns found: {df.columns.tolist()}")
        print(f"\nProcessing MCQ generation...")
        mcq_df = build_mcq_df(df)
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "MCQs_3_tweaked.xlsx")
        mcq_df.to_excel(OUTPUT_FILE, index=False)
        print(f"Saved {len(mcq_df)} results to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()