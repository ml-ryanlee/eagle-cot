"""Evaluate Eagle models on the GSM8K dataset."""
import yaml
import argparse
import torch
import sys
import os
import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from huggingface_hub import login

try:
    from eagle.model.ea_model import EaModel
except ImportError:
    from .eagle.model.ea_model import EaModel

# Try to import config, fallback if not available
try:
    from compact_cot.config import HF_TOKEN, DATASETS_CACHE_DIR, MODELS_CACHE_DIR
except ImportError:
    HF_TOKEN = None
    DATASETS_CACHE_DIR = None
    MODELS_CACHE_DIR = None

# Regex pattern at module level (it's just a constant)
INVALID_ANS = "[invalid]"
ANSWER_PATTERNS = {
    "gsm8k": re.compile(r"#### (\-?[0-9\.\,]+)"),           # GSM8K format
    "boxed": re.compile(r"\\boxed\{(\-?[0-9\.\,]+)\}"),     # LaTeX \boxed{}
    "dollar": re.compile(r"\$\$(\-?[0-9\.\,]+)\$\$"),       # $$number$$
    "answer_is": re.compile(r"[Aa]nswer is:?\s*(\-?[0-9\.\,]+)"),  # "Answer is: X"
    "numbers": re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")
}

def load_config(config_path: str) -> dict:
    """Load experiment config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def extract_value(completion: str, pattern: re.Pattern) -> str:
    """Extract numerical answer from completion (private helper). Used for GSM8K label answer extraction"""
    match = pattern.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return INVALID_ANS

def extract_last_value(completion, pattern: re.Pattern):
    """Extract last entry matching a designated pattern. Used for model generation answer extraction"""
    matches = list(re.finditer(pattern, completion))
    if len(matches) > 0:
        match = matches[-1]
        return match.group().replace(",","")
    else:
        return INVALID_ANS

def save_results(questions: list, predictions: list, labels: list, model_name: str, descriptor: str) -> str:
    """Save questions, predictions, and labels to a JSONL file.
    
    Args:
        questions: List of question strings
        predictions: List of prediction strings
        labels: List of label strings
        model_name: Model name for output filename
        descriptor: Additional descriptor for filename
        
    Returns:
        Path to the saved output file
    """
    # Create output filename based on model name
    model_name_safe = model_name.replace("/", "_")
    output_file = f"results/{model_name_safe}_{descriptor}_predictions.jsonl"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Write to JSONL
    with open(output_file, "w") as f:
        for question, prediction, label in zip(questions, predictions, labels):
            result = {
                "question": question,
                "prediction": prediction,
                "label": label
            }
            f.write(json.dumps(result) + "\n")
    
    return output_file

def format_prompt(question: str, prompt_prefix: str = "", prompt_suffix: str = "") -> str:
    """Format the prompt for the model."""
    return f"{prompt_prefix}{question}{prompt_suffix}"

def model_generate(config: dict):
    """Generate answers using Eagle model."""
    
    # Load Eagle model
    model = EaModel.from_pretrained(
        base_model_path=config["base_model_path"],
        ea_model_path=config["ea_model_path"],
        total_token=config.get("total_token", 60),
        depth=config.get("depth", 7),
        top_k=config.get("top_k", 10),
        threshold=config.get("threshold", 1.0),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=config.get("load_in_4bit", False),
        load_in_8bit=config.get("load_in_8bit", False),
        device_map="auto",
        use_eagle3=config.get("use_eagle3", True),
    )
    model.eval()
    print(f"Loaded Eagle Model with base: {config['base_model_path']} and ea: {config['ea_model_path']}")

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test", cache_dir=DATASETS_CACHE_DIR)
    print(f"Loaded GSM8K test set with {len(dataset)} examples")

    # Get generation parameters
    temperature = config.get("temperature", 0.0)
    top_p = config.get("top_p", 0.0)
    top_k_gen = config.get("top_k_gen", 0)
    max_new_tokens = config.get("max_new_tokens", 512)
    use_eagle_inference = config.get("use_eagle_inference", True)
    
    # Format prompts
    prompt_prefix = config.get("prompt_prefix", "")
    prompt_suffix = config.get("prompt_suffix", "")
    
    predictions = []
    questions = dataset["question"]
    
    print(f"Starting generation with {len(questions)} questions...")
    print(f"Using Eagle inference: {use_eagle_inference}")
    print(f"Generation params - temperature: {temperature}, top_p: {top_p}, max_new_tokens: {max_new_tokens}")
    
    for i, question in enumerate(questions):
        if i % 50 == 0:
            print(f"Processing question {i+1}/{len(questions)}")
            
        # Format the prompt
        prompt = format_prompt(question, prompt_prefix, prompt_suffix)
        
        # Tokenize
        input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Generate with Eagle
        try:
            if use_eagle_inference:
                # Use Eagle's accelerated generation
                generated_ids = model.ea_generate(
                    input_ids,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    is_llama3=config.get("is_llama3", False)
                )
                # ea_generate is a generator, get the final result
                for output_ids in generated_ids:
                    final_output_ids = output_ids
            else:
                # Use regular generation
                generated_ids = model.naive_generate(
                    input_ids,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    is_llama3=config.get("is_llama3", False)
                )
                # naive_generate is also a generator, get the final result
                for output_ids in generated_ids:
                    final_output_ids = output_ids
            
            # Extract only the new tokens
            input_length = input_ids.shape[1]
            generated_tokens = final_output_ids[0, input_length:]
            
            # Decode the generation
            generated_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            predictions.append(generated_text)
            
        except Exception as e:
            print(f"Error generating for question {i}: {e}")
            predictions.append("")
    
    # Get labels
    labels = dataset["answer"]
    
    return questions, predictions, labels

def main():
    # Huggingface login
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Eagle models on GSM8K")
    parser.add_argument('--config', type=str, help="Path to config YAML")
    parser.add_argument('--base-model-path', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        help="Base model path")
    parser.add_argument('--ea-model-path', type=str, default="yuhuili/EAGLE-LLaMA3-Instruct-8B",
                        help="Eagle model path")
    parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('--top-p', type=float, default=0.0, help="Top-p sampling")
    parser.add_argument('--max-new-tokens', type=int, default=512, help="Max new tokens to generate")
    parser.add_argument('--use-eagle3', action='store_true', default=True, help="Use Eagle-3 inference")
    parser.add_argument('--use-eagle-inference', action='store_true', default=True, 
                        help="Use Eagle accelerated inference")
    parser.add_argument('--save-results', action='store_true', help="Save detailed results")
    parser.add_argument('--prompt-prefix', type=str, default="", help="Prefix for prompts")
    parser.add_argument('--prompt-suffix', type=str, default="", help="Suffix for prompts")
    parser.add_argument('--answer-pattern', type=str, default="numbers", 
                        choices=list(ANSWER_PATTERNS.keys()), help="Pattern to extract answers")
    
    args = parser.parse_args()

    # Load config if provided, otherwise use command line args
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "base_model_path": args.base_model_path,
            "ea_model_path": args.ea_model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "use_eagle3": args.use_eagle3,
            "use_eagle_inference": args.use_eagle_inference,
            "save_results": args.save_results,
            "prompt_prefix": args.prompt_prefix,
            "prompt_suffix": args.prompt_suffix,
            "answer_pattern": args.answer_pattern,
            "total_token": 60,
            "depth": 7,
            "top_k": 10,
            "threshold": 1.0,
            "is_llama3": "llama-3" in args.base_model_path.lower()
        }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Generate answers and extract labels
    questions, predictions, labels = model_generate(config)
    
    # Save results if requested
    if config.get("save_results", False):
        model_name = config["ea_model_path"].split("/")[-1]
        output_file = save_results(questions, predictions, labels, model_name, "eagle_gsm8k")
        print(f"Results saved to {output_file}")

    # Extract answers and report score
    pred_pattern = ANSWER_PATTERNS[config.get("answer_pattern", "numbers")]
    pred_answers = [extract_last_value(p, pred_pattern) for p in predictions]
    true_answers = [extract_value(l, ANSWER_PATTERNS['gsm8k']) for l in labels]
    
    # Calculate metrics
    correct = sum(pred == true for pred, true in zip(pred_answers, true_answers))
    accuracy = correct / len(true_answers)
    
    print(f"\nResults:")
    print(f"Total questions: {len(true_answers)}")
    print(f"Correct answers: {correct}")
    print(f"Eagle GSM8K accuracy: {accuracy:.2%}")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"\nQuestion {i+1}: {questions[i]}")
        print(f"Generated: {predictions[i][:200]}...")
        print(f"Predicted answer: {pred_answers[i]}")
        print(f"True answer: {true_answers[i]}")
        print(f"Correct: {'✓' if pred_answers[i] == true_answers[i] else '✗'}")

if __name__ == "__main__":
    main()