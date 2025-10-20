import argparse
import json
import os
import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool
import random
import chess
from tqdm import tqdm
import pdb


def load_tasks(dataset_root: Path, max_tasks: Optional[int] = None, n_samples_per_task: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load tasks from all JSONL files in dataset root directory with optional sampling per task type."""
    tasks = []

    # Find all JSONL files in the dataset root
    jsonl_files = list(dataset_root.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {dataset_root}")

    print(f"Found {len(jsonl_files)} JSONL files:")
    for file in sorted(jsonl_files):
        print(f"  - {file.name}")

    # Load tasks from all JSONL files
    for file_path in sorted(jsonl_files):
        print(f"Loading {file_path.name}...")
        file_tasks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    file_tasks.append(json.loads(line.strip()))
        print(f"  Loaded {len(file_tasks)} tasks")
        tasks.extend(file_tasks)

    print(f"Total tasks loaded: {len(tasks)}")

    if n_samples_per_task:
        # Group tasks by task_type
        tasks_by_type = {}
        for task in tasks:
            task_type = task.get('task_type', 'unknown')
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(task)

        # Sample n_samples_per_task from each type
        sampled_tasks = []
        for task_type, type_tasks in tasks_by_type.items():
            if len(type_tasks) > n_samples_per_task:
                # Use random sampling with a fixed seed for reproducibility
                random.shuffle(type_tasks)
                sampled = type_tasks[:n_samples_per_task]
            else:
                sampled = type_tasks
            sampled_tasks.extend(sampled)
            print(f"Task type '{task_type}': {len(sampled)}/{len(type_tasks)} tasks selected")

        tasks = sampled_tasks

    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]

    return tasks


def get_context(fen: str) -> str:
    """Generate chess context from FEN.

    Some dataset entries append move hints after a pipe ("|") like:
    "<FEN> | e2e4 e7e5". Strip that part before parsing.
    """
    fen_clean = fen.split('|', 1)[0].strip()
    try:
        board = chess.Board(fen_clean)
    except Exception:
        # Fallback: try to coerce whitespace and retry; otherwise return minimal context
        try:
            board = chess.Board(" ".join(fen_clean.split()))
        except Exception:
            return ""

    # Piece arrangement
    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
            key = f"{color} {names[piece.piece_type]}"
            if key not in pieces:
                pieces[key] = []
            pieces[key].append(chess.square_name(square))

    arrangement = ", ".join(f"{k}: {v}" for k, v in pieces.items())
    legal_moves = ", ".join(sorted(move.uci() for move in board.legal_moves))

    return f"Piece arrangement: {arrangement}\nLegal moves: {legal_moves}\n\n"


def format_prompt(task: Dict[str, Any], add_context: bool = False, format_example_group: int = 1) -> str:
    """Format task into prompt."""
    question = task['question']

    if add_context and 'input' in task:
        context = get_context(task['input'])
        question = question.replace('CONTEXT_PLACEHOLDER', context)
    else:
        question = question.replace('CONTEXT_PLACEHOLDER', '')

    if 'format_examples' in task and task['format_examples']:
        examples_list = task['format_examples']
        if len(examples_list) >= 2:
            # Select specific example based on group (1 or 2)
            if format_example_group == 2 and len(examples_list) >= 2:
                example = examples_list[1]
            else:
                example = examples_list[0]
        else:
            # Fallback to first/only example
            example = examples_list[0] if examples_list else ""
        question = question.replace('FORMAT_EXAMPLE_PLACEHOLDER', example)

    return question


def extract_answer(response: str) -> Tuple[str, bool]:
    """Extract final answer and return whether extraction was successful."""
    # Look for the last occurrence of FINAL ANSWER: in the response
    matches = list(re.finditer(r'FINAL ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL))
    if matches:
        # Take the last match and extract only the answer part (group 1)
        answer = matches[-1].group(1).strip()
        # Remove any leading "FINAL ANSWER:" if it got captured
        answer = re.sub(r'^FINAL ANSWER:\s*', '', answer, flags=re.IGNORECASE).strip()
        # Strip markdown formatting (**, *, etc.)
        answer = re.sub(r'^\*+|\*+$', '', answer).strip()
        # Strip any remaining whitespace including newlines
        answer = answer.strip()
        return answer, True

    # Try to extract from "The final answer is $\boxed{...}$" format
    boxed_matches = list(re.finditer(r'[Tt]he\s+final\s+answer\s+is\s+\$?\\boxed\{([^}]+)\}\$?', response))
    if boxed_matches:
        # Take the last match and extract the content inside \boxed{}
        answer = boxed_matches[-1].group(1).strip()
        return answer, True

    return "", False


def calculate_total_usage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate total usage statistics from results."""
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    error_count = 0
    extraction_success_count = 0

    for result in results:
        usage = result.get('inference', {}).get('usage', {})
        if usage:
            total_cost += usage.get('cost', 0.0)
            total_prompt_tokens += usage.get('prompt_tokens', 0)
            total_completion_tokens += usage.get('completion_tokens', 0)
            total_tokens += usage.get('total_tokens', 0)
        else:
            error_count += 1

        # Track extraction success
        if result.get('inference', {}).get('extraction_successful', False):
            extraction_success_count += 1

    return {
        'total_cost': total_cost,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_tokens': total_tokens,
        'error_count': error_count,
        'extraction_success_count': extraction_success_count,
        'extraction_success_rate': extraction_success_count / len(results) if results else 0.0,
        'avg_cost_per_task': total_cost / len(results) if results else 0.0,
        'avg_tokens_per_task': total_tokens / len(results) if results else 0.0,
        'cost_per_1k_tokens': (total_cost / (total_tokens / 1000)) if total_tokens > 0 else 0.0,
        'tokens_per_dollar': (total_tokens / total_cost) if total_cost > 0 else 0.0,
        'avg_prompt_tokens': total_prompt_tokens / len(results) if results else 0.0,
        'avg_completion_tokens': total_completion_tokens / len(results) if results else 0.0,
        'completion_ratio': (total_completion_tokens / total_prompt_tokens) if total_prompt_tokens > 0 else 0.0
    }


def evaluate_answer_with_error_type(extracted: str, correct_answer: str, answer_type: str, extraction_successful: bool, usage: Dict[str, Any], max_tokens: int) -> Tuple[bool, str]:
    """
    Evaluate extracted answer and return (is_correct, error_type).

    Error types:
    - "correct": Answer is correct
    - "max_token_reached": Response was truncated due to max token limit
    - "format_error": Could not extract answer from response (format mismatch)
    - "wrong_answer": Answer was extracted but incorrect
    - "multi_extra_items": Multi-answer has extra incorrect items
    - "multi_missing_items": Multi-answer is missing required items
    - "multi_false_items": Multi-answer has false items instead of correct ones
    """

    # Check for max token reached first (highest priority)
    completion_tokens = usage.get('completion_tokens', 0)
    if completion_tokens >= max_tokens * 0.98:  # 98% threshold to account for slight variations
        return False, "max_token_reached"

    # Check for format error
    if not extraction_successful:
        return False, "format_error"

    # Evaluate answer correctness
    if answer_type == "single":
        is_correct = extracted.lower().strip() == correct_answer.lower().strip()
        if is_correct:
            return True, "correct"
        else:
            return False, "wrong_answer"

    elif answer_type == "multi":
        # Parse comma-separated values
        extracted_set = set(item.strip().lower() for item in extracted.split(',') if item.strip())
        correct_set = set(item.strip().lower() for item in correct_answer.split(',') if item.strip())

        if extracted_set == correct_set:
            return True, "correct"

        # Determine specific multi-answer error type
        extra_items = extracted_set - correct_set
        missing_items = correct_set - extracted_set

        if extra_items and not missing_items:
            return False, "multi_extra_items"
        elif missing_items and not extra_items:
            return False, "multi_missing_items"
        else:
            # Has both extra and missing items, or completely wrong items
            return False, "multi_false_items"

    else:
        # Fallback for unknown answer types
        is_correct = extracted.lower().strip() == correct_answer.lower().strip()
        if is_correct:
            return True, "correct"
        else:
            return False, "wrong_answer"


def process_single_task(args_tuple):
    """Process a single task - for multiprocessing."""
    task, model, add_context, format_example_group, api_key, max_retries, timeout, max_tokens, enable_thinking = args_tuple

    # Create inferencer instance for this process
    # Filename suffix is handled by the parent inferencer when saving; child only calls the API
    inferencer = OpenrouterInferencer(model, add_context, max_retries, timeout, max_tokens, enable_thinking)
    inferencer.api_key = api_key

    prompt = format_prompt(task, add_context, format_example_group)
    response, thinking_content, usage = inferencer.call_model(prompt)
    extracted, extraction_successful = extract_answer(response)

    # Use answer_type-aware evaluation with error type classification
    answer_type = task.get('answer_type', 'single')
    correct, error_type = evaluate_answer_with_error_type(
        extracted, task['correct_answer'], answer_type, extraction_successful, usage, max_tokens
    )

    # Include all original task fields plus inference information
    result = dict(task)  # Copy all original fields
    result['inference'] = {
        'prompt': prompt,
        'response': response,
        'thinking_content': thinking_content,
        'extracted': extracted,
        'extraction_successful': extraction_successful,
        'is_correct': correct,
        'error_type': error_type,
        'usage': usage
    }
    return result


def _build_variant_suffix(add_context: bool, format_example_group: int) -> str:
    """Return a short suffix for filenames to distinguish experiment variants."""
    parts = []
    if add_context:
        parts.append("piecearr")
    if format_example_group == 2:
        parts.append("fmt2")
    return ("-" + "-".join(parts)) if parts else ""

class OpenrouterInferencer:
    def __init__(self, model: str, add_context: bool = False, max_retries: int = 5, timeout: int = 60, max_tokens: int = 2048, enable_thinking: bool = False, filename_suffix: str = ""):
        self.model = model
        self.add_context = add_context
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        # Additional filename suffix to differentiate experiment variants in outputs
        self.filename_suffix = filename_suffix
        keys_path = Path(__file__).parent.parent.parent / "keys" / "api_keys.json"
        with open(keys_path, 'r') as f:
            keys = json.load(f)
        self.api_key = keys.get("openrouter_api_key")
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def call_model(self, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """Call model with exponential retry. Returns (content, thinking_content, usage)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "usage": {
                    "include": True
                }
        }

        if self.enable_thinking:
            data["reasoning"] = {"effort": "medium"}
        
        if self.model == "qwen/qwen3-next-80b-a3b-thinking":
            data["provider"] = {
                                'order': [
                                    'google-vertex',
                                    'together',
                                ]
                                }
        
        if self.model == "deepseek/deepseek-chat-v3.1":
            data["provider"] = {
                                'order': [
                                    'fireworks',
                                ]
                                }
        
        if self.model == "deepseek/deepseek-r1-0528":
            data["provider"] = {
                                'order': [
                                    'google-vertex',
                                ]
                                }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout,
                    stream=False
                )
                response.raise_for_status()
                result = response.json()
                # Extract main content
                content = result['choices'][0]['message']['content'].strip()

                # Extract thinking/reasoning content
                thinking_content = ""
                message = result['choices'][0]['message']

                # Try different reasoning extraction methods
                if hasattr(message, 'reasoning') and message.get('reasoning'):
                    thinking_content = message['reasoning']
                elif 'reasoning_details' in message and message['reasoning_details']:
                    reasoning_parts = []
                    for detail in message['reasoning_details']:
                        if isinstance(detail, dict) and 'text' in detail:
                            reasoning_parts.append(detail['text'])
                        elif isinstance(detail, str):
                            reasoning_parts.append(detail)
                    thinking_content = '\n'.join(reasoning_parts)
                elif 'reasoning' in message and message['reasoning']:
                    thinking_content = str(message['reasoning'])

                usage = result.get('usage', {})
                return content, thinking_content, usage

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"ERROR: {e}", "", {}
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)


    def run_inference(self, tasks: List[Dict[str, Any]], num_workers: int = 1, format_example_group: int = 1, output_path: Optional[str] = None, save_interval: int = 10, existing_results: List[Dict[str, Any]] = None, save_existing_first: bool = False) -> List[Dict[str, Any]]:
        """Run inference on tasks with optional multiprocessing."""
        if existing_results is None:
            existing_results = []

        # Save existing results first if we're resuming
        if save_existing_first and existing_results and output_path:
            self._save_existing_results(existing_results, output_path)

        if num_workers == 1:
            # Sequential processing with progress bar and incremental saving
            results = []
            for i, task in enumerate(tqdm(tasks, desc="Processing tasks")):
                prompt = format_prompt(task, self.add_context, format_example_group)
                response, thinking_content, usage = self.call_model(prompt)
                extracted, extraction_successful = extract_answer(response)

                # Use answer_type-aware evaluation with error type classification
                answer_type = task.get('answer_type', 'single')
                correct, error_type = evaluate_answer_with_error_type(
                    extracted, task['correct_answer'], answer_type, extraction_successful, usage, self.max_tokens
                )

                # Include all original task fields plus inference information
                result = dict(task)  # Copy all original fields
                result['inference'] = {
                    'prompt': prompt,
                    'response': response,
                    'thinking_content': thinking_content,
                    'extracted': extracted,
                    'extraction_successful': extraction_successful,
                    'is_correct': correct,
                    'error_type': error_type,
                    'usage': usage
                }
                results.append(result)

                # Save every save_interval responses
                if output_path and (i + 1) % save_interval == 0:
                    # Save only the last save_interval new results
                    start_idx = max(0, len(results) - save_interval)
                    new_batch = results[start_idx:]
                    self._save_incremental_results(new_batch, output_path)

            # Save any remaining results that weren't saved in the last interval
            if output_path and len(results) % save_interval != 0:
                remaining_count = len(results) % save_interval
                remaining_results = results[-remaining_count:]
                self._save_incremental_results(remaining_results, output_path)

            return results
        else:
            # Parallel processing with progress bar and incremental saving
            print(f"Processing {len(tasks)} tasks with {num_workers} workers...")

            # Prepare arguments for multiprocessing
            args_list = [(task, self.model, self.add_context, format_example_group, self.api_key,
                         self.max_retries, self.timeout, self.max_tokens, self.enable_thinking) for task in tasks]

            with Pool(num_workers) as pool:
                # Use imap for progress tracking
                results = []
                for i, result in enumerate(tqdm(pool.imap(process_single_task, args_list),
                                              total=len(args_list),
                                              desc="Processing tasks")):
                    results.append(result)

                    # Save every save_interval responses
                    if output_path and (i + 1) % save_interval == 0:
                        # For parallel processing, we need to maintain order first
                        temp_task_id_to_result = {r['task_id']: r for r in results}
                        temp_ordered_results = []
                        for j in range(i + 1):
                            if tasks[j]['task_id'] in temp_task_id_to_result:
                                temp_ordered_results.append(temp_task_id_to_result[tasks[j]['task_id']])
                        if len(temp_ordered_results) == i + 1:  # All results up to this point are available
                            # Save only the last save_interval new results
                            start_idx = max(0, len(temp_ordered_results) - save_interval)
                            new_batch = temp_ordered_results[start_idx:]
                            self._save_incremental_results(new_batch, output_path)

            # Save any remaining results that weren't saved in the last interval
            if output_path and len(results) % save_interval != 0:
                remaining_count = len(results) % save_interval
                # For parallel processing, maintain order
                temp_task_id_to_result = {r['task_id']: r for r in results}
                temp_ordered_results = []
                for task in tasks:
                    if task['task_id'] in temp_task_id_to_result:
                        temp_ordered_results.append(temp_task_id_to_result[task['task_id']])

                remaining_results = temp_ordered_results[-remaining_count:]
                self._save_incremental_results(remaining_results, output_path)

            # Maintain original order
            task_id_to_result = {r['task_id']: r for r in results}
            ordered_results = [task_id_to_result[task['task_id']] for task in tasks]

            return ordered_results

    def _save_incremental_results(self, new_results: List[Dict[str, Any]], output_path: str):
        """Append new results to the main results file."""
        try:
            # Get model name from output path and create model-based filename
            output_dir = Path(output_path)
            # Extract model name from the inferencer (need to get it from self.model)
            model_safe_name = self.model.replace('/', '_').replace(':', '_')
            if self.enable_thinking:
                model_safe_name += "-thinking"
            # Add variant suffix if provided (e.g., -piecearr, -fmt2)
            model_safe_name += self.filename_suffix
            results_file = output_dir / f"{model_safe_name}.jsonl"

            # Append new results to JSONL file
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'a') as f:
                for result in new_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            # Calculate current accuracy for progress info
            total = len(new_results)
            correct = sum(r['inference']['is_correct'] for r in new_results)
            accuracy = correct / total if total > 0 else 0.0

            print(f"\nIncremental save: {results_file} (+{total} tasks, {accuracy:.3f} accuracy for new tasks)")

        except Exception as e:
            print(f"\nWarning: Failed to save incremental results: {e}")

    def _save_existing_results(self, existing_results: List[Dict[str, Any]], output_path: str):
        """Save existing complete results to start the file."""
        try:
            # Get model name from output path and create model-based filename
            output_dir = Path(output_path)
            # Extract model name from the inferencer (need to get it from self.model)
            model_safe_name = self.model.replace('/', '_').replace(':', '_')
            if self.enable_thinking:
                model_safe_name += "-thinking"
            # Add variant suffix if provided (e.g., -piecearr, -fmt2)
            model_safe_name += self.filename_suffix
            results_file = output_dir / f"{model_safe_name}.jsonl"

            # Write existing results to file (overwrite)
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                for result in existing_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            print(f"\nSaved {len(existing_results)} existing complete results to {results_file}")

        except Exception as e:
            print(f"\nWarning: Failed to save existing results: {e}")


def load_existing_results(results_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing results and return as dict keyed by task_id."""
    if not results_file.exists():
        return {}

    print(f"Loading existing results from {results_file}")

    existing_results = {}
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line.strip())
                task_id = result.get('task_id')
                if task_id:
                    existing_results[task_id] = result

    print(f"Loaded {len(existing_results)} existing results")
    return existing_results


def re_evaluate_results(results: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    """Re-extract answers and re-evaluate existing results."""
    re_evaluated = []

    for result in results:
        # Make a copy to avoid modifying the original
        new_result = dict(result)

        # Get the response from existing inference
        response = result.get('inference', {}).get('response', '')

        # Re-extract the answer using the updated extraction function
        extracted, extraction_successful = extract_answer(response)

        # Re-evaluate with the correct answer
        answer_type = result.get('answer_type', 'single')
        usage = result.get('inference', {}).get('usage', {})

        correct, error_type = evaluate_answer_with_error_type(
            extracted,
            result['correct_answer'],
            answer_type,
            extraction_successful,
            usage,
            max_tokens
        )

        # Update the inference results
        new_result['inference']['extracted'] = extracted
        new_result['inference']['extraction_successful'] = extraction_successful
        new_result['inference']['is_correct'] = correct
        new_result['inference']['error_type'] = error_type

        re_evaluated.append(new_result)

    return re_evaluated


def filter_incomplete_tasks(tasks: List[Dict[str, Any]], existing_results: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter tasks into incomplete (need inference) and complete (already done)."""
    incomplete_tasks = []
    complete_results = []
    max_token_retry_count = 0

    for task in tasks:
        task_id = task.get('task_id')
        if task_id in existing_results:
            existing_result = existing_results[task_id]
            # Check if inference was completed successfully
            if ('inference' in existing_result and
                'response' in existing_result['inference'] and
                existing_result['inference']['response'] and
                not existing_result['inference']['response'].startswith('ERROR')):

                # Check if this result had max_token_reached error - if so, retry it
                error_type = existing_result['inference'].get('error_type', '')
                if error_type == 'max_token_reached':
                    incomplete_tasks.append(task)
                    max_token_retry_count += 1
                else:
                    complete_results.append(existing_result)
            else:
                incomplete_tasks.append(task)
        else:
            incomplete_tasks.append(task)

    if max_token_retry_count > 0:
        print(f"Found {max_token_retry_count} previous results with 'max_token_reached' error - will retry these tasks")

    return incomplete_tasks, complete_results


def main():
    args = parse_arguments()

    # Set num_workers for metadata (used in eval-only mode too)
    num_workers = args.workers

    # Check for eval-only mode first
    if args.eval_only:
        print("Running in EVAL-ONLY mode - re-evaluating existing results")

        # Get the results file path (include variant suffix to match the run)
        model_safe_name = args.model.replace('/', '_').replace(':', '_')
        if args.enable_thinking:
            model_safe_name += "-thinking"
        model_safe_name += _build_variant_suffix(args.add_context, args.use_format_example_group)
        results_file = args.output_dir / f"{model_safe_name}.jsonl"

        # Check if results file exists
        if not results_file.exists():
            print(f"ERROR: Results file does not exist: {results_file}")
            print("Eval-only mode requires existing results. Please run inference first.")
            return

        # Load existing results
        print(f"Loading results from {results_file}")
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))

        print(f"Loaded {len(results)} results")

        # Re-evaluate all results
        print("Re-extracting answers and re-evaluating...")
        results = re_evaluate_results(results, args.max_tokens)

        # Load tasks to ensure we have complete task information
        print(f"Loading tasks from dataset root: {args.dataset_root}")
        tasks = load_tasks(args.dataset_root, args.max_tasks, args.N_samples_per_task)

        # Make sure task order is maintained
        task_id_to_task = {task['task_id']: task for task in tasks}
        task_id_to_result = {r['task_id']: r for r in results}

        # Ensure all task fields are present in results
        for result in results:
            task_id = result['task_id']
            if task_id in task_id_to_task:
                task = task_id_to_task[task_id]
                # Update any missing fields from original task
                for key, value in task.items():
                    if key not in result:
                        result[key] = value

        # Save the re-evaluated results back to the JSONL file
        print(f"Saving re-evaluated results to {results_file}")
        with open(results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Use dummy timing for eval-only mode
        start_time = time.time()
        end_time = time.time()

    else:
        # Normal mode (with or without resume)
        if num_workers > 1:
            print(f"Using {num_workers} workers for parallel processing")

        # Load and run
        print(f"Loading tasks from dataset root: {args.dataset_root}")
        if args.N_samples_per_task:
            print(f"Sampling {args.N_samples_per_task} tasks per task type")
        tasks = load_tasks(args.dataset_root, args.max_tasks, args.N_samples_per_task)
        print(f"Final task count: {len(tasks)}")

        # Check for resume mode
        if args.no_resume:
            print("Starting inference from scratch (--no-resume)")
            incomplete_tasks = tasks
            complete_results = []
        else:
            print("Checking for existing results to resume from...")
            # Check for existing results and filter incomplete tasks
            model_safe_name = args.model.replace('/', '_').replace(':', '_')
            if args.enable_thinking:
                model_safe_name += "-thinking"
            model_safe_name += _build_variant_suffix(args.add_context, args.use_format_example_group)
            results_file = args.output_dir / f"{model_safe_name}.jsonl"
            existing_results = load_existing_results(results_file)

            incomplete_tasks, complete_results = filter_incomplete_tasks(tasks, existing_results)

            print(f"Tasks already completed: {len(complete_results)}")
            print(f"Tasks needing inference: {len(incomplete_tasks)}")

        if incomplete_tasks:
            # Build a filename suffix based on variant flags so this run doesn't overwrite others
            variant_suffix = _build_variant_suffix(args.add_context, args.use_format_example_group)
            inferencer = OpenrouterInferencer(
                args.model,
                args.add_context,
                args.max_retries,
                args.timeout,
                args.max_tokens,
                args.enable_thinking,
                filename_suffix=variant_suffix,
            )

            start_time = time.time()
            # For resume mode, save existing results first, then append new ones
            # For no-resume mode, just append new results
            save_existing_first = not args.no_resume and len(complete_results) > 0
            new_results = inferencer.run_inference(incomplete_tasks, num_workers, args.use_format_example_group, str(args.output_dir), args.save_interval, complete_results, save_existing_first)
            end_time = time.time()

            # Combine complete and new results, maintaining original task order
            task_id_to_result = {}
            for result in complete_results + new_results:
                task_id_to_result[result['task_id']] = result

            # Maintain original task order
            results = [task_id_to_result[task['task_id']] for task in tasks if task['task_id'] in task_id_to_result]
        else:
            print("All tasks already completed!")
            start_time = time.time()
            results = complete_results
            end_time = time.time()

    # Calculate stats
    total = len(results)
    correct = sum(r['inference']['is_correct'] for r in results)
    accuracy = correct / total

    # Calculate usage statistics
    usage_stats = calculate_total_usage(results)

    # Calculate error type distribution
    error_type_stats = {}
    for result in results:
        error_type = result['inference'].get('error_type', 'unknown')
        if error_type not in error_type_stats:
            error_type_stats[error_type] = 0
        error_type_stats[error_type] += 1

    # Convert to rates
    error_type_rates = {}
    for error_type, count in error_type_stats.items():
        error_type_rates[error_type] = count / total if total > 0 else 0

    # Calculate per-task-type and per-task-category stats
    task_type_stats = {}
    task_category_stats = {}

    for i, result in enumerate(results):
        task_type = result['task_type']
        # Extract task_category from the original task
        task_category = tasks[i].get('task_category', 'unknown')
        usage = result.get('inference', {}).get('usage', {})

        # Task type stats
        if task_type not in task_type_stats:
            task_type_stats[task_type] = {'total': 0, 'correct': 0, 'cost': 0.0, 'tokens': 0, 'extraction_success': 0}
        task_type_stats[task_type]['total'] += 1
        task_type_stats[task_type]['cost'] += usage.get('cost', 0.0)
        task_type_stats[task_type]['tokens'] += usage.get('total_tokens', 0)
        if result['inference']['is_correct']:
            task_type_stats[task_type]['correct'] += 1
        if result['inference']['extraction_successful']:
            task_type_stats[task_type]['extraction_success'] += 1

        # Task category stats
        if task_category not in task_category_stats:
            task_category_stats[task_category] = {'total': 0, 'correct': 0, 'cost': 0.0, 'tokens': 0, 'extraction_success': 0}
        task_category_stats[task_category]['total'] += 1
        task_category_stats[task_category]['cost'] += usage.get('cost', 0.0)
        task_category_stats[task_category]['tokens'] += usage.get('total_tokens', 0)
        if result['inference']['is_correct']:
            task_category_stats[task_category]['correct'] += 1
        if result['inference']['extraction_successful']:
            task_category_stats[task_category]['extraction_success'] += 1

    # Add accuracy, cost, and extraction success calculations to task type stats
    for task_type, stats in task_type_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['extraction_success_rate'] = stats['extraction_success'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['avg_cost'] = stats['cost'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['avg_tokens'] = stats['tokens'] / stats['total'] if stats['total'] > 0 else 0.0

    # Add accuracy, cost, and extraction success calculations to task category stats
    for task_category, stats in task_category_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['extraction_success_rate'] = stats['extraction_success'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['avg_cost'] = stats['cost'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['avg_tokens'] = stats['tokens'] / stats['total'] if stats['total'] > 0 else 0.0

    # Create model-based filenames with variant suffix
    model_safe_name = args.model.replace('/', '_').replace(':', '_')
    if args.enable_thinking:
        model_safe_name += "-thinking"
    model_safe_name += _build_variant_suffix(args.add_context, args.use_format_example_group)
    results_file = args.output_dir / f"{model_safe_name}.jsonl"
    stats_file = args.output_dir / f"{model_safe_name}_stats.json"

    # Save results as JSONL (one result per line)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Also save a pretty-printed JSON version for readability
    pretty_file = args.output_dir / f"{model_safe_name}_pretty.json"
    with open(pretty_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"PRETTY JSON SAVED: {pretty_file}")

    # Save stats as separate JSON file
    with open(stats_file, 'w') as f:
        json.dump({
            'model': args.model,
            'accuracy': accuracy,
            'format_correct_rate': usage_stats['extraction_success_rate'],
            'total': total,
            'correct': correct,
            'format_correct': usage_stats['extraction_success_count'],
            'error_type_stats': error_type_stats,
            'error_type_rates': error_type_rates,
            'time': end_time - start_time,
            'usage': usage_stats,
            'task_type_stats': task_type_stats,
            'task_category_stats': task_category_stats,
            'metadata': {
                'dataset_root': str(args.dataset_root),
                'n_samples_per_task': args.N_samples_per_task,
                'max_tasks': args.max_tasks,
                'add_context': args.add_context,
                'workers': num_workers,
                'format_example_group': args.use_format_example_group,
                'enable_thinking': args.enable_thinking
            }
        }, f, indent=2)


    # Print per-task-category stats with cost and extraction success information
    print(f"\nPER-TASK-CATEGORY PERFORMANCE:")
    print("-" * 120)
    print(f"{'Category':30} {'Accuracy':>10} {'Format Rate':>12} {'Count':>12} {'Cost':>10} {'Avg Cost':>12} {'Tokens':>10}")
    print("-" * 120)
    for task_category, stats in sorted(task_category_stats.items()):
        category_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        extraction_rate = stats['extraction_success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{task_category:30} {category_accuracy:10.3f} {extraction_rate:12.3f} {stats['correct']:>5}/{stats['total']:<5} "
              f"${stats['cost']:>8.4f} ${stats['avg_cost']:>10.4f} {int(stats['tokens']):>10,}")

    # Print per-task-type stats with cost and extraction success information
    print(f"\nPER-TASK-TYPE PERFORMANCE:")
    print("-" * 120)
    print(f"{'Task Type':30} {'Accuracy':>10} {'Format Rate':>12} {'Count':>12} {'Cost':>10} {'Avg Cost':>12} {'Tokens':>10}")
    print("-" * 120)
    for task_type, stats in sorted(task_type_stats.items()):
        type_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        extraction_rate = stats['extraction_success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{task_type:30} {type_accuracy:10.3f} {extraction_rate:12.3f} {stats['correct']:>5}/{stats['total']:<5} "
              f"${stats['cost']:>8.4f} ${stats['avg_cost']:>10.4f} {int(stats['tokens']):>10,}")

    print(f"\nOVERALL ACCURACY: {accuracy:.3f} ({correct}/{total})")
    print(f"FORMAT FOLLOWING RATE: {usage_stats['extraction_success_rate']:.3f} ({usage_stats['extraction_success_count']}/{total})")

    print(f"\nERROR TYPE BREAKDOWN:")
    for error_type, count in sorted(error_type_stats.items()):
        rate = error_type_rates[error_type]
        print(f"  {error_type}: {rate:.3f} ({count}/{total})")

    print(f"\nTIME: {end_time - start_time:.1f}s")
    print(f"\nUSAGE STATISTICS:")
    print(f"Total Cost: ${usage_stats['total_cost']:.4f}")
    print(f"Average Cost per Task: ${usage_stats['avg_cost_per_task']:.4f}")
    print(f"Cost per Correct Answer: ${usage_stats['total_cost'] / correct if correct > 0 else 0:.4f}")
    print(f"Total Tokens: {usage_stats['total_tokens']:,}")
    print(f"  - Prompt Tokens: {usage_stats['total_prompt_tokens']:,}")
    print(f"  - Completion Tokens: {usage_stats['total_completion_tokens']:,}")
    print(f"Average Tokens per Task: {usage_stats['avg_tokens_per_task']:.1f}")
    print(f"Cost per 1K Tokens: ${usage_stats['total_cost'] / (usage_stats['total_tokens'] / 1000) if usage_stats['total_tokens'] > 0 else 0:.4f}")
    print(f"Tokens per Dollar: {usage_stats['total_tokens'] / usage_stats['total_cost'] if usage_stats['total_cost'] > 0 else 0:.0f}")
    if usage_stats['error_count'] > 0:
        print(f"API Errors: {usage_stats['error_count']}")
    print(f"\nRESULTS SAVED: {results_file}")
    print(f"PRETTY JSON SAVED: {pretty_file}")
    print(f"STATS SAVED: {stats_file}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    script_dir = Path(__file__).parent
    default_dataset_root = script_dir.parent.parent / "data" / "benchmark"
    default_output_dir = script_dir.parent.parent / "results"

    parser.add_argument('--dataset-root', type=Path, default=default_dataset_root,
                       help='Root directory containing JSONL task files')
    # parser.add_argument('--model', type=str, required=True,
    #                    help='OpenRouter model ID, e.g., google/gemini-2.5-flash')
    # parser.add_argument('--model', type=str, default='mistralai/mistral-medium-3.1')
    # parser.add_argument('--model', type=str, default='google/gemini-2.5-pro')
    # parser.add_argument('--model', type=str, default='google/gemini-2.5-flash')
    # parser.add_argument('--model', type=str, default='anthropic/claude-3.5-haiku')
    # parser.add_argument('--model', type=str, default='anthropic/claude-haiku-4.5')
    parser.add_argument('--model', type=str, default='anthropic/claude-sonnet-4.5')
    # parser.add_argument('--model', type=str, default='anthropic/claude-sonnet-4')
    # parser.add_argument('--model', type=str, default='deepseek/deepseek-chat-v3.1')
    # parser.add_argument('--model', type=str, default='deepseek/deepseek-r1-0528')
    # parser.add_argument('--model', type=str, default='openai/gpt-5-chat')
    # parser.add_argument('--model', type=str, default='openai/gpt-5')
    # parser.add_argument('--model', type=str, default='qwen/qwen3-next-80b-a3b-instruct')
    # parser.add_argument('--model', type=str, default='qwen/qwen3-next-80b-a3b-thinking')
    # parser.add_argument('--model', type=str, default='qwen/qwen3-max')
    # parser.add_argument('--model', type=str, default='meta-llama/llama-4-maverick')
    # parser.add_argument('--model', type=str, default='meta-llama/llama-4-scout')
    # parser.add_argument('--model', type=str, default='google/gemma-3-27b-it')
    parser.add_argument('--output-dir', type=Path, default=default_output_dir,
                       help='Directory to save results')
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--N-samples-per-task', type=int, default=None,
                       help='Number of samples to run per task type (default: None, use all)')
    parser.add_argument('--add-context', action='store_true')
    parser.add_argument('--workers', type=int, default=256,
                       help='Number of parallel workers (default: 8, set to 1 for sequential)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save results every N tasks (default: 10)')
    parser.add_argument('--max-retries', type=int, default=10,
                       help='Maximum retry attempts for API calls (default: 5)')
    parser.add_argument('--timeout', type=int, default=6000,
                       help='Request timeout in seconds (default: 60)')
    parser.add_argument('--max-tokens', type=int, default=4096*2)
    parser.add_argument('--use-format-example-group', type=int, choices=[1, 2], default=1,
                       help='Which format example group to use (1 or 2, default: 1)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start inference from scratch, ignore existing results (default: resume from existing)')
    parser.add_argument('--enable-thinking', action='store_true',
                       help='Enable reasoning/thinking mode for models that support it')
    parser.add_argument('--eval-only', action='store_true',
                       help='Re-evaluate existing results without running inference (regenerates stats and pretty JSON)')

    return parser.parse_args()


if __name__ == "__main__":
    main()
