from datasets import load_dataset
import json
import re
from sklearn.metrics import normalized_mutual_info_score
import verifiers as vf
from transformers import AutoTokenizer
import os


# Global tokenizer instance to avoid reloading
_tokenizer = None


def get_tokenizer():
    """Get or create the tokenizer instance"""
    global _tokenizer
    if _tokenizer is None:
        # Use the same model as specified in the config files
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _tokenizer


def count_tokens(text):
    """Count the number of tokens in a text string"""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(str(text)))


class UserDiffParser(vf.Parser):
    def parse_answer(self, response):
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if not match:
            return "[]"

        answer_block = match.group(1)
        user_lines = re.findall(r"User\s*\d+\s*:\s*([0-9,\s]+)", answer_block)

        groups = []
        for line in user_lines:
            nums = [int(n.strip()) - 1 for n in line.split(",") if n.strip().isdigit()]
            groups.append(nums)

        return json.dumps(groups) # i think parse_answer has to return a string
    
    def get_format_reward_func(self):
        def reward_func(completion, **kwargs):
            text = " ".join(m.get("content", "") for m in completion if m.get("role") == "assistant")
            processed_answer = json.loads(self.parse_answer(text))
            print(processed_answer)
            if len(processed_answer) == 0:
                return 0.0
            return 1.0
        
        return reward_func


def render_question(thread: dict) -> str:
    title = thread["title"]
    body = thread["body"]
    comments = [c["comment"].replace("\n", "\n        ") for c in thread["comments"]]
    comments = "\n".join([f"    <comment>\n      <n>{i+1}</n><user>???</user>\n      <text>\n        {c['comment']}\n      </text>\n    </comment>" for i, c in enumerate(thread["comments"])])

    return f"""<context>
  <submission>
    <title>{title}</title>
    <body>{body}</body>
  </submission>
  <thread>
{comments}
  </thread>
</context>

What you see inside the above <context> block is a Reddit post and a single comment thread under that post. The usernames in the comment thread are redacted. Your task is to determine which comments in the thread are from the same user. Think as much as you need to in a <reasoning> block, then output your answer in an <answer> block in the following format:

<example_answer>
User 1: 1, 3, 4
User 2: 2, 5
</example_answer>
"""


def calculate_answer(thread: dict) -> str:
    comments = thread["comments"]
    usernames = list(set([c["username"] for c in comments]))
    return json.dumps([[i for i, c in enumerate(comments) if c["username"] == u] for u in usernames])


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("cosmicoptima/IFhXR5QAHNW9", split="train")
    dataset = dataset.filter(lambda x: all(k in x for k in ["title", "body", "comments"]))
    dataset = dataset.map(lambda x: {"question": render_question(x), "answer": calculate_answer(x), "task": "reddit_user_differentiation"})
    
    # Filter out examples that are too long
    MAX_PROMPT_TOKENS = 3584
    
    print("Filtering dataset by token count...")
    original_size = len(dataset)
    
    # Add token count to each example
    dataset = dataset.map(lambda x: {"token_count": count_tokens(x["question"])})
    
    # Filter to keep only examples under the token limit
    dataset = dataset.filter(lambda x: x["token_count"] <= MAX_PROMPT_TOKENS)
    
    filtered_size = len(dataset)
    print(f"Filtered dataset from {original_size} to {filtered_size} examples (removed {original_size - filtered_size} examples exceeding {MAX_PROMPT_TOKENS} tokens)")
    
    # Remove the temporary token_count field
    dataset = dataset.remove_columns(["token_count"])

    parser = UserDiffParser()

    def reward(completion, answer, **kwargs):
        # Extract the assistant's response from the message list
        text = " ".join(m.get("content", "") for m in completion if m.get("role") == "assistant")
        answer = json.loads(answer)
        
        # Parse the completion to get predicted groups
        predicted_groups_str = parser.parse_answer(text)
        predicted_groups = json.loads(predicted_groups_str)
        
        # If no valid groups were parsed, return 0
        if not predicted_groups:
            return 0.0
        
        # Get all actual comment indices
        actual_indices = set()
        for group in answer:
            actual_indices.update(group)
        
        # Get all predicted comment indices  
        predicted_indices = set()
        for group in predicted_groups:
            predicted_indices.update(group)
        
        # Validation checks:
        # 1. Check that all actual comments are classified (no missing comments)
        # 2. Check that no extra/out-of-bounds comments are included
        if actual_indices != predicted_indices:
            return 0.0  # Missing or extra comments
        
        # 3. Check for duplicate assignments (comment in multiple groups)
        all_predicted_comments = []
        for group in predicted_groups:
            all_predicted_comments.extend(group)
        if len(all_predicted_comments) != len(set(all_predicted_comments)):
            return 0.0  # Duplicate assignment found
        
        # Convert actual groups (answer) and predicted groups to cluster labels
        # First, determine the total number of comments
        all_indices = set()
        for group in answer:
            all_indices.update(group)
        for group in predicted_groups:
            all_indices.update(group)
        
        if not all_indices:
            return 0.0
            
        max_index = max(all_indices) if all_indices else -1
        
        # Create label arrays for actual groups
        actual_labels = [-1] * (max_index + 1)
        for group_id, group in enumerate(answer):
            for comment_idx in group:
                if 0 <= comment_idx <= max_index:
                    actual_labels[comment_idx] = group_id
        
        # Create label arrays for predicted groups
        predicted_labels = [-1] * (max_index + 1)
        for group_id, group in enumerate(predicted_groups):
            for comment_idx in group:
                if 0 <= comment_idx <= max_index:
                    predicted_labels[comment_idx] = group_id
        
        # Calculate NMI score
        # Filter out any unassigned comments (label = -1)
        valid_indices = [i for i in range(len(actual_labels)) if actual_labels[i] != -1 and predicted_labels[i] != -1]
        
        if not valid_indices:
            return 0.0
        
        actual_valid = [actual_labels[i] for i in valid_indices]
        predicted_valid = [predicted_labels[i] for i in valid_indices]
        
        # Calculate NMI score
        nmi_score = normalized_mutual_info_score(actual_valid, predicted_valid)
        
        return nmi_score
    
    rubric = vf.Rubric(
        funcs=[reward, parser.get_format_reward_func()],
        weights=[0.83, 0.17],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )