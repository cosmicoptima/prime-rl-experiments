from datasets import load_dataset
import json
import re
import verifiers as vf
from transformers import AutoTokenizer


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
    MAX_PROMPT_TOKENS = 2076 # 3264 ... - 128 because 3264 still errored at one point
    
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
        text = " ".join(m.get("content", "") for m in completion if m.get("role") == "assistant")
        answer = json.loads(answer)
        
        predicted_groups_str = parser.parse_answer(text)
        predicted_groups = json.loads(predicted_groups_str)
        
        if not predicted_groups:
            return 0.0
        
        actual_indices = set()
        for group in answer:
            actual_indices.update(group)
        predicted_indices = set()
        for group in predicted_groups:
            predicted_indices.update(group)
        if actual_indices != predicted_indices:
            return 0.0  # Missing or extra comments
        
        all_predicted_comments = []
        for group in predicted_groups:
            all_predicted_comments.extend(group)
        if len(all_predicted_comments) != len(set(all_predicted_comments)):
            return 0.0  # Duplicate assignment found
        
        all_actual_comments = []
        for group in answer:
            all_actual_comments.extend(group)
        
        # Original exact-match reward calculation (commented out)
        # reward = 0.0
        # for group in answer:
        #     if group in predicted_groups:
        #         reward += len(group) / len(all_actual_comments)
        
        # F1 weighted group score with power=4 and comment normalization
        total_comments = sum(len(group) for group in answer)
        weighted_score = 0.0
        
        for actual_group in answer:
            # Calculate this group's weight based on comment share
            weight = len(actual_group) / total_comments
            actual_set = set(actual_group)
            
            # Find best F1 score against all predicted groups
            best_f1 = 0.0
            for pred_group in predicted_groups:
                pred_set = set(pred_group)
                intersection = actual_set & pred_set
                
                # Calculate precision and recall
                precision = len(intersection) / len(actual_group) if len(actual_group) > 0 else 0.0
                recall = len(intersection) / len(pred_group) if len(pred_group) > 0 else 0.0
                
                # Calculate F1 score
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    best_f1 = max(best_f1, f1)
            
            # Add weighted contribution
            weighted_score += weight * best_f1
        
        # Apply power scaling
        reward = weighted_score ** 4
        
        return reward
    
    rubric = vf.Rubric(
        funcs=[reward],
        weights=[1.0],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )