# Gemini Prompt Refinement Loop

An automated system that iteratively improves prompts by using Google's Gemini API to generate responses, scoring them, and refining prompts based on feedback until a quality threshold is met.

## Quick Start

### Prerequisites
- Python 3.14
- Google Gemini API key ([Get free key](https://ai.google.dev))

### Installation & Environment Setup

**Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Set Up API Key** (Choose one method)

#### Option A: Using .env File (Recommended)
Create a `.env` file in the project root:
```bash
echo "GEMINI_API_KEY=your-actual-api-key-here" > .env
```

The application will automatically load it via `python-dotenv`.

#### Option B: Export in Terminal
```bash
# Bash/Zsh
export GEMINI_API_KEY=your-actual-api-key-here
```

#### Option C: Activate Python Virtual Environment (Best Practice)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# Then install dependencies
pip install -r requirements.txt

# Set API key (stays within venv)
export GEMINI_API_KEY=your-actual-api-key-here
```

### Usage
```bash
python Main.py "Your prompt here"
python Main.py "Your prompt here" --log-file file_name.json  # Save history
```

## How It Works

1. **Generate**: Send prompt to Gemini → get response
2. **Judge**: Score response (0-10) + critique
3. **Refine**: If score < threshold, rewrite prompt based on feedback
4. **Repeat**: Loop until score meets target (default 8.0) or max iterations (4)

## Configuration

Edit `llm_config.py` to control loop behavior:

```python
model: str = "gemini/gemini-3.1-flash-lite-preview"           # Model for generating responses
judge_model: str = "gemini/gemini-3.1-flash-lite-preview"     # Model for scoring/critiquing
temperature: float = 0.7                                       # Creativity (0=deterministic, 1=random)
max_iterations: int = 4                                        # Maximum refinement cycles
score_threshold: float = 8.0                                   # Exit when score ≥ this (0-10 scale)
```

### How Configuration Affects Output

| Setting | Impact | Real Example |
|---------|--------|--------------|
| **score_threshold** | Loop stops when judge score reaches this value | `8.0`: output.json stopped at iter 1 (score 10.0), output2.json needed 3 iterations (7→7→8) |
| **max_iterations** | Hard limit on refinement cycles | Set to `4`: if score never hits threshold, loop stops after 4 attempts (see output1.json: 4 iterations) |
| **temperature** | Higher = more creative, lower = more consistent | `0.7`: balanced; `0.0` = deterministic judge, `1.0` = more varied responses |
| **judge_model** | Determines scoring strictness | Same model used for both main & judge; stricter model = higher thresholds needed |

### Judge Instructions Impact

The judge's scoring criteria determine whether responses pass. Default: **score ≥ 8.0 to pass**.

**Example Flow (output2.json):**
- **Iter 1** (score 7): Judge critiques → "lacks real-world applications"
- **Iter 2** (score 7): Judge critiques → "lacks concrete examples"  
- **Iter 3** (score 9 > 8.0): Judge approves → "exceptional explanation" → **STOP**

**Example Flow (output.json):**
- **Iter 1** (score 10): Judge approves → "excellent, accessible explanation" → **STOP** (immediate pass)

Lower `score_threshold` = fewer iterations needed. Higher = more refinement cycles.

## Core Modules

- **`llm_client.py`**: Wrapper around LiteLLM with retry logic (exponential backoff + jitter)
- **`prompt_loop.py`**: Main loop that generates responses, judges them, and refines prompts
- **`llm_config.py`**: Centralized configuration management
- **`Main.py`**: CLI entry point

## Example

```bash
python Main.py "Explain quantum computing"
```

Output shows iteration score and final optimized prompt.

## Test

```bash
pytest test_prompt_loop.py
```

Uses mock clients for 100% deterministic testing without hitting the real API.

## License

MIT
