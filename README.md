# PDF Enrichment Pipeline

Extracts text from PDFs, detects figures/charts using MinerU, and uses Gemini AI to analyze visual content with 3-page context window.

## Features

- **Text Extraction** - LLMWhisperer for high-quality PDF text extraction
- **Figure Detection** - MinerU's DocLayout-YOLO model detects charts and figures
- **AI Analysis** - Gemini analyzes figures with surrounding page context
- **Structured Output** - Markdown tables, summaries, and explanations for each figure
- **Token Tracking** - Monitors API usage statistics

## Architecture

```
PDF Input
    │
    ├─────────────────┬───────────────────┐
    ▼                 ▼                   │
┌────────────┐  ┌───────────┐            │
│LLMWhisperer│  │  MinerU   │            │
│  (text)    │  │ (figures) │            │
└─────┬──────┘  └─────┬─────┘            │
      │               │                  │
      │    ┌──────────┘                  │
      ▼    ▼                             │
┌────────────────────┐                   │
│  Gemini Analysis   │◄──────────────────┘
│ (3-page context)   │     (context)
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Enriched Output   │
└────────────────────┘
```

## Prerequisites

1. **Python 3.9+**

2. **MinerU** - Document layout analysis
   ```bash
   git clone https://github.com/opendatalab/MinerU.git
   cd MinerU && pip install -e .
   ```

3. **API Keys**
   - [Google Gemini](https://makersuite.google.com/app/apikey)
   - [LLMWhisperer](https://unstract.com/llmwhisperer/)

## Installation

```bash
git clone https://github.com/yourusername/advanced_ocr.git
cd advanced_ocr
pip install -r requirements.txt
```

## Configuration

Edit the config section at the top of `pdf_enrich_pipeline_stats.py`:

```python
# API Keys
GEMINI_API_KEY = "your_key_here"
LLMWHISPERER_API_KEY = "your_key_here"

# Paths
PDF_DIR = Path("./pdfs")      # Your input PDFs
OUT_DIR = Path("./output")    # Output directory
MINERU_PATH = Path("./MinerU")

# Settings
DEVICE = "cpu"                # or "cuda"
DPI = 300
GEMINI_MODEL = "gemini-2.5-flash"
```

Or use environment variables:
```bash
export GEMINI_API_KEY="your_key"
export LLMWHISPERER_API_KEY="your_key"
```

## Usage

```bash
python pdf_enrich_pipeline_stats.py
```

## Output

For each PDF, generates `{filename}_enriched.txt` containing:
- Extracted text per page
- For each figure:
  - Markdown table of data
  - Summary (4-6 lines)
  - Context explanation (2-3 lines)

Example:
```
[Page text...]

    | Date | Value | Change |
    |------|-------|--------|
    | Q1   | $2.1B | +3.2%  |

    Summary: Quarterly revenue chart showing...

    Explanation: The upward trend indicates...

page_end
[Next page...]
```

## License

MIT
