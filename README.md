# pdf2skills

Convert PDF books into Skills callable to Claude Code.

Please see recent updates: https://github.com/kitchen-engineer42/Anything2Workspace

## Quick Start

### 1. Install Dependencies

```bash
pip install requests python-dotenv python-Levenshtein numpy scikit-learn spacy jieba PyPDF2

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 2. Setup MinerU Client

Copy `mineru_client.py` to the parent directory of `pdf2skills/`:

```
your_project/
├── mineru_client.py      # MinerU API client
├── pdf2skills/           # This folder
│   ├── run_pipeline.py
│   └── ...
└── test_data/            # Your PDFs here
```

Get your MinerU API key at: https://mineru.net/

### 3. Configure Environment

Create a `.env` file in the `pdf2skills` directory:

```bash
# SiliconFlow API (Required - primary LLM provider)
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# MinerU API (Required - PDF to Markdown conversion)
MINERU_API_KEY=your_mineru_api_key
MINERU_BASE_URL=https://mineru.net/api/v4/extract/task

# Pipeline Configuration
CHUNK_MAX_TOKENS=30000
CHUNK_MAX_ITERATIONS=3
OUTPUT_LANGUAGE=English

# Rate Limiting (adjust based on your API tier)
GLM_RATE_LIMIT_SECONDS=3.0
FUSION_RATE_LIMIT_SECONDS=2.0
BUCKET_THRESHOLD=0.5
```

### 4. Run the Pipeline

```bash
# Basic usage
python run_pipeline.py your_book.pdf

# With custom output directory
python run_pipeline.py your_book.pdf --output-dir ./output

# For English PDFs
python run_pipeline.py your_book.pdf --language en

# Resume interrupted processing
python run_pipeline.py your_book.pdf --resume

# Cache LLM responses (in <output_dir>/.llm_cache) so re-runs are free
python run_pipeline.py your_book.pdf --resume --llm-cache
```

### 5. Output Structure

After processing, you'll find:

```
your_book_output/
├── full.md                          # Extracted markdown
├── full_chunks/                     # Chunked documents
│   ├── chunks_index.json
│   └── chunks/
├── full_chunks_density/             # Semantic analysis
│   ├── density_scores.json
│   └── heatmap.html                 # Visual density map
└── full_chunks_skus/                # Knowledge units
    ├── skus/                        # Individual SKU files
    ├── buckets.json                 # Grouped SKUs
    └── generated_skills/            # Claude Code Skills
        ├── index.md                 # Skill navigation
        └── <skill-name>/
            ├── SKILL.md             # Main skill file
            └── references/          # Detailed documentation
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | MinerU API | PDF → Markdown with OCR |
| 2 | Onion Peeler | Recursive semantic chunking |
| 3 | Semantic Density | NLP scoring + LLM calibration |
| 4 | SKU Extractor | Knowledge unit extraction |
| 5 | Knowledge Fusion | Tag normalization + deduplication |
| 6 | Skill Generator | SKU → Claude Skill conversion |

## Using Individual Modules

Each module can be run independently:

```bash
# Chunking only
python -m pdf2skills.onion_peeler path/to/full.md

# Density analysis only
python -m pdf2skills.semantic_density path/to/chunks_dir

# SKU extraction only
python -m pdf2skills.sku_extractor path/to/chunks_dir -d density_scores.json

# Knowledge fusion only
python -m pdf2skills.knowledge_fusion path/to/skus_dir

# Skill generation only
python -m pdf2skills.skill_generator path/to/skus_dir -o output_dir
```

## LLM Layer (`llm_client.py`)

Every stage talks to the model through one shared client, `llm_client.py`:

- **Any OpenAI-compatible provider.** Set `LLM_BASE_URL` / `LLM_API_KEY` (falls back to `SILICONFLOW_BASE_URL` / `SILICONFLOW_API_KEY`).
- **Retries with exponential backoff + jitter** on 429 / 5xx / timeouts / malformed responses, honouring `Retry-After`. Non-retryable 4xx errors fail fast.
- **Robust JSON parsing.** Code fences, `<think>` blocks, prose around the JSON and trailing commas are tolerated. If parsing still fails, the model is asked once to fix its own output.
- **A shared, thread-safe rate limiter** per model, so stages that share GLM-4.7 also share one budget.
- **An optional disk cache** (`--llm-cache` or `LLM_CACHE_DIR`). Re-runs and `--resume` don't pay twice for identical prompts.
- **A usage summary** (calls, retries, tokens per model) printed at the end of `run_pipeline.py`.

### Models by role

Defaults target SiliconFlow; override any role with `MODEL_<ROLE>`:

| Role | Stage | Default model |
|------|-------|---------------|
| `chunking` | Onion Peeler | `deepseek-ai/DeepSeek-V3.2` |
| `density` | Density calibration | `deepseek-ai/DeepSeek-R1` |
| `extraction` | SKU extraction | `Pro/zai-org/GLM-4.7` |
| `fusion` | Knowledge fusion | `deepseek-ai/DeepSeek-V3` |
| `skill` | Skill generation | `Pro/zai-org/GLM-4.7` |
| `router` | Router generation | `Pro/zai-org/GLM-4.7` |
| `glossary` | Glossary (LLM-assisted) | `Pro/zai-org/GLM-4.7` |
| `embedding` | Fusion similarity | `Pro/BAAI/bge-m3` |

Per-role overrides: `MODEL_<ROLE>`, `RATE_LIMIT_<ROLE>`, `TIMEOUT_<ROLE>`, `MAX_TOKENS_<ROLE>` (default when a call site doesn't pass one). Example: `MODEL_EXTRACTION=deepseek-ai/DeepSeek-V3.2`.

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_MAX_TOKENS` | 30000 | Maximum tokens per chunk |
| `CHUNK_MAX_ITERATIONS` | 3 | Max recursive chunking depth |
| `OUTPUT_LANGUAGE` | English | Output language for skills |
| `GLM_RATE_LIMIT_SECONDS` | 3.0 | Min interval between GLM calls (extraction/skill/router/glossary) |
| `FUSION_RATE_LIMIT_SECONDS` | 2.0 | Min interval between fusion LLM / embedding calls |
| `BUCKET_THRESHOLD` | 0.5 | Tag overlap threshold for grouping |
| `LLM_MAX_RETRIES` | 5 | Retries per request on transient errors |
| `LLM_BACKOFF_BASE_SECONDS` / `LLM_BACKOFF_MAX_SECONDS` | 2 / 60 | Exponential backoff bounds |
| `LLM_JSON_RETRIES` | 1 | "Fix your JSON" follow-ups when output can't be parsed |
| `LLM_CACHE_DIR` | (off) | Directory for the response cache |
| `EMBED_BATCH_SIZE` | 32 | Texts per embedding request |
| `LLM_VERBOSE` | 1 | Set `0` to silence retry/truncation logs |

See `.env.example` for a template.

## Tests

```bash
pip install pytest
python -m pytest tests -q   # offline; HTTP is faked
```

## Troubleshooting

**Rate limit errors (429)**
- 429s are retried automatically with backoff. If you still see failures, increase `GLM_RATE_LIMIT_SECONDS` or `LLM_MAX_RETRIES`
- Use `--resume --llm-cache` to continue without re-paying for completed calls

**Memory issues with large PDFs**
- Use the `pdf_splitter.py` utility to split large PDFs
- Adjust `CHUNK_MAX_TOKENS` to smaller values

**Missing spaCy model**
```bash
python -m spacy download en_core_web_sm
```

## License

MIT License

## Version

1.0 - Initial release
