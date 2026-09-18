# Meeting Prep AI: A Multi-Agent System with Crew AI

An AI-powered multi-agent system for automated meeting preparation and note-taking, built with [CrewAI](https://github.com/joaomdmoura/crewAI) and local LLMs via [Ollama](https://ollama.ai).

## What It Does

Meeting Prep AI uses a crew of specialized AI agents to handle two core workflows:

**Pre-Meeting Research** — Research participants, analyze industry trends, develop strategic talking points, and compile a comprehensive briefing document before your meeting.

**Post-Meeting Notes** — Feed in a raw meeting transcript (even noisy audio-to-text output) and get back structured meeting minutes with agenda, summary, and actionable discussion points as clean JSON.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Meeting Prep AI                         │
├──────────────────────────┬──────────────────────────────────┤
│   Pre-Meeting Crew       │   Post-Meeting Crew              │
│                          │                                  │
│  ┌──────────────────┐    │  ┌────────────────────┐          │
│  │ Research Agent    │    │  │ Note-Taker Agent   │          │
│  │ (Exa Search)     │    │  │ (Transcript → JSON)│          │
│  └────────┬─────────┘    │  └────────┬───────────┘          │
│  ┌────────▼─────────┐    │  ┌────────▼───────────┐          │
│  │ Industry Analyst  │    │  │ Validation Agent   │          │
│  │ (Exa Search)     │    │  │ (Schema Check)     │          │
│  └────────┬─────────┘    │  └────────┬───────────┘          │
│  ┌────────▼─────────┐    │  ┌────────▼───────────┐          │
│  │ Strategy Advisor  │    │  │ Backup Agent       │          │
│  └────────┬─────────┘    │  │ (File Persistence) │          │
│  ┌────────▼─────────┐    │  └────────────────────┘          │
│  │ Briefing Agent    │    │                                  │
│  └──────────────────┘    │                                  │
├──────────────────────────┴──────────────────────────────────┤
│  Ollama (llama3.1)  │  Exa Search API  │  Local File I/O   │
└─────────────────────┴──────────────────┴────────────────────┘
```

## Features

- **Multi-agent orchestration** via CrewAI with specialized roles
- **Local LLM support** through Ollama (default: `llama3.1`)
- **Noisy transcript handling** — agents are designed to reason through transcription errors
- **Structured JSON output** with agenda, summary, and discussion points
- **Web research** via Exa Search API for pre-meeting preparation
- **Output validation** with optional validation agent to verify JSON schema
- **File persistence** — all outputs saved locally for backup

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running with a model pulled (e.g., `ollama pull llama3.1`)
- **Exa API key** (only needed for pre-meeting research)

## Installation

```bash
git clone https://github.com/<your-username>/meeting-prep-ai.git
cd meeting-prep-ai

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_MODEL` | No | Ollama model name (default: `llama3.1`) |
| `OPENAI_API_KEY` | No | Set to `NA` if using Ollama only |
| `EXA_API_KEY` | For `prep` | API key for Exa web search |

## Usage

### Generate Meeting Notes from a Transcript

```bash
python -m src.main notes \
  --transcript path/to/transcript.txt \
  --agenda "Q3 Product Roadmap Review"
```

With validation enabled:

```bash
python -m src.main notes \
  --transcript path/to/transcript.txt \
  --agenda "Q3 Product Roadmap Review" \
  --validate
```

### Prepare for an Upcoming Meeting

```bash
python -m src.main prep \
  --participants "alice@company.com,bob@partner.org" \
  --context "Partnership discussion for Q4 integration" \
  --objective "Secure agreement on API integration timeline"
```

## Output Format

The note-taking crew produces structured JSON:

```json
{
    "agenda": "Brief one-line agenda",
    "summary": "Detailed 300-word summary of the meeting...",
    "discussion": [
        {
            "discussion_point": "Description of what was discussed",
            "person_responsible": "Assignee Name",
            "completion_date": "Target date",
            "remarks": "Additional context or notes"
        }
    ]
}
```

See [`examples/sample_output.json`](examples/sample_output.json) for a complete example.

## Project Structure

```
├── src/
│   ├── main.py              # CLI entry point
│   ├── config.py             # LLM and environment configuration
│   ├── agents/
│   │   ├── research.py       # Research, analysis, strategy, briefing agents
│   │   ├── note_taker.py     # Note-taking and backup agents
│   │   └── validation.py     # Output validation agent
│   ├── tasks/
│   │   ├── research.py       # Pre-meeting research tasks
│   │   ├── note_taker.py     # Note-taking and backup tasks
│   │   └── validation.py     # Validation task
│   ├── tools/
│   │   ├── exa_search.py     # Exa web search integration
│   │   ├── file_io.py        # File save and check utilities
│   │   └── validation.py     # JSON schema validation
│   └── crews/
│       ├── meeting_prep.py   # Pre-meeting research crew
│       └── meeting_notes.py  # Post-meeting note-taking crew
├── examples/
│   └── sample_output.json    # Example output
├── output/                   # Runtime output directory (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | [CrewAI](https://github.com/joaomdmoura/crewAI) |
| LLM Runtime | [Ollama](https://ollama.ai) |
| LLM Integration | [LangChain](https://github.com/langchain-ai/langchain) |
| Web Search | [Exa](https://exa.ai) |
| Language | Python 3.10+ |

## License

MIT
