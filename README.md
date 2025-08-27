# Meeting Transcript Summarizer

A LangChain/LangGraph program designed to analyze meeting transcripts, extract key information, and generate concise summaries. This tool leverages advanced LLM capabilities with robust error handling, flexible configuration, and an iterative refinement approach for accurate results.

## Features

*   **Automated Analysis:** Extracts topics, key insights, and decisions from meeting transcripts.
*   **Iterative Refinement:** Processes documents chunk-by-chunk, ensuring focused and accurate analysis.
*   **LLM Flexibility:** Supports multiple LLM providers (Ollama, OpenAI, Google Generative AI) with dynamic loading.
*   **Structured Output:** Utilizes Pydantic models to ensure consistent and structured output from LLM interactions.
*   **Robustness:** Includes retry mechanisms for LLM calls and comprehensive error handling.
*   **Configurable:** Easily customize behavior via command-line arguments or a YAML configuration file.
*   **Progress Tracking:** Provides real-time progress updates during analysis.

## Installation

To install the project and its dependencies, run the following command:

```bash
pip install interview-summarizer
```

This will install the necessary packages and set up the `interview-summarizer` command-line tool.

## Prerequisites

You will also need to set up API keys for OpenAI or Google Generative AI if you plan to use those models. For Ollama, ensure you have the Ollama server running and the desired model pulled.

## Usage

You can run the program by providing a transcript file and specifying your desired LLM model and output options.

```bash
interview-summarizer <path_to_transcript.txt> --model <ollama|openai|google> [options]
```

Alternatively, you can use a YAML configuration file:

```bash
interview-summarizer --config config.yaml
```

### Command-Line Arguments

*   `<transcript_path>`: Path to the transcript file (.txt). (Required if not using `--config` with a pre-defined path).
*   `--output <file_path>`: Path to save the analysis results.
*   `--format <console|json|markdown>`: Output format for the results. Defaults to `console`.
*   `--config <file_path>`: Path to a YAML configuration file.
*   `--provider <ollama|openai|google>`: Specify the LLM provider to use. Overrides `model_provider` in config.
*   `--model <name>`: Specify the model name for the chosen provider (e.g., `llama3`, `gpt-4o`, `gemini-1.5-flash-latest`). Overrides the default model name for the selected provider in config.
*   `--openai-api-base-url <url>`: Specify an alternative base URL for the OpenAI API (e.g., `http://localhost:1234/v1` for local LLMs).

### Configuration Files

The program uses `prompts.yaml` and `config.yaml` for its default settings and prompt templates. These files are included in the installed package.

*   You can find the default `config.yaml` and `prompts.yaml` in the installed package directory (e.g., `path/to/your/python/site-packages/interview_summarizer/`).
*   To customize, you can copy these files, modify them, and then specify your custom `config.yaml` using the `--config` command-line argument.

### Example `config.yaml`

```yaml
model_provider: openai
model_name: gpt-4o-mini
chunk_size: 4000
chunk_overlap: 400
max_file_size_mb: 50
output_format: markdown
output_file: analysis_results.md
max_retries: 3
retry_delay: 2.0
enable_progress_bar: true
allowed_extensions: ['.txt']
prompts_file: prompts.yaml
openai_api_base_url: http://localhost:1234/v1 # Example for local LLM or proxy
```

## Program Flow

The `interview-summarizer` program orchestrates a multi-step analysis of meeting transcripts using a LangChain/LangGraph workflow.

1.  **Initialization:**
    *   The program starts by parsing command-line arguments and loading configuration settings from a `config.yaml` file (or using defaults).
    *   Prompt templates are loaded from `prompts.yaml` using a `PromptManager`.
    *   An initial state is prepared, containing the transcript path, configuration, and placeholders for results.

2.  **Workflow Definition (LangGraph):**
    *   A `StateGraph` defines the sequential flow of analysis steps:
        *   `load_and_split`: Loads and preprocesses the transcript.
        *   `identify_topics`: Identifies key discussion topics.
        *   `summarize_topics`: Generates summaries for each identified topic.
        *   `extract_key_insights`: Extracts overarching key insights.
        *   `extract_decisions`: Pinpoints decisions made or advised.
        *   `generate_executive_summary`: Creates a final executive overview.
    *   Edges connect these nodes, ensuring a structured progression through the analysis.

3.  **Transcript Loading and Chunking:**
    *   The `load_and_split_transcript` node validates the input file and reads its content.
    *   It then uses `RecursiveCharacterTextSplitter` to divide the transcript into smaller, manageable `Document` chunks. This is crucial for handling large transcripts and fitting them within the context window limits of Language Models (LLMs).

4.  **Iterative Analysis with LLMs:**
    *   Most subsequent analysis steps (e.g., `identify_topics`, `summarize_topics`) employ a generic `IterativeRefiner` class.
    *   The `IterativeRefiner` processes the document chunks sequentially. It applies an "initial prompt" to the first chunk and then uses a "refine prompt" for subsequent chunks, incorporating previous results to build a comprehensive and coherent output. This ensures context is maintained across the entire document.
    *   LLMs are dynamically instantiated using `LLMProvider.create_llm`, which selects the appropriate LangChain chat model (Ollama, OpenAI, or Google Generative AI) based on the configured `model_provider`.
    *   `PydanticOutputParser` is extensively used to enforce structured output (e.g., JSON lists of topics, insights, decisions), ensuring data consistency.
    *   `@retry_on_failure` decorators are applied to these nodes, making the analysis robust against transient LLM errors.
    *   Progress updates are displayed on the console throughout the process.

5.  **Executive Summary Generation:**
    *   The `generate_executive_summary` node takes the individual topic summaries and refines them into a single, concise executive overview of the entire meeting.

6.  **Output and Completion:**
    *   Upon completion of the LangGraph workflow, the final analysis results are collected.
    *   Results can be printed to the console or saved to a specified output file in either JSON or Markdown format, providing a structured summary of the meeting.
    *   The program logs any errors or warnings encountered during the process and exits with an appropriate status code.
