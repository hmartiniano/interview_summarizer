# Meeting Summarizer

A LangChain/LangGraph program designed to analyze meeting transcripts, extract key information, and generate concise summaries. This tool leverages advanced LLM capabilities with robust error handling, flexible configuration, and an iterative refinement approach for accurate results.

## Features

*   **Automated Analysis:** Extracts topics, key insights, decisions, and action items from meeting transcripts.
*   **Iterative Refinement:** Processes documents chunk-by-chunk, ensuring focused and accurate analysis.
*   **LLM Flexibility:** Supports multiple LLM providers (Ollama, OpenAI, Google Generative AI) with dynamic loading.
*   **Structured Output:** Utilizes Pydantic models to ensure consistent and structured output from LLM interactions.
*   **Robustness:** Includes retry mechanisms for LLM calls and comprehensive error handling.
*   **Configurable:** Easily customize behavior via command-line arguments or a YAML configuration file.
*   **Progress Tracking:** Provides real-time progress updates during analysis.

## Installation

To install the project and its dependencies, run the following command:

```bash
pip install meeting-summarizer
```

This will install the necessary packages and set up the `meeting-summarizer` command-line tool.

## Prerequisites

You will also need to set up API keys for OpenAI or Google Generative AI if you plan to use those models. For Ollama, ensure you have the Ollama server running and the desired model pulled.

## Usage

You can run the program by providing a transcript file and specifying your desired LLM model and output options.

```bash
meeting-summarizer [transcript_path] --model [provider]/<model_name> [options]
```

Alternatively, you can use a YAML configuration file:

```bash
meeting-summarizer --config config.yaml
```

## Configuration

The application can be configured in multiple ways, with the following order of precedence (highest to lowest):

1.  **Command-Line Arguments:** Direct, explicit settings for a single run.
2.  **Environment Variables:** Ideal for containerized environments or CI/CD systems.
3.  **YAML Configuration File:** For defining a base set of preferences.
4.  **Default Values:** Hardcoded defaults in the application.

### 1. Command-Line Arguments

*   `transcript_path`: Path to the transcript file (.txt).
*   `--model`: Specify the model in `provider/model_name` format (e.g., `openai/gpt-4o`). Overrides `MODEL_PROVIDER` and `MODEL_NAME` environment variables.
*   `--output`: Path to save the analysis results. Overrides `OUTPUT_FILE`.
*   `--format`: Output format (`console`, `json`, `markdown`). Overrides `OUTPUT_FORMAT`.
*   `--config`: Path to a YAML configuration file.
*   `--log-level`: Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Overrides `LOG_LEVEL`.
*   `--interview`: Use interview-focused prompts.
*   `--iterative-analysis`, `--merge-topics`, etc.: Flags to control specific workflow steps.

### 2. Environment Variables

You can set environment variables to configure the application. For local development, you can create a `.env` file in the project root (a `.env.example` is provided).

*   `MODEL_PROVIDER`: The LLM provider (e.g., `ollama`, `openai`).
*   `MODEL_NAME`: The name of the model (e.g., `llama3`, `gpt-4o`).
*   `OPENAI_API_BASE_URL`: An alternative base URL for the OpenAI API.
*   `OLLAMA_MODEL_OPTIONS`: A JSON string for Ollama options (e.g., `'{"num_ctx": 8192}'`).
*   `OUTPUT_FORMAT`: The output format.
*   `OUTPUT_FILE`: The path to the output file.
*   `LOG_LEVEL`: The logging level.

API keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) are also read from environment variables by the underlying LangChain libraries.

### 3. YAML Configuration File

You can use a `config.yaml` file to set base configurations. Use the `--config` argument to specify a path to your file.

**Example `config.yaml`:**
```yaml
model_provider: openai
model_name: gpt-4o-mini
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

The `meeting-summarizer` program orchestrates a multi-step analysis of meeting transcripts using a LangChain/LangGraph workflow.

1.  **Initialization:**
    *   The program starts by parsing command-line arguments and loading configuration settings from a `config.yaml` file (or using defaults).
    *   Prompt templates are loaded from `prompts_meeting.yaml` using a `PromptManager`.
    *   An initial state is prepared, containing the transcript path, configuration, and placeholders for results.

2.  **Workflow Definition (LangGraph):**
    *   A `StateGraph` defines the sequential flow of analysis steps:
        *   `load_and_split`: Loads and preprocesses the transcript.
        *   `identify_topics`: Identifies key discussion topics.
        *   `summarize_topics`: Generates summaries for each identified topic.
        *   `extract_key_insights`: Extracts overarching key insights.
        *   `extract_decisions`: Pinpoints decisions made or advised.
        *   `extract_action_items` (optional): Extracts action items.
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
