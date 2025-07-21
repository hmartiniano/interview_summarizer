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
pip install .
```

This will install the necessary packages and set up the `interview-summarizer` command-line tool.

## Prerequisites

You will also need to set up API keys for OpenAI or Google Generative AI if you plan to use those models. For Ollama, ensure you have the Ollama server running and the desired model pulled.

## Usage

You can run the program by providing a transcript file and specifying your desired LLM model and output options.

```bash
python interview_summarizer.py <path_to_transcript.txt> --model <ollama|openai|google> [options]
```

Alternatively, you can use a YAML configuration file:

```bash
python interview_summarizer.py --config config.yaml
```

### Command-Line Arguments

*   `<transcript_path>`: Path to the transcript file (.txt). (Required if not using `--config` with a pre-defined path).
*   `--output <file_path>`: Path to save the analysis results.
*   `--format <console|json|markdown>`: Output format for the results. Defaults to `console`.
*   `--config <file_path>`: Path to a YAML configuration file.
*   `--model <ollama|openai|google>`: Specify the LLM provider to use. Overrides `model_provider` in config.
*   `--ollama-model <name>`: Ollama model name (e.g., `llama3`). Overrides `ollama_model_name` in config.
*   `--openai-model <name>`: OpenAI model name (e.g., `gpt-4o`). Overrides `openai_model_name` in config.
*   `--google-model <name>`: Google model name (e.g., `gemini-1.5-flash`). Overrides `google_model_name` in config.

### Example `prompts.yaml`

The program uses a `prompts.yaml` file to manage prompt templates. A basic structure might look like this:

```yaml
identify_topics:
  initial_prompt: |
    You are an expert in meeting analysis. Identify the main topics discussed in the following text.
    Return a JSON list of topics.
    Text: {text}
    {format_instructions}
  refine_prompt: |
    You are an expert in meeting analysis. You have already identified the following topics: {existing_answer}.
    Now, refine and add any new topics from the following text.
    Return a JSON list of topics.
    Text: {text}
    {format_instructions}

summarize_topics:
  initial_prompt: |
    Summarize the following text focusing on the topic: "{topic}".
    Text: {text}
  refine_prompt: |
    You have already summarized the topic "{topic}" as: {existing_answer}.
    Now, refine and expand this summary using the following text.
    Text: {text}

extract_key_insights:
  initial_prompt: |
    From the following text, extract key insights.
    Return a JSON list of insights.
    Text: {text}
    {format_instructions}
  refine_prompt: |
    You have already extracted the following insights: {existing_answer}.
    Now, refine and add any new key insights from the following text.
    Return a JSON list of insights.
    Text: {text}
    {format_instructions}

extract_decisions:
  initial_prompt: |
    From the following text, identify any decisions made or advised.
    Return a JSON list of decisions.
    Text: {text}
    {format_instructions}
  refine_prompt: |
    You have already identified the following decisions: {existing_answer}.
    Now, refine and add any new decisions from the following text.
    Return a JSON list of decisions.
    Text: {text}
    {format_instructions}

generate_executive_summary:
  initial_prompt: |
    Based on the following topic summaries, generate a concise executive overview of the meeting.
    Summaries: {text}
  refine_prompt: |
    You have already generated a partial executive overview: {existing_answer}.
    Now, refine and expand this overview using the following additional summaries.
    Summaries: {text}
```

### Example `config.yaml`

```yaml
model_provider: openai
openai_model_name: gpt-4o-mini
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
```

## Program Flow

The `interview_summarizer.py` program orchestrates a multi-step analysis of meeting transcripts using a LangChain/LangGraph workflow.

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
