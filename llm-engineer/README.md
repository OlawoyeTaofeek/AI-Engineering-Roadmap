# 🧑‍🚀 The LLM Engineer Roadmap

A structured, hands-on curriculum for becoming an **AI / LLM Engineer** — from running your first model to securing production systems. Each chapter below maps to a folder in this repository, with notes, code, and exercises.

> 📌 Fork it, star it, learn it, ship it. PRs adding notebooks, resources, or corrections are welcome — see [Contributing](#-contributing).

---

## 📖 Table of Contents

| # | Chapter | Core Topics |
|---|---------|--------------|
| 1 | [Running LLMs](#1--running-llms) | LLM APIs · Open-source LLMs · Prompt engineering · Structuring outputs |
| 2 | [Building a Vector Storage](#2--building-a-vector-storage) | Ingesting documents · Splitting documents · Embedding models · Vector databases |
| 3 | [Retrieval Augmented Generation](#3--retrieval-augmented-generation-rag) | Orchestrators · Retrievers · Memory · Evaluation |
| 4 | [Advanced RAG](#4--advanced-rag) | Query construction · Agents and tools · Post-processing · Program LLMs |
| 5 | [Agents](#5--agents) | Agent fundamentals · Agent frameworks · Multi-agents |
| 6 | [Inference Optimization](#6--inference-optimization) | Flash Attention · Key-value cache · Speculative decoding |
| 7 | [Deploying LLMs](#7--deploying-llms) | Local deployment · Demo deployment · Server deployment · Edge deployment |
| 8 | [Securing LLMs](#8--securing-llms) | Prompt hacking · Backdoors · Defensive measures |

---

## 🗂️ Repository Structure

```
llm-engineer-roadmap/
├── README.md
├── LICENSE
├── requirements.txt
├── 01_running_llms/
│   ├── notes.md
│   ├── notebooks/
│   └── resources.md
├── 02_vector_storage/
│   ├── notes.md
│   ├── notebooks/
│   └── resources.md
├── 03_rag/
│   ├── notes.md
│   ├── notebooks/
│   └── resources.md
├── 04_advanced_rag/
│   ├── notes.md
│   ├── notebooks/
│   └── resources.md
├── 05_agents/
│   ├── notes.md
│   ├── notebooks/
│   └── resources.md
├── 06_inference_optimization/
│   ├── notes.md
│   ├── notebooks/
│   └── resources.md
├── 07_deploying_llms/
│   ├── notes.md
│   ├── notebooks/
│   └── resources.md
└── 08_securing_llms/
    ├── notes.md
    ├── notebooks/
    └── resources.md
```

Each chapter folder follows the same pattern: **`notes.md`** (concepts explained in your own words), **`notebooks/`** (runnable code), **`resources.md`** (curated links, papers, tools).

---

## 🚀 How to Use This Repo

1. Clone it: `git clone https://github.com/<your-username>/llm-engineer-roadmap.git`
2. Work chapter by chapter — each builds conceptually on the last (1–2 → RAG basics; 3–4 → RAG mastery; 5 → autonomy; 6–8 → production).
3. Do the exercises inside each `notebooks/` folder before reading the "answers" if provided.
4. Track your progress with the checklists below.
5. Keep a personal `journal.md` per chapter — writing forces retention.

---

## 1. 🏃 Running LLMs

**Goal:** Get comfortable calling, running, and controlling LLM output — the foundation for everything else.

### 1.1 LLM APIs
- Calling hosted models (OpenAI, Anthropic, Google, Mistral, Cohere, etc.)
- Understanding tokens, context windows, temperature/top-p, streaming
- Cost and rate-limit management
- **Exercise:** Build a CLI chatbot that streams responses from an API.

### 1.2 Open-source LLMs
- Model families: Llama, Mistral, Qwen, Gemma, Phi, DeepSeek
- Running locally: `llama.cpp`, Ollama, LM Studio, vLLM, TGI
- Choosing model size vs. hardware (quantization intro — expanded in Ch.6)
- **Exercise:** Run a 7–8B model locally and benchmark tokens/sec on your hardware.

### 1.3 Prompt Engineering
- Zero-shot, few-shot, chain-of-thought, ReAct-style prompting
- System prompts vs. user prompts; instruction hierarchy
- Prompt templates and reusable patterns
- **Exercise:** Solve the same reasoning task with 3 prompting strategies; compare accuracy.

### 1.4 Structuring Outputs
- JSON mode / function-calling-style structured generation
- Grammar-constrained decoding (e.g., outlines, guidance, instructor)
- Validating and repairing malformed outputs
- **Exercise:** Force a model to always return valid JSON matching a Pydantic schema.

**✅ Checklist**
- [ ] Called at least 2 different LLM APIs
- [ ] Ran an open-source model locally
- [ ] Wrote a reusable prompt template library
- [ ] Got 100% valid structured output on a test set

---

## 2. 🗄️ Building a Vector Storage

**Goal:** Turn unstructured documents into a searchable knowledge base.

### 2.1 Ingesting Documents
- Loaders for PDF, HTML, Markdown, DOCX, CSV, images (OCR)
- Cleaning and normalizing text
- Metadata extraction (source, date, author)

### 2.2 Splitting Documents
- Fixed-size vs. recursive vs. semantic chunking
- Overlap strategy and why it matters
- Chunking trade-offs for retrieval quality vs. context cost

### 2.3 Embedding Models
- Dense embeddings (OpenAI, Cohere, BGE, E5, Nomic)
- Sparse/lexical methods (BM25) and hybrid search
- Domain-specific fine-tuning of embeddings

### 2.4 Vector Databases
- Options: FAISS, Chroma, Qdrant, Weaviate, Pinecone, Milvus, pgvector
- Indexing strategies (HNSW, IVF) and ANN trade-offs
- Filtering, hybrid queries, and metadata search

**Exercise:** Build an end-to-end ingestion pipeline: PDF → chunks → embeddings → vector DB → similarity search.

**✅ Checklist**
- [ ] Implemented at least 2 chunking strategies
- [ ] Compared 2 embedding models on the same corpus
- [ ] Stood up a local vector database
- [ ] Ran filtered + semantic hybrid queries

---

## 3. 🦜 Retrieval Augmented Generation (RAG)

**Goal:** Ground LLM answers in retrieved knowledge instead of parametric memory alone.

### 3.1 Orchestrators
- Frameworks: LangChain, LlamaIndex, Haystack
- When to use a framework vs. writing raw pipeline code

### 3.2 Retrievers
- Dense retrieval, hybrid retrieval, re-ranking (cross-encoders)
- Multi-query and query expansion
- Retrieval evaluation metrics (recall@k, MRR, nDCG)

### 3.3 Memory
- Short-term conversational memory vs. long-term persistent memory
- Summarization-based memory compression
- Entity/fact memory stores

### 3.4 Evaluation
- Faithfulness, relevance, and answer correctness
- RAG-specific eval frameworks: RAGAS, TruLens, DeepEval
- Building a golden Q&A test set

**Exercise:** Build a full RAG chatbot over your own document set, with an automated eval pipeline scoring faithfulness and relevance.

**✅ Checklist**
- [ ] Built a RAG pipeline with at least one orchestrator
- [ ] Added re-ranking to the retrieval step
- [ ] Implemented conversational memory
- [ ] Scored the pipeline with a RAG eval framework

---

## 4. 🧰 Advanced RAG

**Goal:** Move beyond naive RAG to production-grade retrieval systems.

### 4.1 Query Construction
- Text-to-SQL / text-to-Cypher for structured data sources
- Metadata filter extraction from natural language queries
- Query routing across multiple data sources

### 4.2 Agents and Tools
- Giving the RAG system tools (search, calculator, code execution)
- Tool-calling loops and deciding when to retrieve vs. act

### 4.3 Post-processing
- Re-ranking, contextual compression, deduplication
- Answer synthesis from multiple retrieved chunks
- Citation and source attribution

### 4.4 Program LLMs
- Programmatic prompting frameworks (DSPy and similar)
- Optimizing prompts/pipelines against a metric automatically

**Exercise:** Build a query router that sends questions to either a SQL database or a vector store depending on intent, then synthesizes a cited answer.

**✅ Checklist**
- [ ] Implemented text-to-SQL or text-to-Cypher
- [ ] Added a tool-calling step to a RAG pipeline
- [ ] Added contextual compression / re-ranking
- [ ] Tried a programmatic prompt-optimization framework

---

## 5. 🕵️ Agents

**Goal:** Build systems that plan, act, and use tools autonomously.

### 5.1 Agent Fundamentals
- ReAct, Plan-and-Execute, and reflection loops
- Tool use, function calling, and action spaces
- Handling agent failure and retries

### 5.2 Agent Frameworks
- LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Anthropic's agent tooling
- Choosing a framework based on control vs. abstraction trade-offs

### 5.3 Multi-agents
- Orchestrator/worker patterns
- Debate and critique patterns between agents
- Shared state and communication protocols

**Exercise:** Build a multi-agent system where a "planner" agent delegates subtasks to "worker" agents and a "critic" agent reviews the final output.

**✅ Checklist**
- [ ] Built a single ReAct-style agent with tools
- [ ] Used at least one agent framework
- [ ] Built a multi-agent workflow with role specialization
- [ ] Added a self-critique / verification step

---

## 6. ⚙️ Inference Optimization

**Goal:** Understand what actually makes LLMs fast (or slow) and expensive (or cheap) to run.

### 6.1 Flash Attention
- Why standard attention is memory-bound, not just compute-bound
- How FlashAttention fuses operations to avoid materializing the full attention matrix

### 6.2 Key-Value Cache
- Why the KV cache dominates inference memory at scale
- Multi-query attention (MQA) and grouped-query attention (GQA)
- PagedAttention-style memory management (vLLM)

### 6.3 Speculative Decoding
- Using a small draft model to propose tokens, verified by the large model
- Trade-offs: latency gains vs. draft-model quality

**Exercise:** Benchmark tokens/sec for a model with and without a KV-cache-optimized serving engine (e.g., vLLM vs. naive HF `generate`).

**✅ Checklist**
- [ ] Explained why attention is O(n²) and what FlashAttention changes
- [ ] Measured KV cache memory growth with sequence length
- [ ] Benchmarked a serving engine with continuous batching
- [ ] Tried speculative decoding on a supported stack

---

## 7. 🌐 Deploying LLMs

**Goal:** Ship a model or LLM app that other people can actually use.

### 7.1 Local Deployment
- Serving via Ollama, llama.cpp server, LM Studio
- Building a simple local API wrapper

### 7.2 Demo Deployment
- Gradio, Streamlit, Chainlit for quick shareable demos
- Hugging Face Spaces for free hosting

### 7.3 Server Deployment
- Containerizing with Docker
- Serving frameworks: vLLM, TGI, Ray Serve, TensorRT-LLM
- Autoscaling, load balancing, observability (logging, tracing, cost tracking)

### 7.4 Edge Deployment
- Quantized formats: GGUF, AWQ, GPTQ
- On-device inference: mobile (MLC-LLM, Core ML), browser (WebLLM, transformers.js)

**Exercise:** Take one project from Chapters 3–5 and deploy it as a public demo, then containerize it for a "production" server target.

**✅ Checklist**
- [ ] Ran a model behind a local API
- [ ] Shipped a public demo (Spaces or similar)
- [ ] Containerized a serving stack with Docker
- [ ] Quantized and ran a model on a resource-constrained target

---

## 8. 🛡️ Securing LLMs

**Goal:** Understand how LLM systems fail under adversarial pressure, and how to defend them.

### 8.1 Prompt Hacking
- Prompt injection (direct and indirect, e.g., via retrieved documents)
- Jailbreaking techniques and why they work
- Data/prompt leaking attacks

### 8.2 Backdoors
- Data poisoning during fine-tuning
- Trigger-based backdoor attacks
- Supply-chain risks (untrusted model weights, LoRA adapters, plugins)

### 8.3 Defensive Measures
- Input/output filtering and guardrail frameworks (Llama Guard, NeMo Guardrails, etc.)
- Least-privilege tool access for agents
- Sandboxing code execution, human-in-the-loop for high-stakes actions
- Red-teaming your own pipeline

**Exercise:** Red-team your Chapter 4/5 project — attempt prompt injection via a poisoned document or tool result, then add a guardrail that blocks it.

**✅ Checklist**
- [ ] Reproduced a prompt injection against your own RAG/agent system
- [ ] Added input/output guardrails
- [ ] Applied least-privilege scoping to agent tools
- [ ] Documented a threat model for one project

---

## 🧩 Suggested Learning Path

```
Chapter 1 ──▶ Chapter 2 ──▶ Chapter 3 ──▶ Chapter 4
   (basics)     (retrieval foundation)     (RAG mastery)
                                                │
                                                ▼
                                          Chapter 5 (Agents)
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        ▼                       ▼                       ▼
                Chapter 6 (Speed)      Chapter 7 (Ship it)      Chapter 8 (Secure it)
```

Chapters 6–8 can be tackled in parallel once you have a working project from Chapters 3–5 to optimize, deploy, and secure.

---

## 🤝 Contributing

Contributions are very welcome:

1. Fork the repo and create a branch: `git checkout -b chapter-x-improvement`
2. Add notes, notebooks, or resources following the existing folder structure
3. Keep explanations original — cite sources, don't copy-paste from articles or docs
4. Open a PR with a clear description of what you added

Please open an issue first for large structural changes.

---

## 📜 License

This project is licensed under the **Apache-2.0 License** — see [`LICENSE`](./LICENSE) for details.

---

## ⭐ Acknowledgements

Built as a personal/community learning path through the modern LLM engineering stack — from first API call to secured production deployment. If this repo helps you, consider starring it and sharing your own notes back via PR.