# Lecture 01 - Introduction to LangChain

## Learning Objectives

After completing this lecture, I can:

- Explain what LangChain is, in one sentence.
- Explain **why** LangChain was needed, using a real system-design example.
- Describe the architecture of a PDF Chat application, from upload to answer.
- Explain Semantic Search and how it differs from Keyword Search.
- Understand Embeddings as the foundation of Semantic Search.
- List the three biggest challenges in building an LLM-powered app, and how each is solved.
- Explain the concrete benefits LangChain provides (Chains, model-agnostic design, ecosystem, memory).
- List real-world applications built with LangChain.
- Name LangChain's main alternatives.

---

# What is LangChain?

In one sentence:

> LangChain is an **open-source framework for developing applications powered by LLMs**.

If you want to build any LLM-based application, LangChain is the framework that helps you build it.

But a one-line definition doesn't really convey *why* LangChain matters. The right way to understand any tool `X` is to first understand **why `X` was needed in the first place**. So before defining LangChain further, this lecture builds up the *need* for it from scratch, using a real example.

---

# The Origin Story: A PDF Chatbot Idea

Around 2014–2015, as smartphones became affordable and people started reading more PDFs than books, an idea came up:

> What if there was an application where anyone could upload a PDF, read it, **and also chat with it**?

Example: upload a *Machine Learning* textbook, and then:

- "Explain page 5 like I'm 5 years old."
- "Generate true/false questions on Linear Regression for me to practice."
- "Give me notes on the Decision Trees section."

This is powerful because you're not just reading the document — you're **having a conversation with it**.

This idea is essentially what tools like ChatPDF and Google NotebookLM do today. Let's break down how such a system is actually built.

---

# High-Level System Design

Here's how the system behaves end to end:

1. User uploads a PDF → it gets stored in a database.
2. User opens the PDF and asks a query, e.g. **"What are the assumptions of Linear Regression?"**
3. The system must find *where in the document* this topic is discussed.

This step is a **search operation** — and there are two ways to do it.

## Keyword Search vs. Semantic Search

**Keyword Search**: look for the literal words — "assumptions", "linear", "regression" — anywhere they appear in the book. This is inefficient: the word "assumptions" alone might appear on dozens of unrelated pages, so you get a lot of noisy, low-relevance results.

**Semantic Search**: instead of matching words, understand the *meaning* of the query and retrieve pages that are actually about that meaning. This returns fewer pages, but far more relevant ones.

Say semantic search returns **page 372** and **page 461** as the two pages discussing the assumptions of Linear Regression.

## Combining Query + Retrieved Pages

The original user query and the retrieved pages are combined into a **system query**, which is sent to the most important component of the whole app — referred to here as the **"Brain."**

The Brain has exactly two jobs:

1. **Natural Language Understanding (NLU)** — understand the query itself, regardless of whether it's in English, Hindi, or any other language.
2. **Context-aware text generation** — read the given pages and generate a relevant answer from them, not from its own general knowledge.

So: understand the question → search only within the given pages → generate the answer. That's the full loop.

### Why not just hand the entire book to the Brain?

A fair question: if the Brain can understand a query and extract an answer from given pages, why not skip semantic search entirely and just feed it the *whole* 1000-page book?

Analogy: imagine you have a doubt in your math textbook.

- **Scenario A**: you go to your teacher and hand over the entire book, saying "I have a doubt somewhere in here."
- **Scenario B**: you tell your teacher, "I have a doubt on page 155 specifically."

Scenario B obviously gets you a faster, more accurate answer. The same logic applies here — handing the Brain two highly relevant pages (instead of the entire book) is both **computationally cheaper** and produces **better quality answers**. This is exactly why the semantic search step exists.

---

# How Semantic Search Actually Works

Say you have three paragraphs — about cricketers Virat Kohli, Jasprit Bumrah, and Rohit Sharma — and a question: **"How many runs has Virat scored?"**

A human instantly knows the answer is in the Virat Kohli paragraph. But how does code figure that out?

### Step 1: Convert everything into Embeddings

Every paragraph is converted into an **embedding** — a vector (a list of numbers) that represents the *semantic meaning* of the text. Techniques like Word2Vec, Doc2Vec, or BERT embeddings can generate these. Say each vector has 100 dimensions.

```
Paragraph (Virat)   -> Vector A (100 dimensions)
Paragraph (Bumrah)  -> Vector B (100 dimensions)
Paragraph (Rohit)   -> Vector C (100 dimensions)
```

### Step 2: Convert the query into an embedding too

```
Query: "How many runs has Virat scored?" -> Vector Q (100 dimensions)
```

### Step 3: Compare via similarity

Now there are 4 vectors sitting in the same 100-dimensional space. Compute the similarity between the query vector and each of the three paragraph vectors. Whichever paragraph vector is **closest** to the query vector is the one that's actually relevant — here, the Virat Kohli paragraph — and that paragraph is used to generate the answer.

This exact mechanism is what needs to be implemented over the PDF in the app.

---

# Detailed System Design (Low-Level)

Putting it all together, here's the full pipeline:

```
User uploads PDF
        │
        ▼
Stored in cloud storage (e.g. AWS S3)
        │
        ▼
Document Loader (loads PDF into the system)
        │
        ▼
Text Splitter (splits into chunks — by chapter, page, or paragraph)
        │
        ▼
Embedding Model (generates a vector for each chunk)
        │
        ▼
Vector Database (stores all the chunk embeddings)
```

Then, when the user asks a query:

```
User Query
        │
        ▼
Same Embedding Model generates the query's vector
        │
        ▼
Compare query vector against all stored vectors (distance/similarity)
        │
        ▼
Return Top-K closest vectors -> extract their corresponding pages
        │
        ▼
Combine original query + retrieved pages -> "system query"
        │
        ▼
Send to the Brain (LLM): NLU + context-aware text generation
        │
        ▼
Final Answer shown to the user
```

Example: with a 1000-page PDF split page-by-page, that's 1000 chunks → 1000 embeddings stored in the vector database. A new query gets embedded the same way, compared against all 1000 vectors, and (say) the top 5 closest ones are returned.

---

# The Three Big Challenges of Building This System

## Challenge 1: Building the "Brain"

The Brain needs to (a) deeply understand any query, and (b) generate relevant, context-aware text from retrieved pages. Both are genuinely hard NLP problems — real breakthroughs only came in **2017 with the Transformers paper**, followed by BERT and GPT.

**Good news**: this challenge is already solved. LLMs already exist with both NLU and context-aware generation capability — you just need to *use* one, rather than build one from scratch.

## Challenge 2: Computation / Hosting the LLM

LLMs are enormous deep learning models trained on huge amounts of internet data. Hosting one yourself means heavy infrastructure engineering and very high cost.

**Good news**: this is also solved — companies like OpenAI and Anthropic host these massive models on their own servers and expose them via an **API**. You simply call the API, pay only for what you use, and never have to manage the underlying infrastructure yourself. So instead of "using an LLM," in practice you're "using an LLM API."

## Challenge 3: Orchestrating the Entire System

This is the challenge LangChain directly addresses. Even after solving Challenges 1 and 2, the system still has many **moving components**:

```
1. AWS S3 (document storage)
2. Text Splitter (a model in itself)
3. Embedding Model
4. Vector Database
5. LLM
```

...and multiple tasks that must run in sequence: load → split → embed → store → retrieve → generate. Wiring all of this together **by hand** is genuinely difficult — and it gets worse over time. For example, if you later decide to switch from OpenAI's API to Google's API (for cost reasons), or switch vector databases, hand-written code means rewriting large chunks of your pipeline.

**This is exactly where LangChain comes in.** It gives you built-in, plug-and-play functionality to connect all these components together — and swapping one component (say, one LLM provider for another) becomes a matter of changing a couple of lines of code, not rewriting the whole system.

---

# Benefits of LangChain

## 1. Chains

The core idea LangChain is even named after. A **Chain** lets you connect multiple components/tasks into a pipeline, where the output of one component automatically becomes the input of the next — no manual wiring required. You can build:

- **Sequential chains** (step after step)
- **Parallel chains**
- **Conditional chains**

This lets you express arbitrarily complex pipelines (load → split → embed → store → retrieve → generate) very expressively.

## 2. Model-Agnostic Development

Swap the LLM, the embedding model, or the vector database with minimal code changes. Your core business logic stays untouched regardless of which underlying provider or component you use.

## 3. A Complete Ecosystem

LangChain provides interfaces for practically every component you'd need:

- Document loaders for any source (cloud, local files, PDFs, etc.)
- ~50 kinds of text splitters
- Many embedding models
- Many vector databases

Whatever component your company wants to use, LangChain likely already has an interface for it.

## 4. Memory & State Handling

Example: a user asks "What are the assumptions of Linear Regression?" and gets an answer. Then they ask, "Also give me a few interview questions on this ML algorithm" — without repeating "Linear Regression." Without memory, the system has no idea what "this algorithm" refers to.

LangChain solves this with a **conversation memory** concept, so the system can track context across turns without the user having to restate it every time.

---

# What Can You Build With LangChain?

1. **Conversational Chatbots** — the most popular use case. Internet-scale companies (food delivery, e-commerce, etc.) face a scale problem when talking to customers; a chatbot handles the first layer of communication and escalates to a human only when needed.

2. **AI Knowledge Assistants** — like a chatbot, but grounded in your own data. Example: a chatbot embedded in a course platform that a student can ask questions of *while watching a specific lecture*, using that lecture's content as its knowledge base.

3. **AI Agents** — "chatbots on steroids." Agents don't just talk, they *act* — e.g., an AI agent on a travel-booking platform that a less tech-savvy user can simply tell "book me the cheapest flight from X to Y on this date," and the agent executes the whole task using tools.

4. **Workflow Automation** — automating workflows at a personal, professional, or company level using LLMs.

5. **Summarization & Research Helpers** — tools to simplify research papers or long books. Useful especially because you often can't upload very large documents (context length limits) or private company data into a general tool like ChatGPT — but a company can use LangChain to build its own internal "ChatGPT-like" tool that processes arbitrarily large documents *and* keeps private data in-house.

---

# Alternatives to LangChain

LangChain is not the only framework for building LLM-powered applications. Two other popular ones:

| Framework | Notes |
|-----------|-------|
| **LlamaIndex** | Slightly more popular in some circles; a dedicated course exists on it as well |
| **Haystack** | A similar kind of library/framework for building LLM applications |

Choosing between them depends on pricing, team familiarity, and which tool fits a given use case best. A detailed comparative study (pros/cons of LangChain vs. LlamaIndex vs. Haystack) is left for later, once LangChain itself is well understood.

---

# Key Terms

| Term | Meaning |
|------|---------|
| LLM | Large Language Model |
| LangChain | Open-source framework for building LLM-powered apps |
| The "Brain" | Informal name for the LLM component that understands queries and generates context-aware answers |
| NLU | Natural Language Understanding |
| Embedding | Vector (numeric) representation of text's meaning |
| Vector Database | Stores embeddings for fast similarity search |
| Semantic Search | Search based on meaning, not exact words |
| Chain | A pipeline connecting multiple components/tasks together |
| Model-Agnostic | Ability to swap underlying models/providers with minimal code change |
| Agent | An LLM-powered system that can take actions, not just respond |

---

# Key Takeaways

- LangChain exists to solve the **orchestration problem**: wiring together document loaders, text splitters, embedding models, vector databases, and LLMs into one working pipeline.
- The two hardest underlying problems — understanding language and generating context-aware text — are already solved by modern LLMs; hosting/scaling them is already solved by LLM APIs (OpenAI, Anthropic, etc.).
- What's left, and what's genuinely hard, is connecting all the moving pieces together and keeping that wiring flexible — that's LangChain's job.
- Semantic Search (via embeddings + similarity) is far more efficient and accurate than keyword search, and giving an LLM only the relevant chunks (not the whole document) is both cheaper and more accurate.
- LangChain's core value: Chains (pipeline building), model-agnostic swapping, a broad component ecosystem, and built-in memory/state handling.
- Popular applications: conversational chatbots, AI knowledge assistants, AI agents, workflow automation, and summarization/research tools.
- LlamaIndex and Haystack are the two other major frameworks in this space.

---

# What's Next?

In the next lecture, we'll explore the **complete ecosystem of LangChain** in more depth before moving into hands-on coding and building actual LLM-powered applications.
