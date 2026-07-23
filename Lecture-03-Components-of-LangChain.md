# Lecture 03 - Components of LangChain

## Learning Objectives

After completing this lecture, I can:

- List all six core components of LangChain.
- Explain the Models component and why standardization across LLM providers was necessary.
- Differentiate Language Models from Embedding Models.
- Explain the Prompts component, including dynamic, role-based, and few-shot prompting.
- Explain the Chains component, including Sequential, Parallel, and Conditional chains.
- Explain the Indexes component and its four sub-parts (Document Loader, Text Splitter, Vector Store, Retriever).
- Explain why LLM API calls are stateless, and how the Memory component solves this.
- Explain the Agents component and how it differs from a simple chatbot.

---

# Recap of Lecture 02

Lecture 01 covered:

- LangChain is an open-source framework for building LLM-powered applications.
- The motivating example: building a PDF-chat application, and the system design behind it (multiple components, many interactions between them).
- LangChain's core advantage: it lets these components be **orchestrated efficiently**, producing a working pipeline with minimal code.
- **Chains** — connecting components so one's output automatically becomes the next one's input.
- **Model-agnostic design** — switching from one LLM provider to another takes barely any code change.
- Real-world use cases: conversational chatbots, AI knowledge assistants, and AI agents.

If any of this is unfamiliar, it's worth revisiting Lecture 01 first — this lecture builds directly on it.

A quick note on approach: this lecture (like the last one) is intentionally **conceptual, not code-heavy**. Most online LangChain resources jump straight into projects without first building a mental model of the library, which makes the code harder to actually understand. The plan here is: build the conceptual foundation first (Lectures 01–02), then move into hands-on coding and full projects.

---

# The Six Components of LangChain

```
1. Models
2. Prompts
3. Chains
4. Memory
5. Indexes
6. Agents
```

Understanding these six components covers the large majority of what LangChain is about. Every future video in this series maps back to one of these six. Let's go through them one at a time.

---

# 1. Models

> Models are the core interface through which you interact with AI models.

## The Backstory

In the history of NLP, one application everyone wanted to build was the **chatbot** — arguably the most popular NLP application ever attempted. But building one required solving two hard problems:

1. **Natural Language Understanding (NLU)** — actually understanding what the user means (e.g., "Hi, can you check my email").
2. **Context-aware text generation** — generating a sensible reply even after understanding the query.

Both problems were eventually solved together by **LLMs**, since they were trained on massive amounts of internet-scale data — which gave them both language understanding and generation capability in one shot.

## A New Problem: Size

Solving NLU and generation created a new problem: LLMs, having been trained on this much data, contain **billions of parameters**, making them enormous — good LLMs today are often **larger than 100GB**. No individual, and often no small company, can realistically host and run a file that large on their own hardware or cloud budget.

This was solved by **APIs**: providers like OpenAI and Anthropic host these massive models on their own servers and expose access through an API. Anyone can hit the API with a query, and get a response back — without ever running the model themselves. You only pay for what you use.

## A Third Problem: Non-Standardized APIs

Different LLM providers wrote their APIs differently. If you're a developer wanting to use, say, both OpenAI's GPT and Anthropic's Claude in the same application, you'd need to write **different code** for each — different SDKs, different call patterns, different response formats.

Example of the mismatch:

```python
# Talking to OpenAI's API
# ...uses the `openai` package, its own client/call pattern

# Talking to Anthropic's API (Claude)
# ...uses the `anthropic` package, a slightly different call pattern
```

If you build your app around one provider and later decide to switch (say, for cost reasons), you'd have to rewrite significant portions of your codebase. **Standardization became a real challenge.**

## LangChain's Solution: The Models Component

LangChain's Models component is an interface that lets you talk to **any** provider's AI model in a **standardized** way. Switching providers becomes a matter of changing one or two lines — the import/package name and the model name — while the rest of the calling code and the way you parse the response stays the same.

```python
# LangChain + OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")
response = model.invoke("Hello, how are you?")
print(response.content)

# LangChain + Anthropic (Claude) — nearly identical structure
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
response = model.invoke("Hello, how are you?")
print(response.content)
```

## Two Types of Models in LangChain

| Type | Input | Output | Use Case |
|------|-------|--------|----------|
| **Language Models (LLMs)** | Text | Text | Chatbots, AI agents, general text generation |
| **Embedding Models** | Text | Vector | Semantic search (covered in Lecture 01) |

LangChain's documentation lists many supported providers for both types — under Chat Models you'll find entries like Anthropic, Mistral AI, Azure, OpenAI, Vertex AI, Bedrock (AWS), Hugging Face, and more, along with which features each supports (tool calling, structured/JSON output, local execution, multimodal input). A similar list exists for embedding model providers (OpenAI, Mistral AI, IBM, Llama, and others).

**In short**: the Models component standardizes communication with AI models, so that a small code change lets you swap providers freely.

---

# 2. Prompts

> Prompts are the inputs provided to an LLM.

Whatever text you send into an LLM — like typing "What is Campus X?" into ChatGPT — is a prompt.

## Why Prompts Matter So Much

LLM output is **extremely sensitive** to prompt wording. Example: asking an LLM to "Explain Linear Regression in an academic tone" vs. "Explain Linear Regression in a fun tone" — changing just one word changes the output dramatically. This sensitivity has given rise to an entire field over the last two years: **Prompt Engineering** (and the job title **Prompt Engineer**) — a legitimate, important area of study around LLMs, even if it gets memed on social media.

Because prompts matter this much, LangChain provides a dedicated, flexible **Prompts** component. A few of the powerful patterns it enables:

## a) Dynamic & Reusable Prompts

Build a prompt template with placeholders instead of hardcoding text:

```
Summarize {topic} in {tone} tone.
```

A user might ask to summarize "Cricket" in a "fun" tone; another user later might ask for "Biology" in a "serious" tone. The same template gets reused, with placeholders filled in dynamically each time.

## b) Role-Based Prompts

Set up a system-level message defining a role, with a placeholder:

```
System: You are an experienced {profession}.
User: Tell me about {topic}.
```

One user might fill this as "an experienced doctor" + "viral fever"; another as "an experienced engineer" + "developing bridges." This guides the LLM to respond *as* that role.

## c) Few-Shot Prompting

Show the LLM a few labeled examples before asking it to classify something new — useful for tasks like a customer-support ticket classifier:

```
Classify the following customer support tickets into one of:
Billing Issue, Technical Problem, or General Inquiry.

Example 1: "I was charged twice for my subscription this month." -> Billing Issue
Example 2: "The app crashes every time I try to log in." -> Technical Problem
Example 3: "Can you explain how to upgrade my plan?" -> General Inquiry

New ticket: "<user's new ticket>" -> ?
```

Given the prior labeled examples, the LLM classifies the new ticket accordingly.

These are just a few examples of what's possible with LangChain's Prompts component — dedicated deep-dive videos on this will follow later in the series.

---

# 3. Chains

Chains are important enough that LangChain is literally **named after** this concept.

> Chains let you build **pipelines** in LangChain — where the output of one stage automatically becomes the input of the next, with no manual wiring required.

## Example: Sequential Chain

Task: take a ~1000-word English text as input, and output a Hindi summary in under 100 words.

Flow:

```
English Text Input
        │
        ▼
   LLM 1 -> Translates text into Hindi
        │
        ▼
   LLM 2 -> Summarizes the Hindi text (< 100 words)
        │
        ▼
   Final Hindi Summary
```

Without Chains, you'd manually call LLM 1, extract its output, manually feed it into LLM 2, and extract that output again. With a Chain, you simply provide the input once, call the chain, and the entire sequence executes automatically behind the scenes.

## Example: Parallel Chain

Task: given a topic (e.g., "9/11 incident"), generate a detailed report by combining the outputs of *multiple* LLMs.

```
              ┌──> LLM 1 -> Report A ──┐
Input Topic ──┤                        ├──> LLM 3 (combines A + B) ──> Final Combined Report
              └──> LLM 2 -> Report B ──┘
```

Here, LLM 1 and LLM 2 run **simultaneously** on the same input, and a third LLM merges their outputs.

## Example: Conditional Chain

Task: build an AI agent that collects user feedback and responds differently based on whether the feedback is positive or negative.

```
User Feedback
      │
      ▼
LLM evaluates sentiment
      │
   ┌──┴───────────────┐
   ▼                  ▼
Good feedback     Bad feedback
   │                  │
   ▼                  ▼
Say "Thank you"   Send an email alert
                  to the support team
```

The processing branches based on a condition — again, easily implemented with Chains.

LangChain's Chains component enables arbitrarily complex combinations of these patterns, dramatically reducing the amount of manual orchestration code needed. This will be explored in more depth in a later, dedicated lecture.

---

# 4. Indexes

> Indexes connect your application to external knowledge, such as PDFs, websites, and databases.

Indexes are made up of four sub-components:

```
1. Document Loader
2. Text Splitter
3. Vector Store
4. Retriever
```

## Why Indexes Are Needed

ChatGPT can answer most general questions because it was trained on huge amounts of internet data. But it **cannot** answer questions about private/organizational data it never saw during training — for example:

- "What is the leave policy of my company XYZ?"
- "What is the notice period policy of my company XYZ?"

ChatGPT has no way to know this, since this data was never part of its training set. The solution: **connect the LLM to an external knowledge source** — e.g., feed it your company's rulebook. Then general questions ("Who is the Prime Minister of India?") still get answered from the model's own knowledge, while company-specific questions get answered using the connected external source.

## How the Four Sub-Components Work Together

Using the same example — a 1000-page company rulebook stored on, say, Google Drive:

```
Rulebook (external source)
        │
        ▼
  Document Loader   -> loads the document into the system
        │
        ▼
  Text Splitter     -> breaks it into ~1000 page-level chunks
        │
        ▼
Embedding Model      -> generates a vector (embedding) for each chunk
        │
        ▼
  Vector Store       -> stores all ~1000 embedding vectors for later search
```

Then, when a query comes in ("What is the leave policy of company XYZ?"):

```
User Query
     │
     ▼
  Retriever  -> embeds the query, performs semantic search against
                the Vector Store, retrieves the most relevant chunks
     │
     ▼
Relevant chunks + original query -> sent to the LLM
     │
     ▼
Final Answer
```

**In short**: Indexes are the mechanism for building LLM applications that have access to an external knowledge source — which could be a PDF, a website, or a company database. This is the same underlying architecture discussed in Lecture 01's PDF-chat example, now named and broken into its four constituent LangChain sub-components.

---

# 5. Memory

## The Core Problem: LLM API Calls Are Stateless

Each call to an LLM API is **independent** — it has no memory of previous requests.

Example: ask an LLM "Who is Narendra Modi?" and it correctly explains he's an Indian politician and the current Prime Minister. Ask a follow-up in a **new API call** — "How old is he?" — and the model responds that it has no access to personal data about individuals unless it was shared in that request. It has no idea "he" refers to Narendra Modi, because **it has no memory of the earlier exchange.**

This is a serious problem for chatbots — imagine having to remind a chatbot what you were just talking about, every single message. LangChain's **Memory** component solves exactly this: it lets you add memory to an ongoing conversation.

## Types of Memory in LangChain

| Memory Type | How it Works | Trade-off |
|-------------|---------------|-----------|
| **Conversation Buffer Memory** | Stores the entire conversation so far; sends the full chat history with every new API call | Gets expensive/slow as the conversation grows |
| **Conversation Buffer Window Memory** | Stores only the last *N* interactions (e.g., last 100 messages), constantly updating | Bounded size, but older context is lost |
| **Summarizer-Based Memory** | Generates a summary of the conversation so far and sends that summary instead of the full history | Saves tokens/cost, at the cost of some detail |
| **Custom Memory** | Stores specialized, hand-picked pieces of information (e.g., user preferences, key facts) for advanced use cases | Highly flexible, but requires manual design |

This is a genuinely practical topic that will get deeper, hands-on coverage later in the series.

---

# 6. Agents

> Agents let you build AI agents — systems that don't just talk, but can also take action.

## From Chatbot to Agent

LLMs have two big strengths: **NLU** and **text generation**. The obvious first use case was the chatbot — and ChatGPT itself is essentially a chatbot built this way.

But if a chatbot understands you and can generate a good reply, could it also *do something* for you? Example:

- Ask a travel chatbot: "What's the best travel destination in India during summer?" → It answers from its trained knowledge (e.g., "Shimla" or "Manali" — hill stations).
- Ask further: "What's the cheapest flight from Delhi to Shimla on January 24th?" → An **AI agent** would hit a real API and fetch the actual answer.
- Go one step further: "Can you book the flight?" → The agent goes and **books it** for you.

**This is the core difference between a chatbot and an AI agent**: an agent is a chatbot with superpowers.

## What Makes an Agent Different

Two things a chatbot lacks, that an agent has:

1. **Reasoning capability** — the ability to break down a query and figure out what needs to be done, step by step.
2. **Access to tools** — the ability to call external tools/APIs to actually get things done.

## Worked Example

Suppose an AI agent is given two tools: a **calculator** and a **weather API**. A user asks:

> "Can you multiply today's temperature in Delhi by 3?"

Using a reasoning technique like **Chain of Thought**, the agent breaks this down step by step:

1. "I need today's temperature in Delhi first."
2. Checks its available tools → finds the weather API → calls it with input "Delhi."
3. Weather API returns: **25°C**.
4. "Now I need to multiply 25 by 3 — but for that I need a calculator."
5. Checks its tools again → finds the calculator → calls it with inputs `25`, `3`, `multiply`.
6. Calculator returns **75**.
7. **75** becomes the final answer.

This is how AI agents work: reasoning to figure out the plan, and tools to execute each step of that plan.

**Summary**: an AI agent is an evolved form of a chatbot — one that can perform actions, thanks to its reasoning capabilities and access to tools. This is a fast-moving area of AI research right now, and significant progress is expected in this space over the next year or two.

---

# Key Terms

| Term | Meaning |
|------|---------|
| Models | LangChain's standardized interface to communicate with any AI model provider |
| Language Model | Text-in, text-out model (an LLM) |
| Embedding Model | Text-in, vector-out model, used for semantic search |
| Prompts | The input(s) provided to an LLM; sensitive to wording |
| Prompt Engineering | The field of study around crafting effective prompts |
| Few-Shot Prompting | Providing labeled examples before asking the LLM to handle a new case |
| Chains | LangChain's mechanism for building pipelines where one stage's output auto-feeds the next stage's input |
| Sequential Chain | Stages executed one after another |
| Parallel Chain | Multiple stages executed simultaneously, then combined |
| Conditional Chain | Processing branches based on a condition |
| Indexes | LangChain's mechanism for connecting an LLM app to external knowledge sources |
| Document Loader | Loads documents into the system from an external source |
| Text Splitter | Breaks a document into smaller chunks |
| Vector Store | Stores embedding vectors for semantic search |
| Retriever | Embeds a query, performs semantic search, and returns relevant chunks |
| Stateless | Each API call is independent, with no memory of past calls |
| Memory | LangChain's component for maintaining conversational context across calls |
| Agents | LangChain's component for building systems that can reason and take action using tools |
| Reasoning | An agent's ability to break a task into steps and decide what to do next |
| Chain of Thought | A reasoning technique where a query is broken down step by step |

---

# Key Takeaways

- LangChain has **six core components**: Models, Prompts, Chains, Memory, Indexes, and Agents. Nearly everything else in the framework builds on these.
- **Models** solves the standardization problem across different LLM/embedding providers, so switching providers takes minimal code change.
- **Prompts** matter enormously because LLM output is highly sensitive to prompt wording; LangChain supports dynamic, role-based, and few-shot prompting patterns.
- **Chains** let you build pipelines (sequential, parallel, or conditional) where each stage's output automatically feeds the next stage's input — this is the concept LangChain is named after.
- **Indexes** (Document Loader + Text Splitter + Vector Store + Retriever) let an LLM application access external knowledge sources it wasn't trained on — like a company's private documents.
- **Memory** solves the problem of LLM API calls being stateless, using strategies like buffer memory, windowed memory, summarizer-based memory, or custom memory.
- **Agents** extend chatbots with reasoning and tool access, letting them actually take actions instead of just replying — this is considered one of the most active frontiers in AI right now.

---

# What's Next?

The next lecture will take a deep dive into the **first component — Models** — and start moving toward hands-on, practical implementation.
