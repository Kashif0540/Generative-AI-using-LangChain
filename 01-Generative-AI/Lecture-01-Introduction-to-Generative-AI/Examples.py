"""
==========================================================
Lecture 01 - Introduction to Generative AI
Examples.py

Author : Muhammad Kashif
Repository : Generative-AI-using-LangChain

Description:
This file contains simple Python examples related to the
fundamental concepts introduced in Lecture 01.

Topics:
1. AI Hierarchy
2. AI Learning Roadmap
3. Popular Generative AI Applications
4. Learning Strategy
==========================================================
"""

# ==========================================================
# Example 1 - AI Hierarchy
# ==========================================================

print("=" * 60)
print("Artificial Intelligence Hierarchy")
print("=" * 60)

ai_hierarchy = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Generative AI"
]

for level in ai_hierarchy:
    print("->", level)


print("\n")


# ==========================================================
# Example 2 - AI Engineering Roadmap
# ==========================================================

print("=" * 60)
print("AI Engineering Roadmap")
print("=" * 60)

roadmap = [
    "Python Programming",
    "Mathematics",
    "Machine Learning",
    "Deep Learning",
    "Transformers",
    "Large Language Models",
    "Prompt Engineering",
    "Embeddings",
    "Vector Databases",
    "Retrieval-Augmented Generation (RAG)",
    "AI Agents",
    "Deployment"
]

for step, topic in enumerate(roadmap, start=1):
    print(f"{step}. {topic}")

print("\n")


# ==========================================================
# Example 3 - Popular Generative AI Applications
# ==========================================================

print("=" * 60)
print("Popular Generative AI Applications")
print("=" * 60)

applications = {
    "ChatGPT": "Text Generation",
    "Claude": "Conversational AI",
    "Gemini": "Multimodal AI",
    "GitHub Copilot": "Code Generation",
    "Midjourney": "Image Generation"
}

for app, purpose in applications.items():
    print(f"{app:<20} : {purpose}")

print("\n")


# ==========================================================
# Example 4 - AI Learning Strategy
# ==========================================================

print("=" * 60)
print("Recommended Learning Strategy")
print("=" * 60)

strategy = [
    "Learn",
    "Practice",
    "Build",
    "Document",
    "Deploy",
    "Repeat"
]

for i in range(len(strategy) - 1):
    print(strategy[i], end="  ->  ")

print(strategy[-1])

print("\n")


# ==========================================================
# Example 5 - AI Domains
# ==========================================================

print("=" * 60)
print("Applications of Artificial Intelligence")
print("=" * 60)

domains = [
    "Healthcare",
    "Finance",
    "Education",
    "Cybersecurity",
    "Agriculture",
    "Autonomous Vehicles",
    "E-commerce",
    "Robotics"
]

for domain in domains:
    print("•", domain)

print("\n")


# ==========================================================
# Example 6 - Quick Knowledge Check
# ==========================================================

print("=" * 60)
print("Quick Knowledge Check")
print("=" * 60)

question = "Which technology powers most modern Large Language Models?"

answer = "Transformer Architecture"

print("Question:")
print(question)

print("\nAnswer:")
print(answer)

print("\n")


# ==========================================================
# End of File
# ==========================================================

print("=" * 60)
print("Lecture 01 Examples Completed Successfully!")
print("=" * 60)