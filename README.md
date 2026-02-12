# TruthGuard-AI  
Transformer-based Multi-Class Fact Verification System (RoBERTa + Docker)

## Overview

TruthGuard-AI is an end-to-end NLP system designed to classify the factuality of textual statements into multiple levels of truthfulness using a fine-tuned RoBERTa transformer model.

The project implements a complete data-driven AI pipeline: data preprocessing, supervised fine-tuning, evaluation, reproducibility through Docker, and deployment as a production-ready FastAPI service.

---

## Business Use Case: Enterprise Chatbot Verification Layer

Many companies are deploying internal AI chatbots powered by LLMs to:

- Answer employee questions  
- Generate internal reports  
- Assist with documentation  
- Provide HR or policy information  

However, LLMs can generate partially incorrect or misleading statements.

TruthGuard-AI acts as a **verification layer** that evaluates the factual consistency of generated responses before they are delivered to employees.

### Example Production Pipeline

User → Enterprise Chatbot (LLM) → TruthGuard-AI Verification API → Final Response

The system assigns a veracity score across six levels:

- True  
- Mostly True  
- Half True  
- Barely True  
- False  
- Pants on Fire  

This enables:

- Risk reduction in internal communication  
- Increased reliability of AI-generated content  
- Prioritization of responses requiring human review  
- Stronger governance in enterprise AI systems  

---

## Technical Stack

- Model: RoBERTa (Transformer architecture)
- Framework: PyTorch + HuggingFace Transformers
- Dataset: LIAR dataset
- API: FastAPI
- Containerization: Docker & Docker Compose
- Reproducibility: Controlled random seeds and isolated environments

---

## Industrial Relevance

TruthGuard-AI demonstrates how transformer-based NLP systems can be integrated into enterprise AI infrastructures as a factual validation layer for LLM-powered applications.

It provides a realistic blueprint for deploying trustworthy AI systems in production environments.
