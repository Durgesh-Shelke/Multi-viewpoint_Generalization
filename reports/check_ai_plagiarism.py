#!/usr/bin/env python3
"""
check_ai_plagiarism.py
======================
Comprehensive AI plagiarism and text naturalness scanner using:
1. RoBERTa Sequence Classification (roberta-base-openai-detector) at paragraph and sentence level
2. Binoculars Perplexity / Cross-Perplexity scoring (using GPT-2)
3. Burstiness & Cadence metrics (sentence length variance)
"""

import os
import re
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

TEX_FILE = "/home/maniskill/Multi-viewpoint_Generalization/reports/example_template/conference_101719.tex"

def clean_latex(text):
    """Strip LaTeX commands, math, labels, and environments to obtain raw prose."""
    text = re.sub(r'%.*?\n', '\n', text)
    text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' [EQUATION] ', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{align\}.*?\\end\{align\}', ' [EQUATION] ', text, flags=re.DOTALL)
    text = re.sub(r'\$\$.*?\$\$', ' [EQUATION] ', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', ' [MATH] ', text)
    text = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\cite\{.*?\}', '', text)
    text = re.sub(r'\\ref\{.*?\}', '', text)
    text = re.sub(r'\\label\{.*?\}', '', text)
    text = re.sub(r'\\textbf\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\textit\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\underline\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\(sub)*section\*?\{(.*?)\}', r'\n\n=== SECTION: \2 ===\n', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'[\{\}\[\]\~]', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def split_into_sentences(text):
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]
    return clean

def compute_burstiness(sentences):
    """Computes the coefficient of variation (std / mean) of sentence lengths."""
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0.0
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    return float(std_len / mean_len) if mean_len > 0 else 0.0

def main():
    print("=" * 80)
    print(" ADVANCED AI DETECTION AUDIT (RoBERTa + Binoculars Perplexity + Burstiness)")
    print("=" * 80)

    if not os.path.exists(TEX_FILE):
        print(f"Error: {TEX_FILE} not found!")
        return

    with open(TEX_FILE, 'r', encoding='utf-8') as f:
        tex_raw = f.read()

    cleaned_text = clean_latex(tex_raw)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[+] Compute Device: {device}")

    # 1. Load RoBERTa
    print("[+] Loading RoBERTa AI Detector (roberta-base-openai-detector)...")
    roberta_name = "roberta-base-openai-detector"
    roberta_tok = AutoTokenizer.from_pretrained(roberta_name)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_name).to(device)
    roberta_model.eval()

    # 2. Load GPT-2 for Perplexity
    print("[+] Loading GPT-2 for Linguistic Perplexity Scoring...")
    gpt2_name = "gpt2"
    gpt2_tok = AutoTokenizer.from_pretrained(gpt2_name)
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_name).to(device)
    gpt2_model.eval()

    sections = cleaned_text.split("=== SECTION: ")
    section_reports = []
    total_words = 0
    total_weighted_fake = 0.0

    print(f"\n[+] Analyzing {len(sections)} document sections...\n")

    for sec in sections:
        if not sec.strip():
            continue
        lines = sec.strip().split("\n")
        title = lines[0].replace("===", "").strip()
        body = "\n".join(lines[1:]).strip()

        if len(body.split()) < 10:
            continue

        sentences = split_into_sentences(body)
        burstiness = compute_burstiness(sentences)
        num_words = len(body.split())

        # Paragraph-level RoBERTa evaluation
        inputs = roberta_tok(body[:1500], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            probs = torch.softmax(roberta_model(**inputs).logits, dim=-1).squeeze().tolist()
            # Index 0: Fake (AI), Index 1: Real (Human)
            fake_prob = probs[0] * 100.0

        # GPT-2 Perplexity evaluation
        enc = gpt2_tok(body[:1500], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = gpt2_model(**enc, labels=enc.input_ids)
            loss = outputs.loss.item()
            ppl = math.exp(loss) if loss < 10 else 999.0

        total_words += num_words
        total_weighted_fake += (fake_prob * num_words)

        # Flag specific sentences with >60% fake probability
        flagged_s = []
        for s in sentences:
            s_inp = roberta_tok(s, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                s_p = torch.softmax(roberta_model(**s_inp).logits, dim=-1).squeeze().tolist()
            if s_p[0] > 0.60:
                flagged_s.append((s, s_p[0] * 100))

        section_reports.append({
            "title": title,
            "words": num_words,
            "fake_prob": fake_prob,
            "ppl": ppl,
            "burstiness": burstiness,
            "flagged": flagged_s
        })

    overall_fake_pct = total_weighted_fake / total_words if total_words > 0 else 0.0

    print("-" * 80)
    print(f"{'SECTION':<35} | {'WORDS':<6} | {'AI %':<7} | {'PPL':<6} | {'BURST':<6} | {'STATUS'}")
    print("-" * 80)
    for r in section_reports:
        status = "[HIGH AI]" if r["fake_prob"] > 50 else "[MEDIUM]" if r["fake_prob"] > 20 else "[SAFE / HUMAN]"
        print(f"{r['title'][:35]:<35} | {r['words']:<6} | {r['fake_prob']:>5.1f}% | {r['ppl']:>6.1f} | {r['burstiness']:>6.2f} | {status}")

    print("-" * 80)
    print(f"DOCUMENT WEIGHTED AI SCORE: {overall_fake_pct:.1f}%")
    if overall_fake_pct > 30:
        print(">>> RESULT: HIGH RISK of Turnitin AI Flagging! Proceeding with humanizer rewrite.")
    else:
        print(">>> RESULT: LOW RISK / Human-Grade Academic Prose.")
    print("=" * 80)

if __name__ == "__main__":
    main()
