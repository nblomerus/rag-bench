#!/usr/bin/env python3
"""
scrape_arxiv.py — Download AI/ML research papers from ArXiv API.

Run this on your server (not in a sandbox). It will:
1. Query ArXiv for papers across 18 AI/ML categories
2. Download full PDF text via PyMuPDF or fallback to abstracts
3. Save everything as JSON ready for the RAG pipeline

Usage:
    pip install arxiv pymupdf requests tqdm
    python scrape_arxiv.py                          # Default: ~170 landmark papers
    python scrape_arxiv.py --mode extended           # ~1500+ papers across all topics
    python scrape_arxiv.py --mode abstracts           # ~5000+ abstracts only (fast)
    python scrape_arxiv.py --output /path/to/data     # Custom output directory
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import arxiv
import pymupdf
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Landmark papers — ~170 hand-curated papers every ML engineer should know
# Organized by category, spanning 2017–2025
# ═══════════════════════════════════════════════════════════════════════════
LANDMARK_PAPERS = [
    # ── 1. Transformer Architecture & Foundation Models ────────────────────
    "1706.03762",  # Attention Is All You Need (Vaswani et al., 2017)
    "1810.04805",  # BERT (Devlin et al., 2018)
    "2005.14165",  # GPT-3: Language Models are Few-Shot Learners (Brown et al., 2020)
    "1910.10683",  # T5: Exploring the Limits of Transfer Learning (Raffel et al., 2019)
    "2204.02311",  # PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)
    "2305.10403",  # PaLM 2 Technical Report (Google, 2023)
    "2303.08774",  # GPT-4 Technical Report (OpenAI, 2023)
    "2307.09288",  # Llama 2 (Touvron et al., 2023)
    "2310.06825",  # Mistral 7B (Jiang et al., 2023)
    "2312.11805",  # Gemini: A Family of Highly Capable Multimodal Models (Google, 2023)
    "2403.08295",  # Gemma: Open Models Based on Gemini Research (Google, 2024)
    "2407.21783",  # Llama 3: The Llama 3 Herd of Models (Meta, 2024)
    "2404.14219",  # Phi-3 Technical Report (Microsoft, 2024)
    "2405.04434",  # DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM (2024)
    "2402.00838",  # OLMo: Accelerating the Science of Language Models (AI2, 2024)
    "2407.10671",  # Qwen2 Technical Report (Alibaba, 2024)
    "2408.00118",  # Gemma 2: Improving Open LMs at Practical Size (Google, 2024)
    "2412.15115",  # Qwen2.5 Technical Report (Alibaba, 2024)
    "2412.08905",  # Phi-4 Technical Report (Microsoft, 2024)
    "2412.19437",  # DeepSeek-V3 Technical Report (DeepSeek, 2024)
    "2402.16819",  # Nemotron-4 15B Technical Report (NVIDIA, 2024)
    "2406.11704",  # Nemotron-4 340B Technical Report (NVIDIA, 2024)
    "2311.16867",  # The RefinedWeb Dataset for Falcon LLM (TII, 2023)
    # ── 2. Scaling Laws & Training Dynamics ────────────────────────────────
    "2001.08361",  # Scaling Laws for Neural Language Models (Kaplan et al., 2020)
    "2203.15556",  # Chinchilla: Training Compute-Optimal LLMs (Hoffmann et al., 2022)
    "2206.07682",  # Emergent Abilities of Large Language Models (Wei et al., 2022)
    "2305.16264",  # Scaling Data-Constrained Language Models (Muennighoff et al., 2023)
    "2106.04560",  # Scaling Vision Transformers (Zhai et al., 2021)
    "2010.14701",  # Scaling Laws for Autoregressive Generative Modeling (Henighan et al., 2020)
    "2404.10102",  # Chinchilla Scaling: A Replication Attempt (Besiroglu et al., 2024)
    # ── 3. Alignment, RLHF & Preference Optimization ─────────────────────
    "2203.02155",  # InstructGPT (Ouyang et al., 2022)
    "2212.08073",  # Constitutional AI (Bai et al., 2022)
    "2305.18290",  # DPO: Direct Preference Optimization (Rafailov et al., 2023)
    "2204.05862",  # Training a Helpful and Harmless Assistant (Anthropic, 2022)
    "1707.06347",  # PPO: Proximal Policy Optimization (Schulman et al., 2017)
    "2309.00267",  # RLAIF: Scaling RL from AI Feedback (Google, 2023)
    "2402.01306",  # KTO: Model Alignment as Prospect Theoretic Optimization (2024)
    "2403.07691",  # ORPO: Monolithic Preference Optimization without Reference (2024)
    "2405.14734",  # SimPO: Simple Preference Optimization with a Reference-Free Reward (2024)
    "2310.16944",  # Zephyr: Direct Distillation of LM Alignment (HuggingFace, 2023)
    "2210.10760",  # Scaling Laws for Reward Model Overoptimization (Gao et al., 2022)
    "2310.12036",  # Self-Alignment with Instruction Backtranslation (Meta, 2023)
    # ── 4. Efficient Fine-Tuning & Adaptation ─────────────────────────────
    "2106.09685",  # LoRA: Low-Rank Adaptation (Hu et al., 2021)
    "2305.14314",  # QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
    "2101.00190",  # Prefix Tuning (Li & Liang, 2021)
    "1902.00751",  # Adapter Modules: Parameter-Efficient Transfer (Houlsby et al., 2019)
    "2402.09353",  # DoRA: Weight-Decomposed Low-Rank Adaptation (2024)
    "2309.12307",  # LongLoRA: Efficient Fine-Tuning of Long-Context LLMs (2023)
    "2310.05914",  # NEFTune: Noisy Embeddings for Fine-Tuning (2023)
    "2106.10199",  # BitFit: Simple Parameter-Efficient Fine-Tuning (Ben-Zaken et al., 2021)
    "2110.04366",  # Compacter: Efficient Low-Rank Hypercomplex Adapter (Karimi et al., 2021)
    "2205.05638",  # Few-Shot Parameter-Efficient Fine-Tuning (IA3) (Liu et al., 2022)
    # ── 5. Retrieval-Augmented Generation ─────────────────────────────────
    "2005.11401",  # RAG: Retrieval-Augmented Generation (Lewis et al., 2020)
    "2002.08909",  # REALM: Retrieval-Enhanced Language Model (Guu et al., 2020)
    "2112.04426",  # RETRO: Improving LMs by Retrieving from Trillions of Tokens (2021)
    "2208.03299",  # Atlas: Few-Shot Learning with Retrieval (Izacard et al., 2022)
    "2004.04906",  # DPR: Dense Passage Retrieval (Karpukhin et al., 2020)
    "2004.12832",  # ColBERT: Efficient and Effective Passage Retrieval (Khattab et al., 2020)
    "2212.10496",  # HyDE: Hypothetical Document Embeddings (Gao et al., 2022)
    "2310.11511",  # Self-RAG: Learning to Retrieve, Generate, and Critique (2023)
    "2401.15884",  # CRAG: Corrective Retrieval Augmented Generation (2024)
    "2401.18059",  # RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (2024)
    "2310.03214",  # FreshLLMs: Refreshing LLMs with Search Engine Augmentation (2023)
    "2404.16130",  # GraphRAG: From Local to Global (Microsoft, 2024)
    "2312.10997",  # RAG for LLMs: A Survey (Gao et al., 2023)
    "2307.03172",  # Lost in the Middle: How LMs Use Long Contexts (Liu et al., 2023)
    # ── 6. Long Context, Positional Encoding & State Space Models ─────────
    "2104.09864",  # RoPE / RoFormer (Su et al., 2021)
    "2108.12409",  # ALiBi: Train Short, Test Long (Press et al., 2021)
    "2310.01889",  # Ring Attention with Blockwise Transformers (2023)
    "2312.00752",  # Mamba: Linear-Time Sequence Modeling with Selective SSMs (Gu & Dao, 2023)
    "2405.21060",  # Mamba-2: Transformers are SSMs (Dao & Gu, 2024)
    "2305.13048",  # RWKV: Reinventing RNNs for the Transformer Era (Peng et al., 2023)
    "2403.19887",  # Jamba: A Hybrid Transformer-Mamba Language Model (AI21, 2024)
    "2309.00071",  # YaRN: Efficient Context Window Extension (Peng et al., 2023)
    "2302.10866",  # Hyena Hierarchy: Towards Larger Convolutional LMs (Poli et al., 2023)
    "2307.02486",  # LongNet: Scaling Transformers to 1B Tokens (Microsoft, 2023)
    "2307.08621",  # RetNet: Retentive Network (Sun et al., 2023)
    "2402.10171",  # Data Engineering for Scaling LMs to 128K Context (Fu et al., 2024)
    # ── 7. Diffusion Models & Generative Modeling ─────────────────────────
    "2006.11239",  # DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
    "2112.10752",  # Latent Diffusion / Stable Diffusion (Rombach et al., 2021)
    "2207.12598",  # Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)
    "2303.01469",  # Consistency Models (Song et al., 2023)
    "2212.09748",  # DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2022)
    "2205.11487",  # Imagen: Photorealistic Text-to-Image Diffusion (Saharia et al., 2022)
    "2307.01952",  # SDXL: Improving Latent Diffusion Models (Podell et al., 2023)
    "2011.13456",  # Score-Based Generative Modeling via SDEs (Song et al., 2020)
    "2210.02747",  # Flow Matching for Generative Modeling (Lipman et al., 2022)
    "2310.00426",  # PixArt-alpha: Fast Training of Diffusion Transformer (2023)
    # ── 8. Multi-Modal Models & Vision-Language ───────────────────────────
    "2103.00020",  # CLIP: Connecting Text and Images (Radford et al., 2021)
    "2304.08485",  # LLaVA: Visual Instruction Tuning (Liu et al., 2023)
    "2204.14198",  # Flamingo: A Visual Language Model (Alayrac et al., 2022)
    "2310.03744",  # LLaVA-1.5: Improved Baselines with Visual Instruction Tuning (2023)
    "2312.14238",  # InternVL: Scaling Up Vision Foundation Models (Chen et al., 2023)
    "2308.12966",  # Qwen-VL: A Versatile Vision-Language Model (Alibaba, 2023)
    "2311.03079",  # CogVLM: Visual Expert for Pretrained LMs (Tsinghua, 2023)
    "2303.15343",  # SigLIP: Sigmoid Loss for Language-Image Pre-Training (Google, 2023)
    "2209.06794",  # PaLI: A Jointly-Scaled Multilingual Language-Image Model (2022)
    "2305.18565",  # PaLI-X: On Scaling up a Multilingual Vision-Language Model (2023)
    # ── 9. Mixture of Experts ─────────────────────────────────────────────
    "2101.03961",  # Switch Transformer (Fedus et al., 2021)
    "2401.04088",  # Mixtral of Experts (Mistral AI, 2024)
    "2401.06066",  # DeepSeekMoE: Towards Ultimate Expert Specialization (2024)
    "2006.16668",  # GShard: Scaling Giant Models with Conditional Computation (Google, 2020)
    "2202.08906",  # ST-MoE: Designing Stable and Transferable Sparse Expert Models (2022)
    "1701.06538",  # Outrageously Large Neural Networks: The Sparsely-Gated MoE (Shazeer et al., 2017)
    # ── 10. Agents, Tool Use & Planning ───────────────────────────────────
    "2210.03629",  # ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)
    "2302.04761",  # Toolformer: LMs Can Teach Themselves to Use Tools (Schick et al., 2023)
    "2201.11903",  # Chain-of-Thought Prompting (Wei et al., 2022)
    "2305.10601",  # Tree of Thoughts: Deliberate Problem Solving (Yao et al., 2023)
    "2305.16291",  # Voyager: An Open-Ended Embodied Agent (Fan et al., 2023)
    "2203.11171",  # Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)
    "2303.17580",  # HuggingGPT: Solving AI Tasks with ChatGPT (Shen et al., 2023)
    "2307.13854",  # WebArena: A Realistic Web Environment for Agents (Zhou et al., 2023)
    "2308.03688",  # AgentBench: Evaluating LLMs as Agents (Liu et al., 2023)
    "2405.15793",  # SWE-Agent: Agent-Computer Interfaces (Yang et al., 2024)
    "2211.12588",  # Program of Thoughts Prompting (Chen et al., 2022)
    "2407.16741",  # OpenHands: AI Software Developers as Agents (Wang et al., 2024)
    # ── 11. Code Generation & Code LLMs (NEW) ─────────────────────────────
    "2107.03374",  # Codex / HumanEval: Evaluating LLMs Trained on Code (OpenAI, 2021)
    "2308.12950",  # Code Llama: Open Foundation Models for Code (Meta, 2023)
    "2402.19173",  # StarCoder 2 and The Stack v2 (BigCode, 2024)
    "2401.14196",  # DeepSeek-Coder (DeepSeek, 2024)
    "2203.07814",  # AlphaCode: Competition-Level Code Generation (DeepMind, 2022)
    "2310.06770",  # SWE-bench: Can LMs Resolve Real-World GitHub Issues? (2023)
    "2306.08568",  # WizardCoder: Empowering Code LLMs with Evol-Instruct (2023)
    "2406.11931",  # DeepSeek-Coder-V2: Breaking the Barrier (2024)
    "2409.12186",  # Qwen2.5-Coder Technical Report (Alibaba, 2024)
    # ── 12. Safety, Red-Teaming & Robustness (NEW) ────────────────────────
    "2202.03286",  # Red Teaming Language Models to Reduce Harms (Ganguli et al., 2022)
    "2312.06674",  # Llama Guard: LLM-Based Input-Output Safeguard (Meta, 2023)
    "2307.15043",  # Universal and Transferable Adversarial Attacks on LLMs (GCG) (2023)
    "2401.05566",  # Sleeper Agents: Training Deceptive LLMs (Anthropic, 2024)
    "2310.01405",  # Representation Engineering: A Top-Down Approach to AI Safety (2023)
    "2404.02151",  # Jailbreaking Leading Safety-Aligned LLMs (2024)
    "2402.05668",  # Comprehensive Assessment of Jailbreak Attacks (2024)
    # ── 13. Inference Optimization & Efficiency (NEW) ─────────────────────
    "2205.14135",  # FlashAttention: Fast and Memory-Efficient Attention (Dao et al., 2022)
    "2307.08691",  # FlashAttention-2: Faster Attention (Dao, 2023)
    "2211.17192",  # Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)
    "2306.00978",  # AWQ: Activation-Aware Weight Quantization (Lin et al., 2023)
    "2210.17323",  # GPTQ: Accurate Post-Training Quantization for GPT (Frantar et al., 2022)
    "2309.06180",  # vLLM: Efficient Memory Management with PagedAttention (Kwon et al., 2023)
    "2401.10774",  # Medusa: Simple LLM Inference Acceleration (Cai et al., 2024)
    "2211.10438",  # SmoothQuant: Accurate and Efficient Post-Training Quantization (Xiao et al., 2022)
    "2306.03078",  # SpQR: A Sparse-Quantized Representation for LLMs (2023)
    # ── 14. Small Language Models & On-Device AI (NEW) ────────────────────
    "2306.11644",  # Textbooks Are All You Need (Phi-1) (Gunasekar et al., 2023)
    "2401.02385",  # TinyLlama: An Open-Source Small Language Model (Zhang et al., 2024)
    "2402.14905",  # MobileLLM: Optimizing Sub-Billion LLMs for On-Device (Meta, 2024)
    "2305.11206",  # LIMA: Less Is More for Alignment (Zhou et al., 2023)
    # ── 15. Evaluation & Benchmarking (NEW) ───────────────────────────────
    "2009.03300",  # MMLU: Measuring Massive Multitask Language Understanding (Hendrycks et al., 2020)
    "2306.05685",  # MT-Bench and Chatbot Arena (Zheng et al., 2023)
    "2403.04132",  # Chatbot Arena: An Open Platform for Evaluating LLMs (2024)
    "2206.04615",  # BIG-Bench: Beyond the Imitation Game (Srivastava et al., 2022)
    "2211.09110",  # HELM: Holistic Evaluation of Language Models (Liang et al., 2022)
    "1905.07830",  # HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al., 2019)
    # ── 16. Synthetic Data & Data Curation (NEW) ──────────────────────────
    "2212.10560",  # Self-Instruct: Aligning LMs with Self-Generated Instructions (Wang et al., 2022)
    "2306.02707",  # Orca: Progressive Learning from Complex Explanation Traces (2023)
    "2304.12244",  # WizardLM: Empowering LLMs to Follow Complex Instructions (2023)
    "2306.01116",  # The RefinedWeb Dataset for Falcon LLM (Penedo et al., 2023)
    "2309.16609",  # Textbooks Are All You Need II: phi-1.5 (Microsoft, 2023)
    # ── 17. Knowledge Distillation & Compression (NEW) ────────────────────
    "1910.01108",  # DistilBERT: A Distilled Version of BERT (Sanh et al., 2019)
    "1503.02531",  # Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
    "2305.12870",  # Lion: Adversarial Distillation of Proprietary LLMs (2023)
    # ── 18. Reasoning & Test-Time Compute (NEW — 2024+) ──────────────────
    "2305.20050",  # Let's Verify Step by Step (Process Reward Models) (Lightman et al., 2023)
    "2403.09629",  # Quiet-STaR: LMs Can Teach Themselves to Think Before Speaking (2024)
    "2401.01335",  # SPIN: Self-Play Fine-Tuning (Chen et al., 2024)
    "2501.12948",  # DeepSeek-R1: Incentivizing Reasoning via RL (DeepSeek, 2025)
    "2412.16720",  # OpenAI o1 System Card (2024)
    # ── 19. Video Generation & World Models (NEW) ─────────────────────────
    "2408.06072",  # CogVideoX: Text-to-Video Diffusion with Expert Transformer (2024)
    "2402.17177",  # Sora: A Review on Background, Technology, Limitations (2024)
    "2412.20404",  # Open-Sora: Democratizing Efficient Video Production (2024)
    # ── 20. Speech & Audio Models (NEW) ───────────────────────────────────
    "2212.04356",  # Whisper: Robust Speech Recognition via Large-Scale Supervision (Radford et al., 2022)
    "2301.02111",  # VALL-E: Neural Codec Language Models for TTS (Microsoft, 2023)
    "2209.03143",  # AudioLM: A Language Modeling Approach to Audio (Google, 2022)
]

# ═══════════════════════════════════════════════════════════════════════════
# Extended search queries — 18 categories for broad, deep coverage
# Total max_results across all topics: ~1,400 (before dedup)
# With landmarks + search, extended mode targets ~1,500+ unique papers
# ═══════════════════════════════════════════════════════════════════════════
SEARCH_TOPICS = {
    # ── Original 10 categories (expanded) ─────────────────────────────────
    "transformers": {
        "query": (
            "cat:cs.CL AND ("
            "transformer architecture OR attention mechanism OR self-attention "
            "OR large language model OR foundation model"
            ")"
        ),
        "max_results": 2500,
    },
    "scaling_laws": {
        "query": (
            "cat:cs.LG AND ("
            "scaling laws OR compute optimal training OR emergent abilities "
            "OR neural scaling OR training efficiency"
            ")"
        ),
        "max_results": 1500,
    },
    "alignment_rlhf": {
        "query": (
            "cat:cs.CL AND ("
            "RLHF OR reinforcement learning human feedback OR preference optimization "
            "OR alignment OR DPO OR reward model OR constitutional AI"
            ")"
        ),
        "max_results": 1000,
    },
    "efficient_finetuning": {
        "query": (
            "cat:cs.CL AND ("
            "LoRA OR parameter efficient fine tuning OR adapter OR quantized fine tuning "
            "OR low-rank adaptation OR prompt tuning"
            ")"
        ),
        "max_results": 1250,
    },
    "rag_retrieval": {
        "query": (
            "cat:cs.CL AND ("
            "retrieval augmented generation OR dense passage retrieval OR grounded generation "
            "OR knowledge-grounded OR RAG OR retrieval-enhanced"
            ")"
        ),
        "max_results": 1500,
    },
    "long_context": {
        "query": (
            "cat:cs.CL AND ("
            "long context OR positional encoding OR state space model OR linear attention "
            "OR context window extension OR rotary embedding OR Mamba"
            ")"
        ),
        "max_results": 1000,
    },
    "diffusion": {
        "query": (
            "cat:cs.LG AND ("
            "diffusion model OR denoising score matching OR latent diffusion "
            "OR flow matching OR consistency model OR diffusion transformer"
            ")"
        ),
        "max_results": 1000,
    },
    "multimodal": {
        "query": (
            "cat:cs.CV AND ("
            "vision language model OR multimodal large language model "
            "OR contrastive learning CLIP OR visual instruction tuning"
            ")"
        ),
        "max_results": 1500,
    },
    "mixture_of_experts": {
        "query": (
            "cat:cs.LG AND ("
            "mixture of experts OR sparse MoE OR expert routing "
            "OR conditional computation OR expert parallelism"
            ")"
        ),
        "max_results": 1000,
    },
    "agents_reasoning": {
        "query": (
            "cat:cs.CL AND ("
            "LLM agent OR tool use language model OR chain of thought "
            "OR reasoning OR planning agent OR agentic"
            ")"
        ),
        "max_results": 1500,
    },
    # ── 8 NEW categories ─────────────────────────────────────────────────
    "code_generation": {
        "query": (
            "cat:cs.SE AND ("
            "code generation OR code language model OR program synthesis "
            "OR automated software engineering OR code completion"
            ") OR cat:cs.CL AND (code generation large language model)"
        ),
        "max_results": 1000,
    },
    "safety_alignment": {
        "query": (
            "cat:cs.CL AND ("
            "red teaming language model OR jailbreak OR adversarial attack LLM "
            "OR AI safety OR LLM guardrails OR harmful content detection"
            ")"
        ),
        "max_results": 1000,
    },
    "inference_optimization": {
        "query": (
            "cat:cs.LG AND ("
            "speculative decoding OR model quantization OR KV cache "
            "OR inference optimization OR efficient serving OR model compression"
            ") OR cat:cs.CL AND (efficient inference large language model)"
        ),
        "max_results": 1000,
    },
    "small_language_models": {
        "query": (
            "cat:cs.CL AND ("
            "small language model OR on-device language model OR efficient language model "
            "OR sub-billion parameter OR tiny language model OR mobile LLM"
            ")"
        ),
        "max_results": 1000,
    },
    "evaluation_benchmarks": {
        "query": (
            "cat:cs.CL AND ("
            "benchmark language model OR evaluation large language model "
            "OR LLM leaderboard OR chatbot arena OR holistic evaluation"
            ")"
        ),
        "max_results": 1000,
    },
    "synthetic_data": {
        "query": (
            "cat:cs.CL AND ("
            "synthetic data language model OR self-instruct OR data augmentation LLM "
            "OR instruction tuning data OR data curation"
            ")"
        ),
        "max_results": 1000,
    },
    "reasoning_test_time": {
        "query": (
            "cat:cs.CL AND ("
            "test-time compute OR process reward model OR step-by-step verification "
            "OR reasoning LLM OR chain of thought scaling"
            ")"
        ),
        "max_results": 500,
    },
    "video_generation": {
        "query": (
            "cat:cs.CV AND ("
            "text-to-video OR video diffusion model OR video generation "
            "OR world model video OR temporal generation"
            ")"
        ),
        "max_results": 1000,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PDF download and text extraction
# ═══════════════════════════════════════════════════════════════════════════
def download_pdf(arxiv_id: str, output_dir: Path) -> Path | None:
    """Download a PDF from ArXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = output_dir / f"{arxiv_id.replace('/', '_')}.pdf"

    if pdf_path.exists():
        logger.info(f"Using cached PDF: {arxiv_id}")
        return pdf_path

    max_retries = 8
    backoff = 10
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(pdf_url, timeout=30, stream=True)
            if resp.status_code == 404:
                logger.warning(f"PDF not found (404) for {arxiv_id}, skipping.")
                return None
            if resp.status_code == 429:
                logger.warning(
                    f"429 Too Many Requests for {arxiv_id}, retrying in {backoff} "
                    f"seconds (attempt {attempt}/{max_retries})..."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            resp.raise_for_status()

            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded PDF: {arxiv_id}")
            return pdf_path
        except Exception as e:
            logger.warning(f"Failed to download PDF for {arxiv_id} (attempt {attempt}/{max_retries}): {e}")
            # If the exception is a requests.HTTPError and status code is 404, skip
            if (
                isinstance(e, requests.HTTPError)
                and hasattr(e, "response")
                and getattr(e.response, "status_code", None) == 404
            ):
                logger.warning(f"PDF not found (404) for {arxiv_id}, skipping.")
                return None
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
    logger.error(f"Giving up downloading PDF for {arxiv_id} after {max_retries} attempts.")
    return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def extract_sections_from_text(text: str) -> dict[str, str]:
    """
    Split extracted paper text into sections based on common headers.
    Handles both markdown-style and plain text headers.
    Also handles split headers where section number and title are on separate lines.
    """
    if not text or len(text.strip()) < 100:
        return {"full_text": text}

    sections = {}
    current_section = "preamble"
    current_lines = []

    # Patterns for section headers
    header_patterns = [
        re.compile(r"^#{1,4}\s+(.+)$"),  # Markdown: ## Header
        re.compile(r"^(\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z\s]+)$"),  # Numbered: 1. Introduction, 3.2.1 Attention
        re.compile(r"^([A-Z][A-Z\s]{3,40})$"),  # ALL CAPS: INTRODUCTION
        re.compile(
            r"^(Abstract|Introduction|Related Work|Background|"
            r"Method(?:ology|s)?|Approach|Model|Architecture|"
            r"Experiment(?:s|al)?(?:\s+(?:Setup|Results))?|Results|"
            r"Discussion|Conclusion(?:s)?|Limitation(?:s)?|"
            r"Training|Evaluation|Analysis|Appendix)\s*$",
            re.IGNORECASE,
        ),  # Known section names
    ]

    # Standalone section number (PDF splits number from title across lines)
    section_number_re = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*$")

    lines = text.split("\n")
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        matched_header = None

        for pattern in header_patterns:
            m = pattern.match(stripped)
            if m:
                matched_header = m.group(1) if m.lastindex else stripped
                break

        # Check for split header: standalone section number + title on next line
        if not matched_header and section_number_re.match(stripped) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and next_line[0].isupper() and len(next_line) < 60 and not next_line.endswith("."):
                matched_header = f"{stripped} {next_line}"
                i += 1  # consume the title line too

        if matched_header and len(matched_header) < 80:
            # Save previous section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text and len(section_text) > 30:
                    sections[current_section] = section_text

            # Normalize section name
            current_section = re.sub(r"^[\d.]+\s*", "", matched_header)
            current_section = re.sub(r"[^a-zA-Z0-9\s]", "", current_section)
            current_section = current_section.lower().strip()
            current_section = re.sub(r"\s+", "_", current_section) or "unnamed"
            current_lines = []
        else:
            current_lines.append(lines[i])

        i += 1

    # Save final section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text and len(section_text) > 30:
            sections[current_section] = section_text

    return sections if sections else {"full_text": text}


def build_acronym_dict(text: str) -> dict[str, str]:
    """Extract acronym definitions from paper text."""
    acronyms = {}
    pattern = r"([A-Za-z][A-Za-z\s\-]{2,50})\s*\(([A-Z][A-Z0-9]{1,10})\)"
    for match in re.finditer(pattern, text):
        full_form = match.group(1).strip()
        acronym = match.group(2).strip()
        words = full_form.split()
        if len(words) >= 2:
            acronyms[acronym] = full_form
    return acronyms


def format_authors(authors: list[str], max_authors: int = 3) -> str:
    """Format author list for display."""
    if not authors:
        return "Unknown"
    last_names = []
    for author in authors[:max_authors]:
        parts = author.strip().split()
        if parts:
            last_names.append(parts[-1])
    if len(authors) > max_authors:
        return f"{last_names[0]} et al."
    elif len(last_names) == 1:
        return last_names[0]
    elif len(last_names) == 2:
        return f"{last_names[0]} and {last_names[1]}"
    else:
        return ", ".join(last_names[:-1]) + f", and {last_names[-1]}"


# ═══════════════════════════════════════════════════════════════════════════
# Main scraping functions
# ═══════════════════════════════════════════════════════════════════════════
def fetch_by_ids(
    arxiv_ids: list[str],
    download_pdfs: bool = True,
    pdf_dir: Path | None = None,
) -> list[dict]:
    """Fetch specific papers by ArXiv ID."""
    print(f"Fetching {len(arxiv_ids)} landmark papers...")

    class RobustArxivClient(arxiv.Client):
        def _request(self, url, method="GET", **kwargs):
            max_retries = 10
            backoff = 5
            for attempt in range(1, max_retries + 1):
                resp = arxiv.Client._request(self, url, method, **kwargs)
                if resp.status_code == 429:
                    logger.warning(
                        f"arxiv API 429 Too Many Requests, retrying in {backoff} "
                        f"seconds (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    continue
                return resp
            logger.error(f"arxiv API: Giving up after {max_retries} retries for {url}")
            return resp

    client = RobustArxivClient(
        page_size=20,
        delay_seconds=0.5,  # Reduced for speed (ArXiv handles well)
        num_retries=10,
    )

    docs = []
    search = arxiv.Search(id_list=arxiv_ids)

    # Prepare output path for incremental saving
    output_path = pdf_dir.parent / "scraped_papers.json" if pdf_dir else Path("scraped_papers.json")
    docs_by_id = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                for d in json.load(f):
                    if "arxiv_id" in d:
                        docs_by_id[d["arxiv_id"]] = d
        except Exception:
            pass

    for i, result in enumerate(tqdm(client.results(search), total=len(arxiv_ids), desc="Fetching papers")):
        arxiv_id = result.entry_id.split("/")[-1]
        arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

        # Check if PDF already exists before downloading
        pdf_path = pdf_dir / f"{arxiv_id_clean.replace('/', '_')}.pdf" if download_pdfs and pdf_dir else None
        if download_pdfs and pdf_dir and pdf_path and pdf_path.exists():
            full_text = extract_text_from_pdf(pdf_path)
            sections = extract_sections_from_text(full_text) if full_text else {}
        else:
            full_text = ""
            sections = {}
            if download_pdfs and pdf_dir:
                pdf_path = download_pdf(arxiv_id_clean, pdf_dir)
                if pdf_path:
                    full_text = extract_text_from_pdf(pdf_path)
                    if full_text:
                        sections = extract_sections_from_text(full_text)

        if not sections:
            sections = {"abstract": result.summary.strip()}
            full_text = result.summary.strip()

        year = result.published.year if result.published else None

        doc = {
            "doc_id": f"arxiv_{arxiv_id_clean}",
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "year": year,
            "arxiv_id": arxiv_id_clean,
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "full_text": full_text,
            "sections": sections,
            "acronyms": build_acronym_dict(full_text),
        }
        docs.append(doc)
        docs_by_id[arxiv_id_clean] = doc

        # Batch save every 50 papers instead of every paper
        if (i + 1) % 50 == 0:
            try:
                with open(output_path, "w") as outf:
                    json.dump(list(docs_by_id.values()), outf, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Could not update {output_path}: {e}")

    # Save remaining papers
    if docs_by_id:
        try:
            with open(output_path, "w") as outf:
                json.dump(list(docs_by_id.values()), outf, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not final save {output_path}: {e}")

    # No summary log
    return docs


def fetch_by_search(
    topics: dict,
    download_pdfs: bool = True,
    pdf_dir: Path | None = None,
) -> list[dict]:
    """Fetch papers by search query across topics."""

    class RobustArxivClient(arxiv.Client):
        def _request(self, url, method="GET", **kwargs):
            max_retries = 10
            backoff = 5
            for attempt in range(1, max_retries + 1):
                resp = arxiv.Client._request(self, url, method, **kwargs)
                if resp.status_code == 429:
                    logger.warning(
                        f"arxiv API 429 Too Many Requests, retrying in {backoff} "
                        f"seconds (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    continue
                return resp
            logger.error(f"arxiv API: Giving up after {max_retries} retries for {url}")
            return resp

    client = RobustArxivClient(
        page_size=50,
        delay_seconds=0.5,  # Reduced for speed
        num_retries=10,
    )

    all_docs = []
    seen_ids = set()

    # Progress file logic
    progress_file = pdf_dir.parent / "scrape_progress.json" if pdf_dir else Path("scrape_progress.json")
    try:
        with open(progress_file) as pf:
            progress = json.load(pf)
    except Exception:
        progress = {}

    # Prepare output path for incremental saving
    output_path = pdf_dir.parent / "scraped_papers.json" if pdf_dir else Path("scraped_papers.json")
    docs_by_id = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                for d in json.load(f):
                    if "arxiv_id" in d:
                        docs_by_id[d["arxiv_id"]] = d
        except Exception:
            pass

    for topic_name, topic_config in topics.items():
        if progress.get(topic_name, False):
            print(f"[SKIP] {topic_name}")
            continue
        print(f"\nCategory: {topic_name}")

        search = arxiv.Search(
            query=topic_config["query"],
            max_results=topic_config["max_results"],
            sort_by=arxiv.SortCriterion.Relevance,
        )

        topic_docs = []

        max_retries_topic = 6
        topic_failed = False
        for attempt_topic in range(1, max_retries_topic + 1):
            try:
                for i, result in enumerate(
                    tqdm(
                        client.results(search),
                        total=topic_config["max_results"],
                        desc=f"  {topic_name}",
                    )
                ):
                    arxiv_id = result.entry_id.split("/")[-1]
                    arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

                    if arxiv_id_clean in seen_ids:
                        continue
                    seen_ids.add(arxiv_id_clean)

                    full_text = ""
                    sections = {}

                    if download_pdfs and pdf_dir:
                        pdf_path = download_pdf(arxiv_id_clean, pdf_dir)
                        if pdf_path:
                            full_text = extract_text_from_pdf(pdf_path)
                            if full_text:
                                sections = extract_sections_from_text(full_text)

                    if not sections:
                        sections = {"abstract": result.summary.strip()}
                        full_text = result.summary.strip()

                    year = result.published.year if result.published else None

                    doc = {
                        "doc_id": f"arxiv_{arxiv_id_clean}",
                        "title": result.title.strip(),
                        "authors": [a.name for a in result.authors],
                        "year": year,
                        "arxiv_id": arxiv_id_clean,
                        "categories": result.categories,
                        "pdf_url": result.pdf_url,
                        "full_text": full_text,
                        "sections": sections,
                        "acronyms": build_acronym_dict(full_text),
                        "topic": topic_name,
                    }
                    topic_docs.append(doc)
                    docs_by_id[arxiv_id_clean] = doc

                    # Batch save every 50 papers instead of every paper
                    if (i + 1) % 50 == 0:
                        try:
                            with open(output_path, "w") as outf:
                                json.dump(list(docs_by_id.values()), outf, indent=2, default=str)
                        except Exception as e:
                            logger.warning(f"Could not update {output_path}: {e}")
                break  # Success, break out of retry loop
            except arxiv.HTTPError as e:
                # arxiv.HTTPError has .status attribute for HTTP code
                if getattr(e, "status", None) == 429:
                    wait_time = min(60 * attempt_topic, 600)
                    logger.warning(
                        f"HTTP 429 Too Many Requests for topic '{topic_name}', "
                        f"retrying in {wait_time} seconds (attempt {attempt_topic}/{max_retries_topic})..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"arxiv HTTPError for topic '{topic_name}': {e}")
                    topic_failed = True
                    break
            except Exception as e:
                logger.error(f"Unexpected error for topic '{topic_name}': {e}")
                topic_failed = True
                break

        if topic_failed:
            logger.error(f"Skipping topic '{topic_name}' after repeated failures.")
            progress[topic_name] = "failed"
        else:
            all_docs.extend(topic_docs)
            # Mark topic as completed in progress file
            progress[topic_name] = True

        try:
            with open(progress_file, "w") as pf:
                json.dump(progress, pf, indent=2)
        except Exception as e:
            logger.warning(f"Could not update progress file: {e}")

    # Final save to ensure all data is written
    if docs_by_id:
        try:
            with open(output_path, "w") as outf:
                json.dump(list(docs_by_id.values()), outf, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not final save {output_path}: {e}")

    # No total summary log
    return all_docs


def scrape_core(output_dir: Path, download_pdfs: bool = True) -> list[dict]:
    """Scrape the ~170 landmark papers (core corpus)."""
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    docs = fetch_by_ids(LANDMARK_PAPERS, download_pdfs=download_pdfs, pdf_dir=pdf_dir)
    return docs


def scrape_extended(output_dir: Path, download_pdfs: bool = True) -> list[dict]:
    """Scrape landmark papers + search-based papers (~1500+ total)."""
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Try to load existing output to avoid re-downloading landmark papers
    output_path = output_dir / "scraped_papers.json"
    docs = []
    seen_ids = set()
    if output_path.exists():
        try:
            with open(output_path) as f:
                docs = json.load(f)
            seen_ids = {d["arxiv_id"] for d in docs if "arxiv_id" in d}
            logger.info(f"Loaded {len(docs)} existing papers from {output_path}")
        except Exception as e:
            logger.warning(f"Could not load existing output: {e}")

    # Only fetch missing landmark papers
    missing_landmarks = [pid for pid in LANDMARK_PAPERS if pid not in seen_ids]
    if missing_landmarks:
        logger.info(f"Fetching {len(missing_landmarks)} missing landmark papers...")
        new_landmarks = fetch_by_ids(missing_landmarks, download_pdfs=download_pdfs, pdf_dir=pdf_dir)
        docs.extend(new_landmarks)
        seen_ids.update(d["arxiv_id"] for d in new_landmarks if "arxiv_id" in d)

    # Add search results (resumable)
    search_docs = fetch_by_search(SEARCH_TOPICS, download_pdfs=download_pdfs, pdf_dir=pdf_dir)
    for doc in search_docs:
        if doc["arxiv_id"] not in seen_ids:
            docs.append(doc)
            seen_ids.add(doc["arxiv_id"])

    return docs


def scrape_abstracts(output_dir: Path) -> list[dict]:
    """Scrape ~5000+ abstracts only (fast, no PDF download)."""
    # Increase max_results for abstract-only mode
    abstract_topics = {}
    for name, config in SEARCH_TOPICS.items():
        abstract_topics[name] = {
            "query": config["query"],
            "max_results": config["max_results"] * 5,
        }

    docs = fetch_by_ids(LANDMARK_PAPERS, download_pdfs=False)
    seen_ids = {d["arxiv_id"] for d in docs}

    search_docs = fetch_by_search(abstract_topics, download_pdfs=False)
    for doc in search_docs:
        if doc["arxiv_id"] not in seen_ids:
            docs.append(doc)
            seen_ids.add(doc["arxiv_id"])

    return docs


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Scrape AI/ML papers from ArXiv for RAG-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_arxiv.py                             # ~170 landmark papers with full text
  python scrape_arxiv.py --mode extended             # ~1500+ papers across 18 topics
  python scrape_arxiv.py --mode abstracts            # ~5000+ abstracts only (fast)
  python scrape_arxiv.py --output ~/rag-bench/data   # Custom output directory
  python scrape_arxiv.py --no-pdf                    # Skip PDF download (abstracts only)
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["core", "extended", "abstracts"],
        default="core",
        help="Scraping mode: core (~170 landmark papers), extended (~1500+), or abstracts (~5000+)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data"),
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF download; use abstracts only",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("RAG-Bench ArXiv Scraper")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {args.output}")
    logger.info(f"PDF download: {'disabled' if args.no_pdf else 'enabled'}")

    start = time.time()

    if args.mode == "core":
        docs = scrape_core(args.output, download_pdfs=not args.no_pdf)
    elif args.mode == "extended":
        docs = scrape_extended(args.output, download_pdfs=not args.no_pdf)
    elif args.mode == "abstracts":
        docs = scrape_abstracts(args.output)

    elapsed = time.time() - start

    # Save output
    output_path = args.output / "scraped_papers.json"
    with open(output_path, "w") as f:
        json.dump(docs, f, indent=2, default=str)

    # Also save as parsed_papers.json for direct pipeline compatibility
    compat_path = args.output / "parsed_papers.json"
    with open(compat_path, "w") as f:
        json.dump(docs, f, indent=2, default=str)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SCRAPING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Papers scraped: {len(docs)}")
    logger.info(f"Time elapsed: {elapsed:.1f}s")
    logger.info(f"Output: {output_path}")
    logger.info(f"Pipeline-ready: {compat_path}")

    # Stats
    with_fulltext = sum(1 for d in docs if len(d.get("full_text", "")) > 500)
    abstract_only = len(docs) - with_fulltext
    years = [d["year"] for d in docs if d.get("year")]

    logger.info("\nStats:")
    logger.info(f"  Full text: {with_fulltext} papers")
    logger.info(f"  Abstract only: {abstract_only} papers")
    if years:
        logger.info(f"  Year range: {min(years)} - {max(years)}")

    # Show topic distribution
    topics = {}
    for d in docs:
        t = d.get("topic", "landmark")
        topics[t] = topics.get(t, 0) + 1
    logger.info(f"  Topics: {json.dumps(topics, indent=4)}")

    logger.info(f"\nNext step: copy {compat_path} to your rag-bench/data/ folder")
    logger.info("Then run: python main.py")


if __name__ == "__main__":
    main()
