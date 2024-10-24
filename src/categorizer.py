from groq import Groq
import json
from arxiv_paper import get_arxiv_papers_with_id, ArxivPaper
from typing import Optional

MAX_SUB_CATEGORIES = 20


def get_categories_and_subcategories(
    summary: str,
    client: Optional[Groq] = None,
) -> tuple[str, list[str]]:
    """
    Get categories and sub-categories from the given summary.

    Parameters
    ----------
    summary : str
        The summary of the paper.

    Returns
    -------
    tuple[str, list[str]]
        The category and sub-categories of the paper.
    """

    if client is None:
        client = Groq()

    completion = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {
                "role": "system",
                "content": '# Objective\n\nYou are an AI assistant that helps categorize research papers into appropriate categories and sub-categories based on their content. For each paper or topic, identify:\n\n- **Category**: The broad area of research.\n- **Sub-categories**: Specific topics, methods, applications, or domains within that area.\n\nProvide the output in the following JSON format:\n\n{\n  "category": str,\n  "sub_categories": list[str]\n}\n\nIf multiple categories or sub-categories apply, include them all. Be consistent with terminology and ensure that categories are not too broad. You can provide up to 20 sub-categories but only 1 category.\n\n# Examples\n\n### 1. **Category: Machine Translation**\n- **Sub-categories**: Neural machine translation (NMT), rule-based translation, statistical machine translation, hybrid translation systems, transformer-based models, sequence-to-sequence models, attention mechanisms, domain adaptation in MT, multilingual translation, low-resource languages, unsupervised translation, zero-shot translation, bilingual evaluation understudy (BLEU), character-level translation, phrase-based translation, back-translation, post-editing, neural decoding strategies, translation memory systems, cross-lingual embeddings\n\n### 2. **Category: Recommender Systems**\n- **Sub-categories**: Collaborative filtering, matrix factorization, content-based filtering, hybrid recommender systems, deep learning-based recommendation, session-based recommendations, graph-based recommendations, context-aware recommendations, user-item embeddings, implicit feedback, cold-start problem, knowledge-based recommendations, reinforcement learning for recommendations, time-aware recommendations, explainable recommendations, personalization, adversarial learning in recommendation, cross-domain recommendations, sequence-based recommendations, diversity in recommendations\n\n### 3. **Category: Federated Learning**\n- **Sub-categories**: Privacy-preserving learning, distributed optimization, model aggregation, communication efficiency, differential privacy in federated learning, secure aggregation, heterogeneity handling, personalization in federated learning, federated transfer learning, client-server architecture, adversarial attacks in federated learning, robustness in federated systems, decentralized federated learning, cross-device learning, model compression for federated systems, on-device learning, federated multitask learning, federated reinforcement learning, client sampling strategies, fairness in federated learning\n\n### 4. **Category: Few-Shot Learning**\n- **Sub-categories**: Meta-learning, prototypical networks, task adaptation, contrastive learning, transfer learning, episodic training, metric-based learning, memory-augmented networks, relation networks, few-shot object detection, few-shot image classification, few-shot segmentation, zero-shot learning, hybrid models (few-shot & zero-shot), data augmentation for few-shot, embedding space learning, attention mechanisms, pretraining for few-shot learning, siamese networks, Bayesian few-shot learning\n\n### 5. **Category: Multi-modal Learning**\n- **Sub-categories**: Audio-visual learning, text-vision fusion, image-language models, cross-modal retrieval, multi-modal fusion, co-attention mechanisms, multi-task learning in multi-modal, multi-modal embeddings, speech-vision integration, multi-modal transformers, contrastive learning for multi-modal data, self-supervised learning in multi-modal, alignment across modalities, zero-shot learning in multi-modal, knowledge distillation for multi-modal, multi-modal summarization, visual question answering, video-language modeling, object-text relations, graph-based multi-modal learning\n\n### 6. **Category: Active Learning**\n- **Sub-categories**: Query-by-committee, uncertainty sampling, pool-based sampling, stream-based sampling, active learning for deep neural networks, cost-effective labeling, diversity-based sampling, human-in-the-loop systems, self-training, ensemble-based active learning, batch-mode active learning, active learning for NLP, active learning in vision tasks, multi-class active learning, adaptive active learning, clustering-based sampling, transfer learning in active learning, label noise handling, semi-supervised active learning, reinforcement learning for active learning\n\n### 7. **Category: Meta-Learning**\n- **Sub-categories**: Model-agnostic meta-learning (MAML), task-specific learning, few-shot optimization, gradient-based meta-learning, memory-augmented meta-learning, black-box optimization, meta-learning for reinforcement learning, multitask meta-learning, transfer learning in meta-learning, adversarial meta-learning, neural architecture search (NAS) for meta-learning, probabilistic meta-learning, meta-learning in robotics, unsupervised meta-learning, online meta-learning, multi-agent meta-learning, personalized meta-learning, meta-regularization, self-supervised meta-learning, curriculum meta-learning\n\n### 8. **Category: Bayesian Learning**\n- **Sub-categories**: Bayesian neural networks, variational inference, Monte Carlo methods, Bayesian optimization, Gaussian processes, Bayesian hierarchical models, Bayesian deep learning, approximate inference, probabilistic programming, Markov chain Monte Carlo (MCMC), Bayesian nonparametrics, Bayesian model selection, Bayesian reinforcement learning, Bayesian decision theory, priors in Bayesian learning, posterior distributions, uncertainty quantification, amortized inference, variational autoencoders (VAEs), Bayesian belief networks\n\n### 9. **Category: Self-Supervised Learning**\n- **Sub-categories**: Contrastive learning, predictive coding, representation learning, masked language modeling, BERT-like architectures, pretext tasks, instance discrimination, mutual information maximization, self-distillation, self-supervised learning for vision, self-supervised learning for audio, time-contrastive learning, cross-modal self-supervision, self-supervised video representation learning, generative self-supervision, clustering-based self-supervision, self-supervised reinforcement learning, self-supervised transfer learning, multi-task self-supervised learning, negative sampling strategies\n\n### 10. **Category: Graph Representation Learning**\n- **Sub-categories**: Graph embeddings, link prediction, graph clustering, graph pooling, graph neural networks (GNNs), graph convolutional networks (GCNs), node classification, attention-based GNNs, graph autoencoders, graph isomorphism networks (GINs), spectral methods in graphs, graph transformers, hierarchical graph representations, spatial graph learning, graph kernel methods, dynamic graph learning, inductive graph learning, multi-relational graphs, semi-supervised learning on graphs, graph signal processing',
            },
            {
                "role": "user",
                "content": summary,
            },
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    result = json.loads(completion.choices[0].message.content)

    return result["category"], result["sub_categories"][:MAX_SUB_CATEGORIES]


def get_categorized_papers(paper_ids: list[str]) -> list[ArxivPaper]:
    """
    Get categorized papers with given IDs.

    Parameters
    ----------
    paper_ids : list[str]
        A list of arXiv paper IDs.

    Returns
    -------
    list[ArxivPaper]
        A list of ArxivPaper objects.
    """

    papers = get_arxiv_papers_with_id(paper_ids)

    for paper in papers:
        category, sub_categories = get_categories_and_subcategories(paper.summary)

        paper.category = category
        paper.sub_categories = sub_categories

    return papers
