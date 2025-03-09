---
layout: post
title: "Understanding Latency in Large Language Models"
categories: misc
---

When I first interacted with **ChatGPT**, I noticed a delay before it responded to my input. It seemed as if the chatbot was "thinking" before presenting its answer, and then it would generate the response word by word, much like a human. Unlike a search engine like Google, which returns a list of results almost instantaneously, **ChatGPT** requires some time to formulate and deliver a coherent response.

Curious about why this happens, I explored the reasons behind this latency. Here’s a simplified explanation of why this delay occurs and what happens behind the scenes.

## Types of Latency in LLMs

Latency in Large Language Models (**LLMs**) like **ChatGPT** can be broadly categorized into two types:

1. **First Token Latency**: The time it takes for the model to produce the first word of its response.
2. **Per Token Latency**: The subsequent delays between each word or token generated in the response.

These two latencies combine to create the overall response time we experience when interacting with the model.

<center>
  <img src="/assets/conversation-with-llm.png" alt="image" width="600" height="auto">
</center>

## Why Does Latency Happen?

To understand why these delays occur, it's important to know a bit about how **transformers**, the underlying architecture of **LLMs**, function. Transformers are powerful models that process input data through layers of complex mathematical operations to generate contextually appropriate outputs.

The inference process in these models can be divided into two distinct phases: `prefill` and `decode`.

<center>
  <img src="/assets/inferencing.png" alt="image" width="600" height="auto">
</center>

### 1. Prefill Phase

During the `prefill` phase, the model processes the input text to compute intermediate representations known as `keys` and `values`. These are used to generate the first output token. This phase is like the model "understanding" the question you've asked. The length of this phase depends on the complexity and length of your input, as the model needs to interpret and encode all the information before generating any response.

### 2. Decode Phase

Once the first token is generated, the model enters the decode phase. Here, it generates each subsequent token one by one. This is a sequential process where the model uses the context from the previous tokens to predict the next. The delay between each word is the **Per Token Latency**. The model continues this process until it reaches a stopping point, like a period or the end of your query.

### Why Not Instant Responses?

Unlike search engines, which retrieve pre-existing data, **LLMs** generate responses on the fly. Each word is created in real-time, based on the input and the model’s learned patterns. This generation process requires substantial computational power, especially for models with billions of parameters. This is why there is a delay before the response appears and why it doesn’t instantly produce a full page of text.

## Conclusion

Understanding the reasons behind **LLM** latency helps set realistic expectations when using these models. The delay you experience is not because the model is slow, but because it is carefully constructing each response to match your input contextually. As AI research progresses, we can expect improvements in reducing these latencies, making interactions even more seamless and human-like.