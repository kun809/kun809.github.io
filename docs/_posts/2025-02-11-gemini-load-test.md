---
layout: post
title: "Load Testing Gemini with OpenText Performance Engineering Solutions"
categories: misc
---

# Gemini Load Testing Guide
This guide complements our existing [blog](https://community.opentext.com/devops-cloud/b/devops-blog/posts/load-testing-openai-vllm-with-opentext-performance-engineering-solutions) on load testing **OpenAI** and **vLLM** models. It focuses on the steps required to perform load tests on **Gemini**, detailing both Non-Streaming and Streaming modes.
## Non-Streaming Mode

### API Example

**Request:**

```http
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=YOUR_API_KEY
Content-Type: application/json

{
  "contents": [{
    "parts": [{"text": "Write a story of about 200 words"}]
  }]
}
```

**Response:**

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "The old lighthouse keeper, Silas, squinted at the churning grey sea...\n"
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "avgLogprobs": -0.19974330114939856
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 10,
    "candidatesTokenCount": 252,
    "totalTokenCount": 262
  },
  "modelVersion": "gemini-1.5-flash"
}
```

**Key Fields to Note:**

1. **`totalTokenCount`**: The total number of tokens in the request and response.
2. **`text`**: The content generated by the model.

### VuGen Script for Non-Streaming

Let’s now explore how to send a request and capture metrics using a VuGen script. To evaluate Gemini’s Non-Streaming mode, use the `web_custom_request` function from the `WEB - HTTP/HTML protocol` to perform a POST request. Parse the `totalTokenCount` from the response and calculate the model’s rate by dividing the token count by the response time. Report it via `lr_user_data_point`, which displays them in the Controller and Analysis graphs for analysis.

**Script:**

```c
float calcSpeed(float token_count, float duration) {
  float duration_seconds = duration / 1000;  // Convert ms to seconds
  return token_count / duration_seconds;    // Calculate tokens per second
}

Action() {
  char* s_model_response_time = NULL;
  char* s_token_count = NULL;
  int model_response_time = 0;
  int token_count = 0;
  float model_rate = 0.0;

  // Register to capture the "totalTokenCount" from the response
  web_reg_save_param_json(
    "ParamName=totalTokenCount",
    "QueryString=$..totalTokenCount",
    "NotFound=warning",
    "SelectAll=Yes",
    SEARCH_FILTERS,
    "Scope=BODY",
    LAST);

  // Add necessary headers
  web_add_header("Content-Type", "application/json");

  // Perform the POST request
  web_custom_request("gemini request",
    "URL=https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=YOUR_API_KEY",
    "Method=POST",
    "Resource=0",
    "Body={ \"contents\":[ { \"parts\":[{\"text\": \"Write a story of about 200 words\"}]} ] }",
    "ResponseTime=geminiResponseTime",
    LAST);

  lr_log_message("Gemini response time: %s ms", lr_eval_string("{geminiResponseTime}"));

  // Calculate rate
  s_model_response_time = lr_eval_string("{geminiResponseTime}");
  s_token_count = lr_eval_string("{totalTokenCount_1}");
  model_response_time = atoi(s_model_response_time);
  token_count = atoi(s_token_count);
  model_rate = calcSpeed(token_count, model_response_time);

  lr_user_data_point("Gemini Rate (tokens/s)", model_rate);

  return 0;
}
```

### Replay Script and Analyze Results

Define a scenario in **Controller** to simulate the desired load and execute the test. Use **Analysis** to visualize metrics such as response time and token processing rate under different load conditions.

**Running Load Test in Controller:**
<div align=center>
  <img src="/assets/gemini/non-streaming-controller.png" alt="image" width="600" height="auto">
</div>

**Analyzing Results in Analysis:**
<div align=center>
  <img src="/assets/gemini/non-streaming-analysis.png" alt="image" width="600" height="auto">
</div>

---

## Streaming Mode

### API Example

**Request:**

```http
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse&key=YOUR_API_KEY
Content-Type: application/json

{
  "contents": [{
    "parts": [{"text": "Write a story of about 200 words"}]
  }]
}
```

**Response:**

```json
data: {"candidates": [{"content": {"parts": [{"text": "The"}],"role": "model"}}],"usageMetadata": {"promptTokenCount": 10,"totalTokenCount": 10},"modelVersion": "gemini-1.5-flash"}

data: {"candidates": [{"content": {"parts": [{"text": " old lighthouse keeper, Silas, squinted at the churning grey sea.  Fifty"}],"role": "model"}}],"usageMetadata": {"promptTokenCount": 10,"totalTokenCount": 10},"modelVersion": "gemini-1.5-flash"}

...
```

**Key Fields to Note:**

1. **`text`**: The content returned incrementally by the model.
2. **`totalTokenCount`**: The total token count, including prompt and generated content.

### VuGen Script for Streaming

To perform a load test in Streaming mode, register an asynchronous request using `web_reg_async_attributes` and process the streaming response in the callback functions.

**Register an Asynchronous Request:**

```c
Action()
{
  web_reg_async_attributes("ID=Push_0",
  "URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse&key=YOUR_API_KEY",
  "Pattern=Push",
  "RequestCB=Push_0_RequestCB",
  "ResponseBodyBufferCB=Push_0_ResponseBodyBufferCB",
  "ResponseCB=Push_0_ResponseCB",
  LAST);
  
  web_add_header("Content-Type","application/json");
  
  web_custom_request("gemini request",
         "URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse&key=YOUR_API_KEY",
         "Method=POST",
         "Resource=0",
         "Body={ \"contents\":[ { \"parts\":[{\"text\": \"Write a story of about 200 words\"}]} ] }",
         "ResponseTime=geminiResponseTime",
         LAST);

  web_sync("ParamCreated=ready","RetryIntervalMs=3000","RetryTimeoutMs=120000",LAST);

  web_stop_async("ID=Push_0", LAST);

  lr_free_parameter("ready");

  return 0;
}
```

### Callback Functions

**Processing Metrics in Callbacks:**

```c
int Push_0_RequestCB()
{
  return WEB_ASYNC_CB_RC_OK;
}

int accumulatedTime = 0;
int responseIndex = 0;

int count_words(const char *str) {
    int count = 0;
    int in_word = 0;

    // Iterate through the string
    while (*str) {
        if (isspace((unsigned char)*str)) {
            // If the current character is a space, we are not in a word
            in_word = 0;
        } else if (!in_word) {
            // If the current character is not a space and we were not in a word, it's a new word
            in_word = 1;
            count++;
        }
        str++;
    }

    return count;
}

int count_gemini_response_words(const char *str) {
    const char *prefix;
    const char *suffix;
    size_t prefix_len;
    size_t str_len;
    const char *start;
    const char *end;
    size_t new_len;
    char *trimmed;
    int res = 0;

    prefix = "data: ";
    suffix = "\r\n\r\n";
    prefix_len = strlen(prefix);
    str_len = strlen(str);

    // Start after "data:" if it exists
    if (strncmp(str, prefix, prefix_len) == 0) {
        start = str + prefix_len;
    } else {
        start = str;
    }

    // End before "\r\n\r\n" if it exists
    end = str + str_len;
    if (str_len >= 4 && strcmp(end - 4, suffix) == 0) {
        end -= 4;
    }

    // Calculate the new length and allocate memory
    new_len = end - start;
    trimmed = (char*)malloc(new_len + 1);
    if (!trimmed) {
        return NULL;
    }

    // Copy the trimmed content
    strncpy(trimmed, start, new_len);
    trimmed[new_len] = '\0';
    
    if (strlen(trimmed) <= 0)
      return 0;
    
  lr_save_string(trimmed, "JSON_Input_Param");
  
  res = lr_eval_json("Buffer={JSON_Input_Param}",
               "JsonObject=s_json_obj", LAST);
  if (res == -1)
    return 0;
  
  lr_json_get_values("JsonObject=s_json_obj",
                "ValueParam=s_new_storename",
                "QueryString=$..text",
                "NotFound=Continue",
                 LAST);

  return count_words(lr_eval_string("{s_new_storename}"));
}

int Push_0_ResponseBodyBufferCB(
  const char *aLastBufferStr,
  int aLastBufferLen,
  const char *aAccumulatedStr,
  int aHttpStatusCode) {

  int latency = 0;
  int word_count = 0;

  if (aLastBufferLen <= 0)
    return WEB_ASYNC_CB_RC_OK;

  latency = calculate_latency();  // Calculate latency between chunks
  word_count = count_gemini_response_words(aLastBufferStr);

  if (responseIndex == 0) {
    lr_user_data_point("First Word Latency (ms)", latency);
  } else {
    lr_user_data_point("Per Word Latency (ms)", latency / word_count);
  }

  responseIndex++;
  return WEB_ASYNC_CB_RC_OK;
}

int Push_0_ResponseCB(...) {
  // Reset state at the end of the conversation
  lr_save_string("OK","ready");
  accumulatedTime = 0;
  responseIndex = 0;
  return WEB_ASYNC_CB_RC_OK;
}
```

### Key Metrics in Streaming Mode

Gemini’s streaming responses do not include token count information. Instead, we calculate word-level latency by parsing the incremental responses.

1. **First Word Latency (ms):** This metric measures the time taken from sending the request to receiving the first word of the model's response. It helps evaluate the initial responsiveness of the model.
2. **Per Word Latency (ms):** This metric calculates the average time it takes for the model to generate each subsequent word. It’s derived by dividing the total latency between responses by the number of words in the response.

### Replay Script and Analyze Results

Define a scenario in **Controller** and execute the streaming load test. Use **Analysis** to evaluate metrics such as word latency and overall performance trends under different loads.

**Running Load Test in Controller:**
<div align=center>
  <img src="/assets/gemini/streaming-controller.png" alt="image" width="600" height="auto">
</div>

**Analyzing Results in Analysis:**
<div align=center>
  <img src="/assets/gemini/streaming-analysis.png" alt="image" width="600" height="auto">
</div>

---

