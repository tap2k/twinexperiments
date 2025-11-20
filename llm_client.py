import os
import json
import openai
import anthropic
from anthropic import Anthropic
import google.generativeai as genai
from huggingface_hub import InferenceClient

MODEL_MAP = [
    # OpenAI models (costs per 1M tokens)
    {'shortName': 'o3', 'provider': 'openai', 'modelName': 'o3', 'inputCost': 2.0, 'outputCost': 8.0, 'maxChars': 400000},
    {'shortName': 'o4-mini', 'provider': 'openai', 'modelName': 'o4-mini', 'inputCost': 4.0, 'outputCost': 16.0, 'maxChars': 400000},
    {'shortName': 'gpt-5', 'provider': 'openai', 'modelName': 'gpt-5', 'inputCost': 1.25, 'outputCost': 10.0, 'maxChars': 400000},
    {'shortName': 'gpt-5-mini', 'provider': 'openai', 'modelName': 'gpt-5-mini', 'inputCost': 0.25, 'outputCost': 2.0, 'maxChars': 400000},
    {'shortName': 'gpt-5-nano', 'provider': 'openai', 'modelName': 'gpt-5-nano', 'inputCost': 0.05, 'outputCost': 0.40, 'maxChars': 400000},
    {'shortName': 'gpt-4o', 'provider': 'openai', 'modelName': 'gpt-4o', 'inputCost': 2.5, 'outputCost': 10.0, 'maxChars': 512000},
    {'shortName': 'gpt-4o-mini', 'provider': 'openai', 'modelName': 'gpt-4o-mini', 'inputCost': 0.15, 'outputCost': 0.6, 'maxChars': 512000},
    {'shortName': 'gpt-oss-120b', 'provider': 'deepinfra', 'modelName': 'openai/gpt-oss-120b', 'inputCost': 0.05, 'outputCost': 0.24, 'maxChars': 512000},
    {'shortName': 'gpt-oss-20b', 'provider': 'deepinfra', 'modelName': 'openai/gpt-oss-20b', 'inputCost': 0.03, 'outputCost': 0.14, 'maxChars': 512000},

    # Anthropic models (costs per 1M tokens)
    {'shortName': 'sonnet-4.5', 'provider': 'anthropic', 'modelName': 'claude-sonnet-4-5', 'inputCost': 3.0, 'outputCost': 15.0, 'maxChars': 800000},
    {'shortName': 'haiku-4.5', 'provider': 'anthropic', 'modelName': 'claude-haiku-4-5', 'inputCost': 1.0, 'outputCost': 5.0, 'maxChars': 800000},
    {'shortName': 'opus-4', 'provider': 'anthropic', 'modelName': 'claude-opus-4', 'inputCost': 15.0, 'outputCost': 75.0, 'maxChars': 800000},
    {'shortName': 'opus-4.1', 'provider': 'anthropic', 'modelName': 'claude-opus-4-1', 'inputCost': 15.0, 'outputCost': 75.0, 'maxChars': 800000},

    # Google models (costs per 1M tokens)
    {'shortName': 'gemini-2.5-pro', 'provider': 'google', 'modelName': 'gemini-2.5-pro', 'inputCost': 1.25, 'outputCost': 10.0, 'maxChars': 1000000},
    {'shortName': 'gemini-2.5-flash', 'provider': 'google', 'modelName': 'gemini-2.5-flash', 'inputCost': 0.15, 'outputCost': 2.50, 'maxChars': 1000000},
    {'shortName': 'gemini-2.5-flash-lite', 'provider': 'google', 'modelName': 'gemini-2.5-flash-lite', 'inputCost': 0.1, 'outputCost': 0.4, 'maxChars': 1000000},
    {'shortName': 'gemini-2-flash', 'provider': 'google', 'modelName': 'gemini-2.0-flash', 'inputCost': 0.1, 'outputCost': 0.4, 'maxChars': 4000000},

    # xAI models (costs per 1M tokens)
    {'shortName': 'grok-4', 'provider': 'xai', 'modelName': 'grok-4', 'inputCost': 3.0, 'outputCost': 15.0, 'maxChars': 1024000},
    {'shortName': 'grok-4-fast-reasoning', 'provider': 'xai', 'modelName': 'grok-4-fast-reasoning', 'inputCost': 0.20, 'outputCost': 0.50, 'maxChars': 8000000},
    {'shortName': 'grok-4-fast-non-reasoning', 'provider': 'xai', 'modelName': 'grok-4-fast-non-reasoning', 'inputCost': 0.20, 'outputCost': 0.50, 'maxChars': 8000000},
    {'shortName': 'grok-3', 'provider': 'xai', 'modelName': 'grok-3', 'inputCost': 3.0, 'outputCost': 15.0, 'maxChars': 4000000},

    # Meta models via DeepInfra (costs per 1M tokens)
    {'shortName': 'llama-4-maverick', 'provider': 'deepinfra', 'modelName': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8', 'inputCost': 0.20, 'outputCost': 0.60, 'maxChars': 2048000},
    {'shortName': 'llama-4-scout', 'provider': 'deepinfra', 'modelName': 'meta-llama/Llama-4-Scout-17B-16E-Instruct', 'inputCost': 0.10, 'outputCost': 0.30, 'maxChars': 40000000},
    {'shortName': 'llama-3.3-70b', 'provider': 'deepinfra', 'modelName': 'meta-llama/Llama-3.3-70B-Instruct', 'inputCost': 0.23, 'outputCost': 0.40, 'maxChars': 524288},

    # DeepSeek models (costs per 1M tokens)
    {'shortName': 'deepseek-v3', 'provider': 'deepinfra', 'modelName': 'deepseek-ai/DeepSeek-V3', 'inputCost': 0.27, 'outputCost': 1.10, 'maxChars': 512000},
    {'shortName': 'deepseek-r1', 'provider': 'deepinfra', 'modelName': 'deepseek-ai/DeepSeek-R1-Turbo', 'inputCost': 0.55, 'outputCost': 2.19, 'maxChars': 512000},

    # Qwen models (costs per 1M tokens)
    {'shortName': 'qwen-3-235b-a22b-think', 'provider': 'deepinfra', 'modelName': 'Qwen/Qwen3-235B-A22B-Thinking-2507', 'inputCost': 0.30, 'outputCost': 2.90, 'maxChars': 1048576},
    {'shortName': 'qwen-3-235b-a22b', 'provider': 'deepinfra', 'modelName': 'Qwen/Qwen3-235B-A22B-Instruct-2507', 'inputCost': 0.13, 'outputCost': 0.60, 'maxChars': 1048576},
    {'shortName': 'qwen-3-next-instruct', 'provider': 'deepinfra', 'modelName': 'Qwen/Qwen3-Next-80B-A3B-Instruct', 'inputCost': 0.14, 'outputCost': 1.10, 'maxChars': 1048576},
    {'shortName': 'qwen-3-32b', 'provider': 'deepinfra', 'modelName': 'Qwen/Qwen3-32B', 'inputCost': 0.10, 'outputCost': 0.30, 'maxChars': 524288},

    # Moonshot models (costs per 1M tokens)
    {'shortName': 'kimi-k2', 'provider': 'deepinfra', 'modelName': 'moonshotai/Kimi-K2-Instruct', 'inputCost': 0.55, 'outputCost': 2.10, 'maxChars': 1024000},
]

# Defaults
DEFAULT_MODEL = 'gemini-2.5-flash-lite'
DEFAULT_RESPONSE = ''

def get_model_info(model):
    """Get model information including costs from MODEL_MAP."""
    return next((m for m in MODEL_MAP if m['shortName'] == model), None)

def calculate_cost(input_tokens, output_tokens, model):
    """Calculate cost for a model call based on token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model short name

    Returns:
        dict with input_cost, output_cost, and total_cost in dollars
    """
    model_info = get_model_info(model)
    if not model_info or 'inputCost' not in model_info:
        return {'input_cost': 0, 'output_cost': 0, 'total_cost': 0, 'note': 'Cost unavailable'}

    # Costs in MODEL_MAP are per 1M tokens
    input_cost = (input_tokens / 1_000_000) * model_info['inputCost']
    output_cost = (output_tokens / 1_000_000) * model_info['outputCost']

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': input_cost + output_cost
    }

def clean_response(response):
    """Clean and parse JSON response."""
    if not response:
        return {}
    try:
        cleaned = response.replace('```json\n', '').replace('\n```', '').strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return {}

def trim_messages(messages, model_type=None, model=DEFAULT_MODEL):
    """Prepare and trim message history based on model's max character limit."""
    if not messages:
        raise ValueError("messages not provided")
        
    # Get model's max chars from MODEL_MAP
    # For custom models not in MODEL_MAP, use default value of 400K
    model_info = next((m for m in MODEL_MAP if m['shortName'] == model), None)
    max_chars = model_info['maxChars'] if model_info else 400000  # Default to 400K for custom models

    prepared_messages = []
    for msg in messages:
        if msg['role'] == 'researcher':
            msg['role'] = 'assistant'
        prepared_messages.append({'role': msg['role'], 'content': msg['content']})

    # Calculate total character count
    total_chars = sum(len(msg['content']) for msg in prepared_messages)

    # Find system message index
    system_idx = next((i for i, msg in enumerate(prepared_messages) 
                      if msg['role'] == 'system'), -1)

    # Trim messages while preserving system message
    while total_chars > max_chars and len(prepared_messages) > 1:
        if system_idx == 0:
            del prepared_messages[1]
        else:
            del prepared_messages[0]
            if system_idx > 0:
                system_idx -= 1
        total_chars = sum(len(msg['content']) for msg in prepared_messages)

    # Move system message to start if it exists and isn't first
    if system_idx > 0:
        system_msg = prepared_messages.pop(system_idx)
        prepared_messages.insert(0, system_msg)

    # Ensure at least one user message
    if not any(msg['role'] == 'user' for msg in prepared_messages):
        prepared_messages.append({'role': 'user', 'content': 'Hello'})

    return prepared_messages

def openai_compatible_chat(messages, model="meta-llama/Llama-3.3-70B-Instruct",
                               json_mode=False, temperature=None,
                               frequency_penalty=0, base_url=None, api_key=None, return_usage=False):
    """Generic chat completion for OpenAI-compatible APIs."""
    try:
        processed_messages = trim_messages(messages, model=model)

        # Handle o1 model specific adjustments
        if model.startswith('o'):
            system_msg = next((msg['content'] for msg in processed_messages
                             if msg['role'] == 'system'), None)
            messages_without_system = [msg for msg in processed_messages
                                     if msg['role'] != 'system']

            if system_msg:
                # Find last user message
                for i in range(len(messages_without_system) - 1, -1, -1):
                    if messages_without_system[i]['role'] == 'user':
                        messages_without_system[i]['content'] = f"{system_msg}\n\n{messages_without_system[i]['content']}"
                        break

            processed_messages = messages_without_system

        client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            base_url=base_url
        )

        params = {
            'messages': processed_messages,
            'model': model,
        }

        if not model.startswith('o'):
            if temperature is not None:
                params['temperature'] = temperature
            if frequency_penalty:
                params['frequency_penalty'] = frequency_penalty
            params['response_format'] = {'type': 'json_object' if json_mode else 'text'}
        else:
            params['reasoning_effort'] = 'medium'

        completion = client.chat.completions.create(**params)
        response = completion.choices[0].message.content

        if return_usage:
            usage = {
                'input_tokens': completion.usage.prompt_tokens if completion.usage else 0,
                'output_tokens': completion.usage.completion_tokens if completion.usage else 0,
            }
            return (json.loads(response) if json_mode else response), usage

        return json.loads(response) if json_mode else response
    except Exception as e:
        print(f"Error in openai_compatible_chat function: {e}")
        if return_usage:
            return DEFAULT_RESPONSE, {'input_tokens': 0, 'output_tokens': 0}
        return DEFAULT_RESPONSE

def claude(messages, model='claude-3-5-haiku-latest',
                json_mode=False, temperature=None,
                frequency_penalty=0):
    """Anthropic Claude chat completion."""
    try:
        # Extract and remove system message
        system_msg = next((msg['content'] for msg in messages
                          if msg['role'] == 'system'), '')
        messages_without_system = [msg for msg in messages
                                 if msg['role'] != 'system']

        processed_messages = trim_messages(messages_without_system,
                                            'claude', model)

        client = Anthropic()
        params = {
            'max_tokens': 64000,  # All Claude 4.x models support 64K output
            'system': system_msg,
            'messages': processed_messages,
            'model': model,
        }
        if temperature is not None:
            params['temperature'] = temperature

        response = client.messages.create(**params)

        text_response = response.content[0].text
        return clean_response(text_response) if json_mode else text_response
    except Exception as e:
        print(f"Error in claude function: {e}")
        return DEFAULT_RESPONSE

def gemini(messages, model='gemini-2.0-flash-001',
                json_mode=False, temperature=None,
                frequency_penalty=0):
    """Google Gemini chat completion."""
    try:
        processed_messages = trim_messages(messages, model=model)

        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        generation_config = {
            'max_output_tokens': 65536  # All Gemini 2.5 models support 64K output
        }
        if temperature is not None:
            generation_config['temperature'] = temperature

        model_client = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        system_msg = next((msg['content'] for msg in processed_messages 
                          if msg['role'] == 'system'), '')

        chat_messages = [
            {'role': 'model' if msg['role'] == 'assistant' else 'user',
             'parts': [{'text': msg['content']}]}
            for msg in processed_messages if msg['role'] != 'system'
        ]

        # Find first user message
        first_user_idx = next((i for i, msg in enumerate(chat_messages) 
                             if msg['role'] == 'user'), -1)
        if first_user_idx == -1:
            return DEFAULT_RESPONSE

        chat_history = chat_messages[first_user_idx:]

        if len(chat_history) == 1:
            prompt = f"{system_msg}{chat_history[0]['parts'][0]['text']}"
            response = model_client.generate_content(prompt)
            text_response = response.text
        else:
            chat = model_client.start_chat(
                history=chat_history[:-1]
            )
            prompt = f"{system_msg}{chat_history[-1]['parts'][0]['text']}"
            response = chat.send_message(prompt)
            text_response = response.text

        return clean_response(text_response) if json_mode else text_response
    except Exception as e:
        print(f"Error in gemini function: {e}")
        return DEFAULT_RESPONSE

def pplx(messages, model='llama-3.1-sonar-small-128k-online',
         json_mode=False, temperature=None,
         frequency_penalty=0.01):
    """Perplexity chat completion."""
    try:
        import requests

        prepared_messages = []
        last_role = None

        for msg in messages:
            if last_role == 'system':
                prepared_messages.append({
                    'role': 'user',
                    'content': 'Welcome!'
                })
                last_role = 'user'

            if msg['role'] != last_role:
                last_role = msg['role']
                prepared_messages.append(msg)

        processed_messages = trim_messages(prepared_messages, model=model)

        payload = {
            'model': model,
            'messages': processed_messages,
            'frequency_penalty': frequency_penalty,
        }
        if temperature is not None:
            payload['temperature'] = temperature

        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            json=payload,
            headers={
                'Authorization': f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                'Content-Type': 'application/json',
            }
        )
        response.raise_for_status()
        data = response.json()
        message = data['choices'][0]['message']['content']

        return clean_response(message) if json_mode else message
    except Exception as e:
        print(f"Error in pplx function: {e}")
        return DEFAULT_RESPONSE

def format_messages_for_llama(messages):
    """Format chat messages for Llama models."""
    formatted = ""
    system_content = ""
    
    # Extract system message if present
    system_msg = next((msg for msg in messages if msg['role'] == 'system'), None)
    if system_msg:
        system_content = system_msg['content']
    
    # Format conversation
    for i, msg in enumerate(messages):
        if msg['role'] == 'system':
            continue  # Skip system message as we handle it separately
        
        if msg['role'] == 'user':
            # For the first user message, include system prompt if available
            if i == 0 or (i == 1 and messages[0]['role'] == 'system'):
                if system_content:
                    formatted += f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{msg['content']} [/INST]"
                else:
                    formatted += f"<s>[INST] {msg['content']} [/INST]"
            else:
                formatted += f"<s>[INST] {msg['content']} [/INST]"
        elif msg['role'] == 'assistant':
            formatted += f" {msg['content']}</s>"
    
    return formatted

def huggingface(messages, model, json_mode=False, temperature=None, frequency_penalty=0):
    """Hugging Face Inference API chat completion for Llama models."""
    try:
        # Extract model name from the "hf:" prefix
        if model.startswith('hf:'):
            model = model[3:]
            
        # Format messages for Llama models
        formatted_prompt = format_messages_for_llama(messages)
        
        # Create client with your token
        client = InferenceClient(token=os.getenv('HF_TOKEN'))

        # Call the model
        params = {
            'model': model,
            'max_new_tokens': 4096,
            'repetition_penalty': 1.0 + frequency_penalty
        }
        if temperature is not None:
            params['temperature'] = temperature

        response = client.text_generation(formatted_prompt, **params)
        
        return clean_response(response) if json_mode else response
    except Exception as e:
        print(f"Error in huggingface function: {e}")
        return DEFAULT_RESPONSE

def call_llm(messages, model='gpt-4o-mini', json_mode=False,
                  temperature=None, frequency_penalty=0):
    """Main entry point for LLM calls."""
    try:
        model = model.strip() if model else DEFAULT_MODEL
        print(f"USING LLM MODEL: {model}")
        
        # Handle Hugging Face models
        if model.startswith('hf:'):
            return huggingface(messages, model, json_mode, temperature, frequency_penalty)

        # Find model info in map
        model_info = next((m for m in MODEL_MAP if m['shortName'] == model), None)
        
        # If model is not in MODEL_MAP, assume it's a custom OpenAI model
        if not model_info:
            return openai_compatible_chat(
                messages, model, json_mode, temperature, frequency_penalty
            )
            
        provider = model_info['provider']
        model_name = model_info['modelName']

        if provider == 'openai':
            return openai_compatible_chat(messages, model_name, json_mode, temperature,
                                             frequency_penalty)
        elif provider == 'anthropic':
            return claude(messages, model_name, json_mode, temperature,
                              frequency_penalty)
        elif provider == 'google':
            return gemini(messages, model_name, json_mode, temperature,
                              frequency_penalty)
        elif provider == 'deepinfra':
            return openai_compatible_chat(
                messages, model_name, json_mode, temperature, frequency_penalty,
                base_url='https://api.deepinfra.com/v1/openai',
                api_key=os.getenv('DEEP_INFRA_API_TOKEN')
            )
        elif provider == 'openrouter':
            return openai_compatible_chat(
                messages, model_name, json_mode, temperature, frequency_penalty,
                base_url='https://openrouter.ai/api/v1',
                api_key=os.getenv('OPENROUTER_API_KEY')
            )
        elif provider == 'xai':
            return openai_compatible_chat(
                messages, model_name, json_mode, temperature, frequency_penalty,
                base_url='https://api.x.ai/v1',
                api_key=os.getenv('XAI_API_KEY')
            )
        else:
            print(f"Unknown provider: {provider}, falling back to {DEFAULT_MODEL}")
            return openai_compatible_chat(messages, DEFAULT_MODEL, json_mode,
                                             temperature, frequency_penalty)
    except Exception as e:
        print(f"Error in call_llm function: {e}")
        return DEFAULT_RESPONSE
