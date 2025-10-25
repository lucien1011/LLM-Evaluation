"""
Utility functions for interacting with Ollama API
These functions can be reused across different benchmarking tasks
"""

import requests
from typing import Dict, Optional


def check_ollama_connection(ollama_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama API is accessible

    Args:
        ollama_url: URL of the Ollama API

    Returns:
        True if connection is successful, False otherwise
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def get_available_models(ollama_url: str = "http://localhost:11434") -> list:
    """
    Get list of available models from Ollama

    Args:
        ollama_url: URL of the Ollama API

    Returns:
        List of model names
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []


def query_ollama(
    prompt: str,
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: int = 60
) -> str:
    """
    Query the Ollama model with a prompt

    Args:
        prompt: The prompt to send to the model
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum number of tokens to generate (None for default)
        timeout: Request timeout in seconds

    Returns:
        Model's response text, or empty string on error
    """
    api_endpoint = f"{ollama_url}/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["options"] = {"num_predict": max_tokens}

    try:
        response = requests.post(api_endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return ""


def query_ollama_chat(
    messages: list,
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    timeout: int = 60
) -> str:
    """
    Query Ollama using chat endpoint (for conversational models)

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API
        temperature: Sampling temperature
        timeout: Request timeout in seconds

    Returns:
        Model's response text, or empty string on error
    """
    api_endpoint = f"{ollama_url}/api/chat"

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }

    try:
        response = requests.post(api_endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama chat: {e}")
        return ""
