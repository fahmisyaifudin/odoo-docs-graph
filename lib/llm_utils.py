"""LLM utility functions for reasoning and response generation."""

import json
from typing import List, Dict, Any
from openai import OpenAI

def generate_direct_llm(
    question: str,
    client: OpenAI,
    model: str = "anthropic/claude-3.5-sonnet"
) -> Dict[str, Any]:
    """
    Use LLM to generate direct answer to a question about Odoo 17.
    
    This function:
    1. Sends question to LLM with Odoo 17 context
    2. Returns the LLM response in JSON format
    """
    
    system_prompt = """You are an expert Odoo consultant specializing in Odoo version 17. Your task is to answer user questions about Odoo features based on your knowledge.

## Your Capabilities:
1. Answer questions about Odoo version 17 features
2. Provide accurate, specific answers based on your knowledge of Odoo version 17
3. Explain your reasoning clearly

## Response Format:
You must respond in this JSON structure:
{
    "answer": "Your direct answer to the question (yes/no)",
    "confidence": "high|medium|low",
}

## Important Rules:
1. Answer based on Odoo 17 (the latest stable version as of your knowledge)
2. If you're uncertain, indicate low confidence and explain why
3. Do not make up information - if you don't know, say so clearly
"""

    user_prompt = f"""## User Question:
{question}

## Task:
Based on your knowledge of Odoo 17, answer the user's question.
Provide your response in the required JSON format.
"""

    print("\n🤖 Sending question to LLM for direct answer...")
    
    try:
        # Call the LLM
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=10000,
        )
        
        # Parse the response
        llm_response = response.choices[0].message.content
        usage_obj = response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, 'prompt_tokens', 0),
            "completion_tokens": getattr(usage_obj, 'completion_tokens', 0),
            "total_tokens": getattr(usage_obj, 'total_tokens', 0)
        }
        
        # Try to parse as JSON
        try:
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                parsed_response = json.loads(json_str)
            else:
                parsed_response = json.loads(llm_response)
            
            print("✓ LLM reasoning complete")
            return {
                "success": True,
                "llm_response": parsed_response,
                "raw_response": llm_response,
                "usage": usage
            }
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Could not parse LLM response as JSON: {e}")
            return {
                "success": False,
                "error": "JSON parse error",
                "raw_response": llm_response,
                "usage": usage
            }
            
    except Exception as e:
        print(f"✗ LLM API error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_llm_reasoning(
    question: str,
    graph_context: str,
    seed_results: List[Dict[str, Any]],
    client: OpenAI,
    model: str = "anthropic/claude-3.5-sonnet"
) -> Dict[str, Any]:
    """
    Use LLM to generate reasoning based on graph traversal results.
    
    This function:
    1. Builds a comprehensive prompt with context
    2. Sends to LLM for reasoning
    3. Returns structured answer with explanation
    """
    
    # Build the system prompt
    system_prompt = """You are an expert Odoo consultant and knowledge graph analyst. Your task is to answer user questions about Odoo Point of Sales (POS) features based on the provided knowledge graph context.

## Your Capabilities:
1. Analyze the knowledge graph structure and relationships
2. Identify relevant features, modules, and configurations
3. Provide accurate, specific answers based on the graph data
4. Explain your reasoning clearly

## Response Format:
You must respond in this JSON structure:
{
    "answer": "Your direct answer to the question",
    "confidence": "high|medium|low",
    "reasoning": "Step-by-step explanation of how you arrived at the answer",
    "sources": [
        {
            "name": "Name of the node/feature",
            "type": "Feature|Module|etc",
            "relevance": "How this supports the answer"
        }
    ],
    "related_features": ["List of related features from the graph"],
    "limitations": "Any limitations or caveats to the answer, or 'None'"
}

## Important Rules:
1. ONLY use information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific about which nodes/features support your answer
4. Reference the knowledge graph structure in your reasoning
5. Do not make up information not present in the context
"""

    # Build the user prompt with context
    user_prompt = f"""## User Question:
{question}

## Knowledge Graph Context:
{graph_context}

## Task:
Based on the knowledge graph context provided above, answer the user's question.
Provide your response in the required JSON format.
"""

    print("\n🤖 Sending to LLM for reasoning...")
    
    try:
        # Call the LLM
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=6000
        )
        
        # Parse the response
        llm_response = response.choices[0].message.content
        
        # Convert usage object to dict
        usage_obj = response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, 'prompt_tokens', 0),
            "completion_tokens": getattr(usage_obj, 'completion_tokens', 0),
            "total_tokens": getattr(usage_obj, 'total_tokens', 0)
        }
        
        # Try to parse as JSON
        try:
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                parsed_response = json.loads(json_str)
            else:
                parsed_response = json.loads(llm_response)
            
            print("✓ LLM reasoning complete")
            return {
                "success": True,
                "llm_response": parsed_response,
                "raw_response": llm_response,
                "usage": usage
            }
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Could not parse LLM response as JSON: {e}")
            return {
                "success": False,
                "error": "JSON parse error",
                "raw_response": llm_response,
                "usage": usage
            }
            
    except Exception as e:
        print(f"✗ LLM API error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
