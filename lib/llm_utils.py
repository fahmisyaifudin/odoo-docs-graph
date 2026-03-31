"""LLM utility functions for reasoning and response generation."""

import json
from typing import List, Dict, Any
from openai import OpenAI


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

## Top Similar Nodes (Initial Search Results):
"""
    
    # Add seed results summary
    for i, result in enumerate(seed_results[:5], 1):
        props = result.get("properties", {})
        score = result.get("score", 0)
        user_prompt += f"\n{i}. {props.get('name', 'N/A')} (Score: {score:.4f})"
        user_prompt += f"\n   Type: {props.get('type', 'N/A')}"
        user_prompt += f"\n   Section: {props.get('heading', 'N/A')}"
    
    user_prompt += """

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
            max_tokens=4000
        )
        
        # Parse the response
        llm_response = response.choices[0].message.content
        
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
                "raw_response": llm_response
            }
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Could not parse LLM response as JSON: {e}")
            return {
                "success": False,
                "error": "JSON parse error",
                "raw_response": llm_response
            }
            
    except Exception as e:
        print(f"✗ LLM API error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
