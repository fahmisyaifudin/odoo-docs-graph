from lib.pgvector_utils import get_pg_connection
from ask_question import ask_question
import json

def save_answer_result(
    conn,
    question_id: str,
    llm_reasoning_model: str,
    method: str,
    embedding_model: str,
    result: dict
):
    """Save the answer result to the answer_result table."""
    with conn.cursor() as cur:
        query = """
        INSERT INTO answer_result (
            question_id,
            llm_reasoning_model,
            method,
            embedding_model,
            is_correct,
            total_token,
            result
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        # Extract is_correct from llm_reasoning - check if it contains "yes" (case-insensitive)
        llm_reasoning = result.get("llm_reasoning", {})
        if isinstance(llm_reasoning, dict):
            answer_text = llm_reasoning.get("answer", "")
        else:
            answer_text = str(llm_reasoning)

        # Extract total_token from usage
        usage = result.get("usage", {})
        total_token = usage.get("total_tokens", 0)
        
        # Check if answer contains "yes" (case-insensitive)
        is_correct = "yes" in answer_text.lower() if answer_text else None
        
        # Convert result dict to JSON string
        result_json = json.dumps(result, ensure_ascii=False)
        
        cur.execute(query, (
            question_id,
            llm_reasoning_model,
            method,
            embedding_model,
            is_correct,
            total_token,
            result_json
        ))
        
        result_id = cur.fetchone()[0]
        conn.commit()
        
        return result_id


if __name__ == '__main__':
    conn = get_pg_connection()
    
    try:
        with conn.cursor() as cur:
            module = "Point of Sales"
            
            # Fetch questions with their IDs
            query = """
            SELECT id, question, answer 
            FROM question
            WHERE module = %s and deleted_at is null and answer = 1
            """
            cur.execute(query, (module,))
            results = cur.fetchall()
            
            print(f"Found {len(results)} questions for module: {module}\n")
            
            # Process each question
            for row in results:
                question_id, question, expected_answer = row
                
                print(f"Processing question ID: {question_id}")
                print(f"Question: {question}")
                print(f"Expected Answer: {expected_answer}")
                print("-" * 70)
                
                # Call ask_question with different methods
                for method in ["neo4j"]:
                    print(f"\n--- Using method: {method} ---")
                    
                    try:
                        result = ask_question(
                            question=question,
                            top_k=5,
                            method=method
                        )
                        
                        # Add metadata to result
                        result["question"] = question
                        result["expected_answer"] = expected_answer
                        result["method"] = method
                        
                        # Save to database
                        result_id = save_answer_result(
                            conn=conn,
                            question_id=question_id,
                            llm_reasoning_model="google/gemma-3-27b-it",
                            method=method,
                            embedding_model="qwen/qwen3-embedding-8b",
                            result=result
                        )
                        
                        print(f"✓ Result saved with ID: {result_id}")
                        print(f"  Success: {result.get('success', False)}")
                        
                        if result.get('llm_reasoning'):
                            answer = result['llm_reasoning'].get('answer', 'N/A')
                            print(f"  LLM Answer: {answer[:100]}...")
                        
                    except Exception as e:
                        print(f"✗ Error with method {method}: {str(e)}")
                        
                        # Save error result
                        error_result = {
                            "success": False,
                            "error": str(e),
                            "question": question,
                            "method": method
                        }
                        
                        try:
                            save_answer_result(
                                conn=conn,
                                question_id=question_id,
                                llm_reasoning_model="google/gemma-3-27b-it",
                                method=method,
                                embedding_model="qwen/qwen3-embedding-8b",
                                result=error_result
                            )
                        except:
                            pass
                
                print("\n" + "=" * 70 + "\n")
                
    finally:
        conn.close()
        print("\n✓ Testing completed and connection closed.")
