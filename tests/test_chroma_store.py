import os
import sys
import uuid
import time

# Add the project root to python path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.chroma_store import chroma_store

def test_chromadb_integration():
    print("=== STARTING CHROMADB INTEGRATION TEST ===")
    client = chroma_store

    # Create a unique tag for this run to avoid cross-pollution
    run_tag = f"test_run_{uuid.uuid4().hex[:6]}"
    print(f"Using container tag: {run_tag}")

    # 1. Add raw text document (Simulating a college OS lecture notes file)
    test_filepath = "/home/kshayik/College/Notes/OperatingSystems.md"
    test_content = (
        "Operating Systems Lecture 1: Processes and Threads.\n"
        "A process is an executing instance of a computer program. "
        "A thread is the smallest sequence of programmed instructions that can be managed independently.\n"
        "Memory limit context: local buffers must stay below 1.2 GB RAM to prevent kernel swapping."
    )
    
    print("\n[Test 1] Ingesting college note file...")
    try:
        res = client.add_document(
            content=test_content,
            container_tag=run_tag,
            filepath=test_filepath,
            metadata={"semester": "spring_2026", "subject": "CS-301"}
        )
        print(f"Success! Document ID: {res.get('id')} - Status: {res.get('status')}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    # 2. Query the document (RAG search)
    print("\n[Test 2] Querying notes for thread definitions...")
    try:
        results = client.search_documents(
            query="What is a thread in operating systems?",
            container_tag=run_tag,
            limit=1,
            chunk_threshold=0.0
        )
        print(f"Raw search results: {results}")
        if results:
            match = results[0]
            print(f"Found match (Score: {match.get('similarity', 0):.4f}):")
            print(f"  - Content: {match.get('content')}")
            print(f"  - File path: {match.get('filepath')}")
        else:
            print("No matches found.")
    except Exception as e:
        print(f"Search failed: {e}")

    # 3. Ingest Conversation (Conversational context)
    print("\n[Test 3] Ingesting chat conversation...")
    conv_id = f"session_{uuid.uuid4().hex[:6]}"
    messages = [
        {"role": "user", "content": "I am studying CS-301 Operating Systems today. Remember this."},
        {"role": "assistant", "content": "Got it! I will remember you are studying CS-301 Operating Systems."}
    ]
    try:
        res = client.ingest_conversation(
            conversation_id=conv_id,
            messages=messages,
            container_tags=[run_tag]
        )
        print(f"Success! Ingested conversation. Response: {res}")
    except Exception as e:
        print(f"Conversation ingestion failed: {e}")

    # 4. Add Memories Directly (Bypassing workflow/memory agent)
    print("\n[Test 4] Ingesting memories directly...")
    try:
        res = client.add_memories(
            memories=[
                "User's name is kshayik.",
                "User is building Mr. Meeseeks OS companion locally on Ubuntu."
            ],
            container_tag=run_tag
        )
        print(f"Success! Created memories: {res}")
    except Exception as e:
        print(f"Direct memories creation failed: {e}")

    # 5. Fetch Profile (Memory State)
    print("\n[Test 5] Fetching user profile...")
    try:
        profile = client.get_profile(container_tag=run_tag)
        print(f"Success! Profile data: {profile}")
    except Exception as e:
        print(f"Profile fetch failed: {e}")

if __name__ == "__main__":
    test_chromadb_integration()
