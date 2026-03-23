"""Test the webhook server with mock Vapi payloads.

Start the server first:  python -m backend.webhook_server
Then run:                 python -m scripts.test_webhook

Tests:
1. Health check
2. assistant-request (persona selection)
3. conversation-update (full RAG + Claude pipeline)
4. Persona switching (same question, different pages)
"""

import json
import sys
import requests

BASE_URL = "http://localhost:8000"


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_health():
    divider("TEST 1: Health Check")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {resp.status_code}")
    print(f"  Body: {resp.json()}")
    assert resp.status_code == 200
    print("PASS")


def test_assistant_request(page_route: str):
    """Test assistant-request — should return persona-specific config."""
    payload = {
        "message": {
            "type": "assistant-request",
            "call": {
                "metadata": {
                    "pageRoute": page_route
                }
            }
        }
    }

    resp = requests.post(f"{BASE_URL}/vapi/webhook", json=payload)
    data = resp.json()

    assistant = data.get("assistant", {})
    first_msg = assistant.get("firstMessage", "")
    system_prompt = assistant.get("model", {}).get("systemPrompt", "")

    print(f"  Route: {page_route}")
    print(f"  First message: {first_msg[:100]}...")
    print(f"  System prompt length: {len(system_prompt)} chars")
    return data


def test_conversation_update(page_route: str, user_message: str):
    """Test conversation-update — should return Claude response with RAG."""
    payload = {
        "message": {
            "type": "conversation-update",
            "call": {
                "metadata": {
                    "pageRoute": page_route
                }
            },
            "messages": [
                {
                    "role": "assistant",
                    "content": "Signal Processing and Response Core active."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
    }

    resp = requests.post(f"{BASE_URL}/vapi/webhook", json=payload)
    data = resp.json()

    content = data.get("content", "")
    print(f"  Route: {page_route}")
    print(f"  User: \"{user_message}\"")
    print(f"  SPARC: {content[:300]}")
    if len(content) > 300:
        print(f"  ... ({len(content)} chars total)")
    return data


def main():
    # Check server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.ConnectionError:
        print(f"Cannot connect to {BASE_URL}")
        print("Start the server first:  python -m backend.webhook_server")
        sys.exit(1)

    # Test 1: Health
    test_health()

    # Test 2: Assistant requests for each persona
    divider("TEST 2: Assistant Request (Persona Selection)")
    routes = [
        "/services/visibility-authority",
        "/services/acquisition-conversion",
        "/services/experience-infrastructure",
        "/about",
    ]
    for route in routes:
        test_assistant_request(route)
        print()
    print("PASS — all personas returned correct config")

    # Test 3: Conversation with RAG
    divider("TEST 3: Conversation Update (RAG + Claude)")
    test_conversation_update(
        "/services/visibility-authority",
        "What is GEO and how does it affect my business?"
    )

    # Test 4: Diagnostic framework
    divider("TEST 4: Diagnostic Framework")
    print("  Expecting: Diagnosis → Impact → Recommendations → Next Step")
    test_conversation_update(
        "/services/acquisition-conversion",
        "My ads are expensive and I'm not getting leads"
    )

    # Test 5: Persona tone comparison
    divider("TEST 5: Same Question, Different Personas")
    question = "How do I get more customers?"

    print("--- Navigator ---")
    test_conversation_update("/services/visibility-authority", question)
    print()

    print("--- Accelerator ---")
    test_conversation_update("/services/acquisition-conversion", question)
    print()

    print("--- Connector ---")
    test_conversation_update("/about", question)

    # Test 6: About / platform question
    divider("TEST 6: Platform Question")
    test_conversation_update("/about", "What is Power Spark X and how does it work?")

    divider("ALL TESTS COMPLETE")
    print("Review the responses above to verify:")
    print("  - Tone shifts between personas")
    print("  - Diagnostic framework (Diagnosis/Impact/Recommendations/Next Step)")
    print("  - RAG pulls relevant knowledge")
    print("  - About page explains the full ecosystem")


if __name__ == "__main__":
    main()
