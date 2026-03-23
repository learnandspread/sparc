"""Configure SPARC™ Vapi Assistant with Custom Knowledge Base.

Flow:
1. Create a Custom Knowledge Base pointing to our /kb/search endpoint
2. Create/update the assistant with the KB attached via model.knowledgeBaseId

Vapi will automatically call our KB endpoint during conversations.

Usage:
    WEBHOOK_URL=https://xxx.ngrok.app/vapi/webhook python -m backend.scripts.setup_vapi_assistant
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

VAPI_BASE_URL = "https://api.vapi.ai"
VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
VAPI_VOICE_ID = os.getenv("VAPI_VOICE_ID", "")
VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID", "")
VAPI_KB_ID = os.getenv("VAPI_KB_ID", "")


def get_headers():
    return {
        "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
        "Content-Type": "application/json",
    }


# ─── Knowledge Base ─────────────────────────────────────────

def create_knowledge_base(server_url: str) -> dict:
    """Create a Custom Knowledge Base pointing to our search endpoint."""
    resp = requests.post(
        f"{VAPI_BASE_URL}/knowledge-base",
        headers=get_headers(),
        json={
            "provider": "custom-knowledge-base",
            "server": {
                "url": server_url,
            },
        },
    )
    if not resp.ok:
        print(f"\nKB Create Error {resp.status_code}:")
        print(json.dumps(resp.json(), indent=2))
        resp.raise_for_status()
    return resp.json()


def list_knowledge_bases() -> list:
    """List all knowledge bases."""
    resp = requests.get(
        f"{VAPI_BASE_URL}/knowledge-base",
        headers=get_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def delete_knowledge_base(kb_id: str) -> None:
    """Delete a knowledge base."""
    resp = requests.delete(
        f"{VAPI_BASE_URL}/knowledge-base/{kb_id}",
        headers=get_headers(),
    )
    resp.raise_for_status()


# ─── Assistant ───────────────────────────────────────────────

def build_assistant_config(persona_key: str = "connector", kb_id: str = "") -> dict:
    """Build the Vapi assistant configuration."""
    from backend.prompts.builder import build_system_prompt
    from backend.prompts.personas import get_persona

    persona = get_persona(persona_key)
    system_prompt = build_system_prompt(persona_key=persona_key)

    model_config = {
        "provider": "openai",
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            }
        ],
    }

    # Attach knowledge base if we have one
    if kb_id:
        model_config["knowledgeBaseId"] = kb_id

    config = {
        "name": "SPARC - Strategic Intelligence Layer",
        "model": model_config,
        "firstMessage": persona.opening_line,
        "serverUrl": WEBHOOK_URL,
        "serverMessages": [
            "end-of-call-report",
            "status-update",
        ],
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en",
        },
        "silenceTimeoutSeconds": 30,
        "responseDelaySeconds": 0.4,
        "maxDurationSeconds": 600,
        "endCallMessage": "Signal Processing and Response Core entering standby.",
    }

    if VAPI_VOICE_ID:
        config["voice"] = {
            "voiceId": VAPI_VOICE_ID,
            "provider": "11labs",
        }

    return config


def create_assistant(config: dict) -> dict:
    resp = requests.post(
        f"{VAPI_BASE_URL}/assistant",
        headers=get_headers(),
        json=config,
    )
    if not resp.ok:
        print(f"\nAPI Error {resp.status_code}:")
        print(json.dumps(resp.json(), indent=2))
        resp.raise_for_status()
    return resp.json()


def update_assistant(assistant_id: str, config: dict) -> dict:
    resp = requests.patch(
        f"{VAPI_BASE_URL}/assistant/{assistant_id}",
        headers=get_headers(),
        json=config,
    )
    if not resp.ok:
        print(f"\nAPI Error {resp.status_code}:")
        print(json.dumps(resp.json(), indent=2))
        resp.raise_for_status()
    return resp.json()


def list_assistants() -> list:
    resp = requests.get(
        f"{VAPI_BASE_URL}/assistant",
        headers=get_headers(),
    )
    resp.raise_for_status()
    return resp.json()


# ─── Main ────────────────────────────────────────────────────

def main():
    if not VAPI_PRIVATE_KEY:
        print("Error: VAPI_PRIVATE_KEY not set")
        sys.exit(1)

    if not WEBHOOK_URL:
        print("Error: WEBHOOK_URL not set")
        sys.exit(1)

    # KB search endpoint — separate from the webhook
    base_url = WEBHOOK_URL.rsplit("/vapi/webhook", 1)[0]
    kb_search_url = f"{base_url}/kb/search"

    print("=" * 60)
    print("SPARC Vapi Setup")
    print("=" * 60)
    print(f"\nWebhook URL: {WEBHOOK_URL}")
    print(f"KB Search URL: {kb_search_url}")
    print(f"Voice ID: {VAPI_VOICE_ID or '(not set)'}")

    # Step 1: Create or find Knowledge Base
    kb_id = VAPI_KB_ID
    if not kb_id:
        # Check if one exists
        kbs = list_knowledge_bases()
        existing_kb = [kb for kb in kbs if kb.get("name", "").startswith("SPARC")]
        if existing_kb:
            kb_id = existing_kb[0]["id"]
            print(f"\nFound existing KB: {kb_id}")
            # Delete and recreate to update URL
            print("Recreating KB with current URL...")
            delete_knowledge_base(kb_id)

        print("\nCreating Knowledge Base...")
        kb = create_knowledge_base(kb_search_url)
        kb_id = kb["id"]
        print(f"Created KB: {kb_id}")
    else:
        print(f"\nUsing existing KB: {kb_id}")

    # Step 2: Create or update Assistant
    config = build_assistant_config(kb_id=kb_id)

    if VAPI_ASSISTANT_ID:
        print(f"\nUpdating assistant: {VAPI_ASSISTANT_ID}")
        result = update_assistant(VAPI_ASSISTANT_ID, config)
        print(f"Updated: {result.get('id')}")
    else:
        assistants = list_assistants()
        existing = [a for a in assistants if a.get("name", "").startswith("SPARC")]
        if existing:
            aid = existing[0]["id"]
            print(f"\nUpdating existing assistant: {aid}")
            result = update_assistant(aid, config)
            print(f"Updated: {result.get('id')}")
        else:
            print("\nCreating assistant...")
            result = create_assistant(config)
            print(f"Created: {result.get('id')}")

    print(f"\n{'=' * 60}")
    print("Done! Add these to your .env:")
    print(f"  VAPI_ASSISTANT_ID={result.get('id')}")
    print(f"  VAPI_KB_ID={kb_id}")
    print(f"\nStart server: python -m backend.webhook_server")


if __name__ == "__main__":
    main()
