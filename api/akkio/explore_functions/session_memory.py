import threading
from collections import defaultdict
from typing import List, Dict, Optional

# Session memory management - only keep last message for accuracy
SESSION_MEMORY: Dict[str, Dict[str, str]] = defaultdict(dict)
SESSION_MEMORY_LOCK = threading.Lock()


def manage_session_memory(session_id: str, user_message: Optional[str] = None, bot_message: Optional[str] = None, get_history: bool = False):
    """
    Centralized session memory management - only keeps the last message for accuracy
    """
    with SESSION_MEMORY_LOCK:
        if get_history:
            session_data = SESSION_MEMORY.get(session_id, {})
            last_user_msg = session_data.get("last_user_message")
            if last_user_msg:
                history = [{"role": "user", "content": last_user_msg}]
                print(f"Retrieved last user message for session {session_id}: {last_user_msg[:100]}...")
                return history
            else:
                print(f"No previous message found for session {session_id}")
                return []
        
        if user_message is not None:
            SESSION_MEMORY[session_id]["last_user_message"] = user_message
            print(f"Stored last user message for session {session_id}: {user_message[:100]}...")
        
        if bot_message is not None:
            SESSION_MEMORY[session_id]["last_bot_message"] = bot_message
            print(f"Stored last bot message for session {session_id}: {bot_message[:100]}...")
        
        return None







