
import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from app_builder.graph.builder_graph import app_builder_graph
from app_builder.schemas.requirements import UserRequirement

async def test_graph():
    print("Testing graph compilation...")
    try:
        # Just check if it compiled
        if app_builder_graph:
            print("Graph compiled successfully!")
        
        # Test a small piece of the graph (structuring)
        # We won't run the full graph as it calls LLMs
        pass
    except Exception as e:
        print(f"Graph test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_graph())
