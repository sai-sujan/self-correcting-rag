import sys
import importlib.util

# Find where langgraph is installed
spec = importlib.util.find_spec("langgraph")
print(f"langgraph location: {spec.origin}")
print(f"langgraph path: {spec.submodule_search_locations}")

# Check graph module
try:
    from langgraph import graph
    print("\n✓ langgraph.graph found")
    print("graph contents:", [x for x in dir(graph) if not x.startswith('_')])
except Exception as e:
    print(f"\n✗ Error: {e}")

# Check if there's a different way to import
try:
    import langgraph.prebuilt
    import pkgutil
    print("\nlanggraph.prebuilt submodules:")
    for importer, modname, ispkg in pkgutil.iter_modules(langgraph.prebuilt.__path__):
        print(f"  - {modname}")
except Exception as e:
    print(f"Error listing submodules: {e}")