import sys
import os

# Add backend directory to sys.path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.main import app

if __name__ == "__main__":
    import uvicorn
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
    except (ValueError, TypeError):
        port = 8000

    print(f"🚀 Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
