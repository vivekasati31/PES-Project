import subprocess
import time
from pyngrok import ngrok

# Start Streamlit on localhost:8502
print("Starting Streamlit app on port 8502...")
streamlit_process = subprocess.Popen(
    ["python", "-m", "streamlit", "run", "app.py", "--server.port=8502", "--server.headless=true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for Streamlit to start
time.sleep(5)

# Create ngrok tunnel
print("Creating ngrok tunnel...")
public_url = ngrok.connect(8502, "http")
print("\n" + "="*80)
print("✓ PUBLIC URL (share this worldwide):")
print(f"  {public_url}")
print("="*80)
print("\nStreamlit is running. Press Ctrl+C to stop.\n")

# Keep running
try:
    streamlit_process.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    streamlit_process.terminate()
    ngrok.kill()
