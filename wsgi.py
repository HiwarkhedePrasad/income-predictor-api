# wsgi.py
import sys
import os

# Add your project directory to the sys.path
# This makes sure your 'main.py' module can be found
# Replace 'your-username' and 'your-project-path'
path = '/home/your-username/your-project-path' # e.g., '/home/myusername/income_predictor_api'
if path not in sys.path:
    sys.path.insert(0, path)

# Import your FastAPI app instance
# Assuming your FastAPI app is named 'app' in 'main.py'
from main import app

# Import the asgi_wsgi adapter
from asgi_wsgi import WsgiToAsgi

# Create the ASGI application
asgi_app = app

# Wrap the ASGI app with WsgiToAsgi to create a WSGI callable
application = WsgiToAsgi(asgi_app)

# This is for basic debugging. It will print to your PythonAnywhere server log.
print("FastAPI app wrapped for WSGI via asgi_wsgi.")