# server/run_opendap_server.py
import os
import sys
import subprocess

DEFAULT_PORT = os.environ.get("OPENDAP_PORT", "8000")
HOST_BIND = os.environ.get("OPENDAP_HOST", "0.0.0.0")  # listen on all interfaces by default

if __name__ == "__main__":
    # Command to start Gunicorn with  workers and threads config if needed
    command = [
        "gunicorn",
        "--bind", f"{HOST_BIND}:{DEFAULT_PORT}",
        "--workers", "4",            # e.g., 4 workers (can adjust or make configurable)
        "--threads", "1",            # threads per worker (PyDAP is mostly I/O bound, can also use gevent workers)
        "server.opendap.app:app"       # module:object pointing to our WSGI app
    ]
    print(f"Starting OPeNDAP server at {HOST_BIND}:{DEFAULT_PORT} serving data from 'data/opendap/'...")
    sys.exit(subprocess.call(command))
