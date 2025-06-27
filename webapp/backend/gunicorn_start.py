#!/usr/bin/env python3
"""
Gunicorn startup script for Railway deployment
"""

import os
import multiprocessing

# Gunicorn configuration
bind = f"0.0.0.0:{os.getenv('PORT', 8000)}"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Application
wsgi_app = "integrated_main:app"

if __name__ == "__main__":
    from gunicorn.app.wsgiapp import run
    run() 