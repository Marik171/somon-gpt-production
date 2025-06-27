web: cd webapp/backend && PYTHONPATH=/app/webapp/backend gunicorn integrated_main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120 --access-logfile - --error-logfile -
