.PHONY: setup test test-all run eval

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest tests/test_tools.py tests/test_slack.py tests/test_server.py -v

test-all:
	.venv/bin/pytest tests/ -v

run:
	.venv/bin/uvicorn app.server:app --port 3000

eval:
	.venv/bin/python eval.py
