.PHONY: help install-dev test test_agent test_solver lint format check_llm autoresearch_loop strategy_sweep tool_calling_demo replay_demo capture_replay_fixture render_graph graph_dev

PYTHON ?= python
PYTHON_FILES=src tests
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))

help:
	@echo '----'
	@echo 'install-dev  - install package and development dependencies'
	@echo 'test         - run all local tests'
	@echo 'test_agent   - run AutoResearch agent tests'
	@echo 'test_solver  - run solver-boundary tests'
	@echo 'lint         - run ruff and mypy'
	@echo 'format       - format Python files'
	@echo 'check_llm    - call the configured agent-side LLM once'
	@echo 'autoresearch_loop - run the bounded local AutoResearch loop'
	@echo 'strategy_sweep - generate and evaluate local-improve variants'
	@echo 'tool_calling_demo - replay a small local tool-calling trace'
	@echo 'replay_demo  - replay a saved no-key AutoResearch trajectory'
	@echo 'capture_replay_fixture - refresh replay fixture from a live LLM run'
	@echo 'render_graph - print README Mermaid graph diagrams'
	@echo 'graph_dev    - start LangGraph dev server'

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests

test_agent:
	$(PYTHON) -m pytest tests/unit_tests

test_solver:
	$(PYTHON) -m pytest tests/solver_tests

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format $(PYTHON_FILES) --diff
	$(PYTHON) -m mypy --strict src

format:
	$(PYTHON) -m ruff format $(PYTHON_FILES)
	$(PYTHON) -m ruff check --select I --fix $(PYTHON_FILES)

check_llm:
	$(PYTHON) scripts/check_llm.py

autoresearch_loop:
	ENABLE_LLM_RECOMMENDATIONS=false $(PYTHON) scripts/run_research_loop.py

strategy_sweep:
	$(PYTHON) scripts/run_strategy_sweep.py

tool_calling_demo:
	$(PYTHON) scripts/run_tool_calling_demo.py

replay_demo:
	$(PYTHON) -m autoresearch_agent.replay examples/sample_experiment_log.json

capture_replay_fixture:
	$(PYTHON) scripts/capture_replay_fixture.py

render_graph:
	$(PYTHON) scripts/render_agent_graph.py

graph_dev:
	$(PYTHON) -m langgraph dev
