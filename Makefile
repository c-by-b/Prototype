# C-by-B Prototype Makefile
# Nathan runs experiments via make.

PYTHON ?= venv/bin/python
PROMPT ?= "Find optimal coordinates for offshore wind farm near Cape Cod"

.PHONY: run test serve clean

# Run the full pipeline via CLI with a prompt
run:
	$(PYTHON) -m cbyb.coordinator.safety_socket --prompt $(PROMPT)

# Run the Flask web server
serve:
	$(PYTHON) -m cbyb.app

# Run unit tests (no model loading required)
test:
	$(PYTHON) -m pytest tests/ -v

# Run a quick import check
check:
	$(PYTHON) -c "from cbyb.coordinator.contract import Contract, ContractManager; print('imports OK')"

# Clean results (careful — timestamped, but still)
clean:
	@echo "Cleaning results/ directory"
	rm -f results/*.json results/*.jsonl
