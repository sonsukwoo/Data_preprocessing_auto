VENV?=.venv
PYTHON?=$(VENV)/bin/python
REQUEST?=https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv 데이터를 전처리해줘
MAX_ITERATIONS?=3
TESTS?=tests

.PHONY: bootstrap run test serve

bootstrap:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	PYTHONPATH=src $(PYTHON) -m data_preprocessing.cli --request "$(REQUEST)" --max-iterations $(MAX_ITERATIONS)

test:
	PYTHONPATH=src $(PYTHON) -m pytest $(TESTS)

serve:
	PYTHONPATH=src $(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
