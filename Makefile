.PHONY: init test format lint

init:
	# install lfs
	git lfs install

create_env:
	@echo "Creating virtual environment..."
	bash scripts/create_env.sh

update_output_dir:
	@echo "Updating output directory..."
	git lfs track outs/**
	git add .gitattributes

train:
	@echo "Training model..."
	python -m src.exercise_01.train

evaluate: 
	@echo "Evaluating model..."
	python -m src.exercise_01.evaluate



format:
	ruff src 

lint:
	ruff check src 