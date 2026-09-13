.PHONY: install data features train train-lgbm predict test report clean

install:
	pip install -r requirements.txt

data:
	python src/data/make_dataset.py
	python src/data/make_final_dataset.py

train:
	python src/models/train_model.py --model xgboost

train-lgbm:
	python src/models/train_model.py --model lgbm

predict:
	python src/models/predict_model.py --model xgboost

test:
	pytest tests/ -v

report:
	python scripts/generate_report.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
