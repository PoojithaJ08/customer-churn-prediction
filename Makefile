.PHONY: data train test score dashboard all clean

## Generate synthetic customer data
data:
	python data/generate_data.py

## Train the Random Forest model
train:
	python src/train.py

## Run unit tests
test:
	pytest tests/ -v

## Score all customers (outputs data/scores.csv)
score:
	python src/predict.py --input data/customers.csv --output data/scores.csv

## Launch Streamlit dashboard at localhost:8501
dashboard:
	streamlit run dashboard/app.py

## Run full pipeline: data → train → test → score
all: data train test score
	@echo "✅ Pipeline complete"

## Remove generated files
clean:
	rm -f data/customers.csv data/scores.csv
	rm -f models/churn_model.pkl models/feature_names.pkl models/metrics.json

help:
	@echo "Available commands:"
	@echo "  make data       Generate synthetic customer data"
	@echo "  make train      Train the Random Forest model"
	@echo "  make test       Run unit tests"
	@echo "  make score      Score all customers"
	@echo "  make dashboard  Launch Streamlit dashboard"
	@echo "  make all        Run full pipeline"
	@echo "  make clean      Remove generated files"
