How to Run
To run or re-evaluate the advanced models at any time:

powershell
# Run LightGBM & CatBoost standalone (~55 seconds)
python src/train_and_evaluate_advanced_models.py
# Or run the entire pipeline with all 6 models included
python run_pipeline.py --include-advanced

# Launch Deployable Clinical System (Flask API + Angular Frontend at http://127.0.0.1:5000)
python api/app.py