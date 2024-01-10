1. This is the description of files in the project
	requirements.txt - containing project dependencies to install
	Calibration.ipynb - the work that was done, includes empirical test, visualizations and trial and error code 
	project_config.json - Input arguments for final calibration script
	calibration_model.py - The final model to run on test data

2. Please download, install python and add it to path if not done already. Having jupyter installed is not a requirement but a plus. If jupyter is not installed you can view the experiment, trial and error script (Calibration.ipynb) in VSCode or Google Colab.

2. Please install all required libraries in requirements.txt by running the following command
		pip install -r requirements.txt

3. Please save calibration_model.py and project_config.json in the same directory and run the following command
		python calibration_model.py --config project_config.json

4. Please make sure the arguments are correctly specified in project_config.json, here is a description of the arguments. The default arguments are already specified, so the program will work without issues if you dont change them

"start_date" - Please specify the start date of the QR document
"end_date" - Please specify the end date of the QR document, should be ahead of the start date
"split_date" - Please specify the date where you want to split the data into training and test samples
"period" - Please specify periods, acceptable values - "morning", "midday", "afternoon"
"directory" - Please specify the directory containing the QR documents
"hardcode_imp_features"- Setting this to false will rerun the GBM feature selection which takes around 2 hours for 60 to 90 dates, acceptable values = true, false
"Y" - Please select the dependent variable to calibrate, acceptable values - "Y1", "Y2"