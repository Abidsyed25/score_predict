import pandas as pd
import numpy as np
from mymodelfile import MyModel
import os

if __name__=='__main__':

	# set path to training and test data files
	ball_by_ball_scores_file 	= r"/var/IPL_Ball_by_Ball_2008_2022.csv"
	match_results_file 			= r"/var/IPL_Matches_Result_2008_2022.csv"	
	test_data_file 				= r"/var/test_file.csv"

	# instantiate the model
	a_model = MyModel()

	# read training and test datasets
	ball_by_ball_scores_data = pd.read_csv(ball_by_ball_scores_file)
	match_results_data 		 = pd.read_csv(match_results_file)
	test_data 				 = pd.read_csv(test_data_file)
	
	# train the model
	a_model.fit([ball_by_ball_scores_data,match_results_data])

	# make predictions
	predictions = a_model.predict(test_data)

	# print(os.listdir('/var'))
	# save predictions in appropriate format
	submissions_df = pd.DataFrame(data = predictions, 
								  columns=['predicted_runs'])
	submissions_df.to_csv('/var/submission.csv', index_label='id')