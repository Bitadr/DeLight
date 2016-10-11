#!/bin/bash

rm Alpha*
rm test*
rm train*
rm valid*


#cp SAMPLE_DATA_FILE.pkl temp.pkl
#python Initial_weight.py "temp" NUMBER_OF_INFERENCE_CLASSES
#for i in 1;do
#	echo raw data main
#	echo $i
#	python DNN0.py dropout "LAYER_SIZES_SEPERATED_BY_COMMA" "BATCH_SIZE" "COMMA_SEPERATED_DROPOUT_RATE_OF_EACH_LAYER" "ACTIVATION_FUNCTIONS" n_epochs "temp.pkl" $i 1
#	python adaboost.py "temp" NUMBER_OF_INFERENCE_CLASSES "Alpha" 1
#	python Data_booster.py "temp" NUMBER_OF_INFERENCE_CLASSES
#done
#python final_classifier.py NUMBER_OF_INFERENCE_CLASSES "1" "temp" "0"

cp DAS_O.pkl temp.pkl
python Initial_weight.py "temp" 19
for i in 1;do
	echo raw data main
	echo $i
	python DNN0.py dropout "5625, 2000, 500, 19" 100 "0, 0.5, 0.5" "0, 0" 7000 "temp.pkl" $i 1
	python adaboost.py "temp" 19 "Alpha" 1
	python Data_booster.py "temp" 19
done
python final_classifier.py 19 "1" "temp" "0"

cp DAS_V19x456.pkl temp.pkl
python Initial_weight.py "temp" 19
for i in 1;do
	echo transformed data main
	echo $i
	python DNN0.py dropout "456, 500, 100, 19" 100 "0, 0.5, 0.5" "0, 0" 7000 "temp.pkl" $i 2
	python adaboost.py "temp" 19 "Alpha" 2
	python Data_booster.py "temp" 19
	python final_classifier.py 19 "1" "temp" "1"
done

	
cp DAS_V19x456.pkl temp.pkl
python Initial_weight.py "temp" 19
for i in {1..6};do
	echo transformed data adaboost 1
	echo $i
	python DNN0.py dropout "456, 500, 19" 100 "0, 0.5" "0" 25 "temp.pkl" $i 5
	python adaboost.py "temp" 19 "Alpha" 5
	python Data_booster.py "temp" 19
done


cp DAS_V19x456.pkl temp.pkl
python Initial_weight.py "temp" 19
for i in {1..6};do
	echo transformed data adaboost 2
	echo $i
	python DNN0.py dropout "456, 200, 19" 100 "0, 0.5" "0" 25 "temp.pkl" $i 6
	python adaboost.py "temp" 19 "Alpha" 6
	python Data_booster.py "temp" 19
done

python final_classifier.py 19 "6, 6" "temp" "4, 5"