for dir in ../data/validation/node*; do
	python predict.py -i $dir -v
	done
