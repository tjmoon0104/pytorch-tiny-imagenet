echo "Testing..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm -r ./tiny-imagenet-200/test
python3 val_format.py
find . -name "*.txt" -delete
mkdir models
cp -r tiny-imagenet-200 tiny-224
python3 resize.py