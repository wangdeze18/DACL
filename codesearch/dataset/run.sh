
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip



unzip java.zip

rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
