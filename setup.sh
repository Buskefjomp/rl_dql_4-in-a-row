# Doing all of this from scratch
python3.12 -m venv venv
echo "$(pwd)/src" > venv/lib/python3.12/site-packages/extend_src_path.pth
source venv/bin/activate
pip3 --update pip
pip3 install wheel pytest ruff pdbpp
pip3 install torch 
pip3 install tensorboard

# python 3.10
# python3 -m pip install -r requirements.txt

# python3 -m pip freeze > requiremenst.txt
