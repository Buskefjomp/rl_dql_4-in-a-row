# Doing all of this from scratch
python3.10 -m venv venv
source venv/bin/activate
echo "$(pwd)/src" > venv/lib/python3.10/site-packages/extend_src_path.pth
pip3 pip3 --update pip
pip3 install wheel pytest ruff pdbpp
pip3 install torch 


# python 3.10
# python3 -m pip install -r requirements.txt

# python3 -m pip freeze > requiremenst.txt