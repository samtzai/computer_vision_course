python3 -m venv env 
echo Activate environment
Call ./env/Scripts/activate.bat
echo update pip
pip install --upgrade pip
echo install requirements
pip install -e .