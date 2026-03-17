#call this shell using: source create_env.sh
# deactivate old environment
# source deactivate
# delete old environment
rm env
#Create virtual environment
python -m venv env 
#Activate environment
source env/bin/activate
#update pip
python -m pip install --upgrade pip
#install requirements
python -m pip install -e .[dev]