wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
eval "$(/home/ralves/miniconda3/bin/conda shell.bash hook)"
conda install mamba gh -c conda-forge -y
pip install tensorflow-gpu
# gh auth login
# gh repo clone ronaldokun/code-academy
# git config --global user.email "rsilva@anatel.gov.br"
# git config --global user.name "Ronaldo S.A. Batista"
