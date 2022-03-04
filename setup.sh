wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
eval "$(/home/ralves/miniconda3/bin/conda shell.bash hook)"
conda install mamba gh -c conda-forge -y
# gh auth login
gh repo clone ronaldokun/code-academy
# gh repo clone ronaldokun/fsdl
git config --global user.email "rsilva@anatel.gov.br"
git config --global user.name "Ronaldo S.A. Batista"
git submodule init
git submodule update
mamba env create -f code-academy/02-FRAMEWORKS/environment.yml
cd code-academy/fsdl && make conda-update && make pip-tools
mamba clean --all -y

