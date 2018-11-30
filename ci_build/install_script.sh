if test -e $HOME/miniconda/bin; then
    echo "miniconda is already installed"
    conda update --yes --quiet conda
    if test -e 'conda activate pfla'; then
        echo "The required environment exist"
        conda activate pfla
    else
        conda env create -f environment.yml
        conda activate pfla
        mkdir shapes
        cd shapes
        conda skeleton cran --recursive shapes
        conda build r-shapes
        conda install --yes -c /home/travis/miniconda3/envs/pfla/conda-bld r-shapes
        cd ..
    fi
else
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda
    export PATH=$HOME/miniconda/bin:$PATH
    conda update --yes --quiet conda
    conda env create -f environment.yml
    conda activate pfla
    mkdir shapes
    cd shapes
    conda skeleton cran --recursive shapes
    conda build r-shapes
    conda install --yes -c /home/travis/miniconda3/envs/pfla/conda-bld r-shapes
    cd ..
fi
