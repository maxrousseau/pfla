if test -e $HOME/miniconda/bin; then
    echo "miniconda is already installed"
    export PATH=$HOME/miniconda/bin:$PATH
    conda update --yes --quiet conda
    if test -e 'source activate pfla'; then
        echo "The required environment exist"
        source activate pfla
        conda install --yes -c $HOME/miniconda/envs/pfla/conda-bld r-shapes
    else
        conda env create -f environment.yml
        source activate pfla
        mkdir shapes
        cd shapes
        conda skeleton cran --recursive shapes
        conda build r-shapes
        conda install --yes -c $HOME/miniconda/envs/pfla/conda-bld r-shapes
        cd ..
    fi
else
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda
    export PATH=$HOME/miniconda/bin:$PATH
    conda update --yes --quiet conda
    conda env create -f environment.yml
    source activate pfla
    mkdir shapes
    cd shapes
    conda skeleton cran --recursive shapes
    conda build r-shapes
    conda install --yes -c $HOME/miniconda/envs/pfla/conda-bld r-shapes
    cd ..
fi
