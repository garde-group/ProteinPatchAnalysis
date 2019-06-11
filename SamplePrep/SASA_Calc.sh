#! /bin/bash

# Script to run everything!
# Source environment/bashrc
source ~/.bashrc
source activate /home/cbilodeau/anaconda2/envs/deepchem2

module load gromacs/453

curr=`pwd`
raw=$curr/1raw_pdb
clean=$curr/2clean_pdb
temp=$curr/temp  # Files will be deleted

rm -rf $temp
mkdir $temp

cd $raw
for i in *.pdb
do
    pdb="$i"
    fname=${pdb%.*}

    rm -rf input
    echo '0' >> input
    echo '0' >>input

    g_sas_s -f $i -s $i -or $fname-clean-resarea.xvg < input
    cp $fname-clean-resarea.xvg $clean
    rm -rf input
done
