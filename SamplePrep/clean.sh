#! /bin/bash
source ~/.bashrc

# clean pdb file from raw_pdb folder and place into clean_pdb folder
# 
# Remove waters
# Fix hydrogens
# Remove heteroatoms

curr=`pwd`
raw=$curr/1raw_pdb
clean=$curr/2clean_pdb
temp=$curr/temp  # Files will be deleted

rm -rf $temp
mkdir $temp


## Remove water:
cd $raw
for i in *.pdb
do
    pdb="$i"
    fname=${pdb%.*}
        pdb4amber -i $raw/$i -o $temp/$fname"-clean-nohyd.pdb" --dry
	reduce $temp/$fname"-clean-nohyd.pdb" > $temp/$fname"-clean.pdb"
done

cd $curr
mv $temp/*-clean.pdb $clean
cd $clean

## Remove heteroatoms:
for i in *
do
        sed -i '/HETATM/d' $i
done
