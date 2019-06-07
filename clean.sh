#! /bin/bash

# clean pdb file from raw_pdb folder and place into pdb folder
# 
# remove waters
# fix hydrogens

cd raw_pdb
for i in *
do
        pdb="$i"
        fname=${pdb%.*}
        cfname="-clean.pdb"
#        echo $pdb
#        echo $fname"-clean.pdb"
        pdb4amber -i $i -o "../temp/"$fname"-clean.pdb" --dry --reduce
done
cd ../
mv temp/*-clean.pdb* pdb/.
cd pdb
for i in *
do
# remove heteroatoms
        sed -i '/HETATM/d' $i
done
