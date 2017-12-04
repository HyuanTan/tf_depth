#!/bin/bash

#src=$1
des=$1

# Step 1. Generate mask label & copy data
#echo "Generating label and copy data"
#if [ ! -d $des/sample ]
#then
    #mkdir -p $des/sample
#fi

#if [ ! -d $des/label ]
#then
    #mkdir -p $des/label
#fi

#ls $src/*.json | xargs -n1 basename > $src/tmp
#for f in `cat $src/tmp`
#do
    #echo $f
    #fname=${f%.*}
    ##if [ ! -d $src/${fname}_json ]; then
        ##labelme_json_to_dataset $src/$f
        ##if [ $? != 0 ];then echo "Failed!"; exit -1; fi
    ##fi

    #cp $src/$fname.png $des/sample/
    #if [ $? != 0 ];then echo "Failed!"; exit -1; fi

    #cp $src/${fname}_json/label.png $des/label/${fname}.png
    #if [ $? != 0 ];then echo "Failed!"; exit -1; fi

#done

#rm $src/tmp

cur_dir=`pwd`
ls $des/sample/*.png | xargs -n1 basename > $des/sample/tmp.txt
ls $des/label/*.png | xargs -n1 basename > $des/label/tmp.txt

# Step 2. Crop
echo "Cropping images"
python preprocess.py -f crop --infile $des/sample/tmp.txt --toprate 0.25
if [ $? != 0 ];then echo "Failed!"; exit -1; fi
python preprocess.py -f crop --infile $des/label/tmp.txt --toprate 0.25
if [ $? != 0 ];then echo "Failed!"; exit -1; fi

# Step 3. Resize
echo "Resizing images"
python preprocess.py -f resize --infile $des/sample/tmp.txt --height 300 --width 640
if [ $? != 0 ];then echo "Failed!"; exit -1; fi
python preprocess.py -f resize --infile $des/label/tmp.txt --height 300 --width 640
if [ $? != 0 ];then echo "Failed!"; exit -1; fi


# Step 4. Process mask label
echo "Preprocessing label images"
python preprocess.py -f label_proc --infile $des/label/tmp.txt
if [ $? != 0 ];then echo "Failed!"; exit -1; fi
rm $des/sample/tmp.txt
rm $des/label/tmp.txt

# Step 5. Split training/validation set
echo "Split training/validation set"
cd $des
mkdir val_sample val_label
val_num=$2
for f in `ls sample/*.png | sort -R | head -n $val_num | xargs -n1 basename`
do
    echo $f
    mv sample/$f val_sample/
    mv label/$f val_label/
    echo "val_sample/$f" >> val_sample.txt
    echo "val_label/$f" >> val_label.txt
done
paste -d' ' val_sample.txt val_label.txt > val.txt

cd $cur_dir

# Step 6. Augmentation
echo "Augmentation"
python preprocess.py -f augment --inpath $des/sample/ --maskpath $des/label/
if [ $? != 0 ];then echo "Failed!"; exit -1; fi

# finalize
cd $des
for f in `ls sample/`
do
    echo "sample/$f" >> sample.txt
    echo "label/$f" >> label.txt
done
paste -d' ' sample.txt label.txt > train.txt
cd $cur_dir


# Step 5. Pack
#echo "Packing"
#cd $des/../
#dir=`basename $des`
#if [ $? != 0 ];then echo "Failed!"; exit -1; fi

#tar cjf - $dir | split -b 100m - $dir.tar.bz2
