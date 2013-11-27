j=0
#for ((i =0; i< 12;i++));do
#./a.out images/obama$i.jpg ; mv output.jpg outputs/obama$i.jpg
#done
t=30
#t=$1
echo "threshold: $t"
for i in `ls training/dart*.jpg`; do
new_name=$(basename $i)
./a.out $i $t ; mv output.jpg dart_out/${new_name}
done
