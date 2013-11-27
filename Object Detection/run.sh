j=0
#for ((i =0; i< 12;i++));do
#./a.out images/obama$i.jpg ; mv output.jpg outputs/obama$i.jpg
#done
if [[ $# -ne 1 ]]; then
	t=30
else
	t=$1
fi
#t=$1
echo "threshold: $t"
for i in `ls dartboards/dart*.jpg`; do
new_name=$(basename $i)
output=$(./a.out $i $t ; mv output.jpg dart_out/${new_name})
filename=$(basename $i)
echo "$filename; $output"
done
