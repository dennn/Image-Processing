
t=$1
m=$2

for i in `ls dartboards/dart*.jpg`; do
    new_name=$(basename $i)
    output=$(./a.out $i $t $m; mv output.jpg dart_out/${new_name})
    filename=$(basename $i)
    echo "$filename; $output"
done
