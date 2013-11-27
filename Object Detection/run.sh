for i in `ls dartboards/dart*.jpg`; do
    new_name=$(basename $i)
    output=$(./a.out $i $t ; mv output.jpg dart_out/${new_name})
    filename=$(basename $i)
    echo "$filename; $output"
done
