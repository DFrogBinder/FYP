export DYLD_LIBRARY_PATH='/Users/boyanivanov/Elastix/lib':$DYLD_LIBRARY_PATH
output=$(eval "elastix -m $1 -f $2 -out $3 -p $4") 
echo "$output" 

