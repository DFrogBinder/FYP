export DYLD_LIBRARY_PATH='/Users/boyanivanov/Elastix/lib':$DYLD_LIBRARY_PATH
output=$(eval "transformix -def all -out $1 -tp $2")

