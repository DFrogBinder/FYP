export DYLD_LIBRARY_PATH='/Users/boyanivanov/Elastix/lib':$DYLD_LIBRARY_PATH
while getopts m:f:out:p: flag
do
	case "${flag}" in
		m) moving_image_path=${OPTARG};;
		f) fixed_image_path=${OPTARG};;
		out) output_dir=${OPTARG};;
		p) parameter_file=${OPTARG};;
	esac
done

echo $parameter_file


output=$(eval "elastix -m $moving_image_path -f $fixed_image_path -out $output_dir -p $parameter_file")
echo "$output" 

