outname=$1.yml
# output a minimal set of packages required to recreate an environment
conda-minify --name $1  --how=major -f $outname
# conda-minify outputs dependencies seemingly in random order.
# let's always sort them alphabetically to minimise the diff.
yq eval '.dependencies |= sort' -i $outname

