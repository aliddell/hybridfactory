#!/bin/bash

ME="$(readlink -f $0)"
PROJECTDIR="$(dirname $ME)/.."

ANACONDA="$HOME/anaconda3/envs/hybridfactory/bin"
$ANACONDA/sphinx-apidoc -f -o "$PROJECTDIR/docs/source" "$PROJECTDIR"

cd $PROJECTDIR/docs
SPHINXBUILD="$ANACONDA/sphinx-build" make html
cd $PROJECTDIR/scripts

#Set-Location "$projectdir\docs"
#$env:SPHINXBUILD="$anaconda\sphinx-build.exe"
#& .\make.bat html

#Set-Location "$projectdir\scripts"