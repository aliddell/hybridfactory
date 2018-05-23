$projectdir=Split-Path -Path (Get-Item -Path ".\" -Verbose).FullName -Parent
$anaconda="$env:HOMEPATH\Anaconda3\envs\hybridfactory\Scripts\"

& "$anaconda\sphinx-apidoc.exe" -f -o "$projectdir\docs\source" "$projectdir"
Set-Location "$projectdir\docs"
$env:SPHINXBUILD="$anaconda\sphinx-build.exe"
& .\make.bat html

Set-Location "$projectdir\scripts"