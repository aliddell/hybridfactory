function Get-ProjectDirectory
{
  $Invocation = (Get-Variable MyInvocation -Scope 1).Value
  Split-Path (Split-Path $Invocation.MyCommand.Path)
}

$cwd=Get-Location
$projectdir=Get-ProjectDirectory

$anaconda="$env:HOMEPATH\Anaconda3\envs\hybridfactory\Scripts\"

& "$anaconda\sphinx-apidoc.exe" -f -o "$projectdir\docs\source" "$projectdir\hybridfactory"
Set-Location "$projectdir\docs"
$env:SPHINXBUILD="$anaconda\sphinx-build.exe"
& .\make.bat html

Set-Location "$cwd"
