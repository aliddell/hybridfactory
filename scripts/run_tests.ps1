function Get-ProjectDirectory
{
  $Invocation = (Get-Variable MyInvocation -Scope 1).Value
  Split-Path (Split-Path $Invocation.MyCommand.Path)
}

$cwd=Get-Location
$projectdir=Get-ProjectDirectory
$anaconda="$env:HOMEPATH\Anaconda3\envs\hybridfactory\Scripts\"

Set-Location "$projectdir"

& "$anaconda\py.test.exe" -s --cov-conf test\.coveragerc --cov=hybridfactory test\

Set-Location "$cwd"
