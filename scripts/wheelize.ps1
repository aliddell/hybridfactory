function Get-ProjectDirectory
{
  $Invocation = (Get-Variable MyInvocation -Scope 1).Value
  Split-Path (Split-Path $Invocation.MyCommand.Path)
}

$cwd=Get-Location
$projectdir=Get-ProjectDirectory

$python="$env:HOMEPATH\Anaconda3\envs\hybridfactory\python.exe"

Set-Location "$projectdir"

& "$python" setup.py sdist bdist_wheel

Set-Location "$cwd"
