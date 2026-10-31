param(
    [Parameter(ValueFromRemainingArguments = $true)]
    $RemainingArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

& "$ScriptDir\run.ps1" @RemainingArgs
