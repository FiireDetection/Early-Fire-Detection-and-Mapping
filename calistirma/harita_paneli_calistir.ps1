param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 5000,
    [switch]$DebugMode
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$scriptPath = Join-Path $Root "haritalama\panel.py"

$argsList = @(
    $scriptPath,
    "--host", $HostAddress,
    "--port", "$Port"
)

if ($DebugMode) {
    $argsList += "--debug"
}

python @argsList
