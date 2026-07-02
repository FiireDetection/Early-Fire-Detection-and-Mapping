param(
    [string]$Csv = "outputs\video_tahminleri.csv",
    [double]$Threshold = 0.5,
    [string]$Latitude = "",
    [string]$Longitude = "",
    [string]$Altitude = "",
    [switch]$ClearTable
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir

if (-not (Test-Path -LiteralPath $Csv)) {
    throw "Tahmin CSV bulunamadi: $Csv"
}

$scriptPath = Join-Path $Root "haritalama\csv_postgis_yaz.py"
$argsList = @(
    $scriptPath,
    "--csv", $Csv,
    "--threshold", "$Threshold"
)

if ($Latitude) {
    $argsList += @("--latitude", $Latitude)
}

if ($Longitude) {
    $argsList += @("--longitude", $Longitude)
}

if ($Altitude) {
    $argsList += @("--altitude", $Altitude)
}

if ($ClearTable) {
    $argsList += "--clear-table"
}

python @argsList
