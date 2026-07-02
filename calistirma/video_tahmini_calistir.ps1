param(
    [string]$RgbVideo = "",
    [string]$TermalVideo = "",
    [string]$Model = "model_egitimi\models\dual_branch.pt",
    [string]$Cikti = "outputs\video_tahminleri.csv",
    [int]$Boyut = 224,
    [int]$KareAdimi = 10
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir

if (-not $RgbVideo -and -not $TermalVideo) {
    throw "En az bir video yolu verilmeli: -RgbVideo veya -TermalVideo"
}

if ($RgbVideo -and -not (Test-Path -LiteralPath $RgbVideo)) {
    throw "RGB video bulunamadi: $RgbVideo"
}

if ($TermalVideo -and -not (Test-Path -LiteralPath $TermalVideo)) {
    throw "Termal video bulunamadi: $TermalVideo"
}

if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model dosyasi bulunamadi: $Model"
}

$scriptPath = Join-Path $Root "video_cikarim\video_tahmin.py"
$argsList = @(
    $scriptPath,
    "--model", $Model,
    "--cikti", $Cikti,
    "--boyut", "$Boyut",
    "--kare-adimi", "$KareAdimi"
)

if ($RgbVideo) {
    $argsList += @("--rgb-video", $RgbVideo)
}

if ($TermalVideo) {
    $argsList += @("--termal-video", $TermalVideo)
}

python @argsList
