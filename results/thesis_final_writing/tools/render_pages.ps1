# Render every page of final_thesis_MIST.pdf to PNG with the built-in Windows.Data.Pdf renderer (visual QA only).
#   powershell -NoProfile -ExecutionPolicy Bypass -File render_pages.ps1 -Out <dir> [-Width 600] [-From 1] [-To 999]
param([string]$Out, [int]$Width = 600, [int]$From = 1, [int]$To = 9999,
      [string]$Pdf = (Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "final_thesis_MIST.pdf"))
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$m = [System.WindowsRuntimeSystemExtensions].GetMethods()
$asTaskOp = ($m | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$asTaskAct = ($m | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' })[0]
function Await($op, $type) { $t = $asTaskOp.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
function AwaitAct($a) { $t = $asTaskAct.Invoke($null, @($a)); $t.Wait(-1) | Out-Null }
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.StorageFolder, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime] | Out-Null
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($Pdf)) ([Windows.Storage.StorageFile])
$doc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
$folder = Await ([Windows.Storage.StorageFolder]::GetFolderFromPathAsync((Resolve-Path $Out).Path)) ([Windows.Storage.StorageFolder])
$n = $doc.PageCount
$last = [Math]::Min($To, $n)
for ($i = $From; $i -le $last; $i++) {
    $page = $doc.GetPage($i - 1)
    $f = Await ($folder.CreateFileAsync(("page_{0:D3}.png" -f $i), [Windows.Storage.CreationCollisionOption]::ReplaceExisting)) ([Windows.Storage.StorageFile])
    $s = Await ($f.OpenAsync([Windows.Storage.FileAccessMode]::ReadWrite)) ([Windows.Storage.Streams.IRandomAccessStream])
    $o = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $o.DestinationWidth = $Width
    AwaitAct ($page.RenderToStreamAsync($s, $o))
    $s.Dispose()
    $page.Dispose()
}
Write-Output "pages=$n rendered=$($last - $From + 1)"
