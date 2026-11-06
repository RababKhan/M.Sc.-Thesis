# Measure rendered heights of the content tables and algorithm listings of final_thesis_MIST.docx (read-only) for the
# MIST length audit. Table kinds come from tools/_docx_build_log.json (same creation order as Word's Tables collection).
param([string]$Dir = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)), [int]$FirstMainPage = 24)
$ErrorActionPreference = "Stop"
$docx = Join-Path $Dir "final_thesis_MIST.docx"
$log = Get-Content (Join-Path $Dir "tools\_docx_build_log.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$out = Join-Path $Dir "tools\_table_heights.json"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $d = $word.Documents.Open($docx, $false, $true)
    $d.Repaginate()
    $n = $d.Tables.Count
    Write-Output "tables in Word: $n; in build log: $($log.all_tables.Count)"
    $res = @()
    $labels = @($log.all_tables | Where-Object { $_.kind -eq "content" } | ForEach-Object { $_.label })
    $ci = 0
    for ($i = 1; $i -le $n; $i++) {
        $t = $d.Tables.Item($i)
        if ($t.Range.OMaths.Count -gt 0) { continue }                # display equations
        if ($t.Rows.Item(1).Range.Information(3) -lt $FirstMainPage) { continue }   # front-matter layout tables
        if ($t.Rows.Count -eq 1 -and $t.Columns.Count -eq 1) { $kind = "code"; $lab = "algorithm" }
        else { $kind = "content"; $lab = if ($ci -lt $labels.Count) { $labels[$ci] } else { "?" }; $ci++ }
        $r1 = $t.Rows.Item(1).Range
        $rn = $t.Rows.Item($t.Rows.Count).Range
        $res += [pscustomobject]@{ index = $i; kind = $kind; label = $lab;
            p1 = $r1.Information(3); y1 = $r1.Information(6); pn = $rn.Information(3); yn = $rn.Information(6);
            land = ($t.Range.Sections.Item(1).PageSetup.Orientation) }
        if ($res.Count % 10 -eq 0) { Write-Output ("{0:HH:mm:ss} measured {1}" -f (Get-Date), $res.Count) }
    }
    $res | ConvertTo-Json -Depth 3 | Out-File -FilePath $out -Encoding utf8
    Write-Output "measured $($res.Count)"
    $d.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
